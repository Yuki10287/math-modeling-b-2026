"""Local pilot: one/two-step task rollout from public feedback only.

Root window: at most two known sources and three pending fixed stations,
including the frozen policy's next task. Both horizons complete the same fixed
root task episode. The successor applies the original public routing rule to
the surviving root tasks after simulated observations; newly found sources
enter the NEXT real window, not a new service debt inside this local episode.
It never minimizes individual hidden-world costs. Both horizons use the same
worlds/actions/terminal proxy and transition CAP, not equal consumed work.

The existing geometric completion and optical execution remain unchanged.
Four eligible windows per run bound this pilot. No official client is added.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
sys.path.insert(0, str(Q4))
import ordered_optical_prune as baseline
from localization import optical_plan
from route_planning import open_route, route_length
from shared import core
from window_worlds_v1 import sample_worlds, predict_measure, predict_clear

MODEL_ID = 'window_joint_rollout_v1'
BASELINE_SHA256 = '2854b5d7e38ec5a56f9eb02bdbb126bb71b9c494030d0b3ae7e803e344d0f92a'
WORLD_COUNT = 6
MAX_TRANSITIONS = 60
MAX_PREDICTED_API_CALLS = 6000
MAX_WINDOW_DECISIONS = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':'), default=json_value).encode('utf-8')).hexdigest()


def record_measure(ledger, q, c, reply):
    ledger.append(dict(action='measure', position=np.asarray(q).tolist(),
                       channel=int(c), **copy.deepcopy(reply)))


def record_clear(ledger, q, c, reply):
    ledger.append(dict(action='clear', position=np.asarray(q).tolist(),
                       channel=int(c), **copy.deepcopy(reply)))


@dataclass
class PublicState:
    position: object
    channel: int
    beliefs: dict
    local_steps: dict
    cover: object
    discovered: set
    cleared: set
    ledger: list
    max_active_measures: int = 6


def public_payload(state):
    return dict(position=state.position, channel=state.channel,
        beliefs={str(c):dict(P=b.P, positives=b.positives, negatives=b.negatives,
            measured=b.measured, failed_clears=b.failed_clears, near=b.near)
            for c,b in sorted(state.beliefs.items())}, local_steps=state.local_steps,
        discovered=state.discovered, cleared=state.cleared, ledger=state.ledger,
        negative=state.cover.negative, covered=state.cover.covered,
        witnesses=state.cover.witnesses, max_active_measures=state.max_active_measures)


def public_digest(state):
    return digest(public_payload(state))


def unknown(state):
    if len(state.discovered) >= 16:
        return []
    return [c for c in range(1,21) if c not in state.discovered and not state.cover.complete(c)]


def tasks_and_points(state, fixed_tasks=None):
    pending = state.cover.needed_stations(unknown(state))
    tasks = [('scan',k) for k in pending]
    points = [state.cover.stations[k] for k in pending]
    for c,b in state.beliefs.items():
        tasks.append(('source',c))
        points.append(core.mec(b.P)[0])
    if fixed_tasks is not None:
        active = [(task,point) for task,point in zip(tasks,points) if task in fixed_tasks]
        tasks, points = [row[0] for row in active], [row[1] for row in active]
    return tasks, np.asarray(points, dtype=float).reshape((-1,2))


def baseline_task(state, fixed_tasks=None):
    """Public-state successor. No scenario/world/true-source argument exists."""
    tasks, points = tasks_and_points(state,fixed_tasks)
    return tasks[open_route(points,state.position)[0]] if tasks else None


def window_tasks(state, original=None):
    original = baseline_task(state) if original is None else tuple(original)
    tasks, points = tasks_and_points(state)
    selected = []
    for kind, limit in (('source',2),('scan',3)):
        group = sorted(((float(np.linalg.norm(p-state.position)), task)
                        for task,p in zip(tasks,points) if task[0] == kind),
                       key=lambda row:(row[0],row[1][1]))
        part = [task for _,task in group[:limit]]
        if original is not None and original[0] == kind and original not in part:
            part[-1:] = [original]
        selected.extend(part)
    # Stable ties preserve the original next task, not task-name lexical order.
    return ([original] if original in selected else []) + sorted(t for t in selected if t != original)


def terminal_value(state, fixed_tasks=None):
    """Public fixed-task-episode heuristic, NOT a full global cost/time bound.

    Original representative-point open route plus local optical score at each
    representative and the cost of testing the CURRENT unknown channel set at
    the remaining root stations. New discoveries never add local service debt.
    Future discoveries/service exits are not simulated in the terminal itself.
    """
    tasks, points = tasks_and_points(state,fixed_tasks)
    if not tasks:
        return dict(total_s=0., travel_s=0., service_s=0., scan_s=0.)
    order = open_route(points,state.position)
    travel = route_length(points,state.position,order)/5
    service, scan_fee, radio = 0., 0., state.channel
    channels = unknown(state)
    for j in order:
        kind, index = tasks[j]
        q = points[j]
        if kind == 'source':
            b = state.beliefs[index]
            safe = b.near if b.near is not None else core.nearest_certified_clear(b.P,q)
            service += float(np.linalg.norm(safe-q))/5+5 if safe is not None else optical_plan(b.P,q)['score']
        else:
            for c in sorted(channels,key=lambda c:(c != radio,c)):
                if any(np.linalg.norm(q-p) < 1e-7 for p in state.cover.negative[c]):
                    continue
                scan_fee += 5+int(c != radio)
                radio = c
    return dict(total_s=float(travel+service+scan_fee), travel_s=float(travel),
                service_s=float(service), scan_s=float(scan_fee))


class BudgetExceeded(RuntimeError):
    pass


class PredictionBudget:
    def __init__(self, cap=MAX_TRANSITIONS, api_cap=MAX_PREDICTED_API_CALLS):
        self.max_transitions, self.max_api_calls = cap, api_cap
        self.used_transitions = self.predicted_measure = self.predicted_clear = 0
        self.terminal_evaluations = 0

    def transition(self):
        if self.used_transitions >= self.max_transitions:
            raise BudgetExceeded('macro_transition_cap')
        self.used_transitions += 1

    def api_call(self, kind):
        if self.predicted_measure+self.predicted_clear >= self.max_api_calls:
            raise BudgetExceeded('predicted_api_cap')
        if kind == 'measure':
            self.predicted_measure += 1
        else:
            self.predicted_clear += 1

    def row(self):
        return vars(self).copy()


class PredictionAPI:
    """Private generative model; the copied solver sees public API methods only."""
    def __init__(self, state, world, budget):
        self.position = np.asarray(state.position).copy()
        self.channel = int(state.channel)
        self.removed = set(state.cleared)
        self.world, self.budget = world, budget
        self.cost_s = 0.
        self.events = []

    def _move(self,q):
        q = np.asarray(q,float)
        require(q.shape == (2,) and np.isfinite(q).all(), 'invalid predictive coordinate')
        self.cost_s += float(np.linalg.norm(q-self.position))/5
        self.position = q.copy()

    def measure(self,q,c):
        self.budget.api_call('measure')
        self._move(q)
        self.cost_s += 5+int(c != self.channel)
        self.channel = int(c)
        reply = predict_measure(self.world,q,c,cleared=self.removed)
        self.events.append(dict(action='measure',position=self.position.tolist(),channel=int(c),**reply))
        return reply

    def clear(self,q,c):
        self.budget.api_call('clear')
        self._move(q)
        reply = predict_clear(self.world,q,c,cleared=self.removed)
        success = reply['clear_result'] == 'success'
        self.cost_s += 5 if success else 3
        if success:
            self.removed.add(int(c))
        self.events.append(dict(action='clear',position=self.position.tolist(),channel=int(c),**reply))
        return reply


def _source_function():
    raw = Path(baseline.__file__).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BASELINE_SHA256,'frozen baseline changed')
    tree = ast.parse(raw.decode('utf-8-sig'))
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'solve_multi')


def _add_public_ledger(function):
    """Log accepted feedback; no alteration to control/geometry/fee conditions."""
    inner = {n.name:n for n in function.body if isinstance(n,ast.FunctionDef)}
    measure = inner['measure']
    index = next(i for i,n in enumerate(measure.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == 'kind')
    measure.body[index+1:index+1] = ast.parse('record_measure(planner_ledger, q, c, reply)').body
    clear = inner['clear']
    index = next(i for i,n in enumerate(clear.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == 'kind')
    clear.body[index+1:index+1] = ast.parse("record_clear(planner_ledger, q, c, {'clear_result': kind})").body


def _compile_macro():
    function = _source_function()
    _add_public_ledger(function)
    nested = [n for n in function.body if isinstance(n,ast.FunctionDef)]
    setup = ast.parse('''
cover, beliefs, local_steps = state.cover, state.beliefs, state.local_steps
cleared, discovered = state.cleared, state.discovered
planner_ledger = state.ledger
max_active_measures = state.max_active_measures
variant, trace, calls = 'share25', [], 0
''').body
    dispatch = ast.parse('''
kind, index = task
if kind == 'scan':
    if not scan(cover.stations[index].copy()):
        raise RuntimeError('predictive shared clear conflict')
else:
    status = service(index)
    if status == 'failed':
        raise RuntimeError('predictive optical or positive feedback conflict')
    if not share_at_stop(set()):
        raise RuntimeError('predictive service shared clear conflict')
state.position = np.asarray(api.position).copy()
state.channel = int(api.channel)
return trace
''').body
    node = ast.parse('def macro(state, api, task):\n    pass').body[0]
    node.body = setup+nested+dispatch
    namespace = baseline.solve_multi.__globals__.copy()
    namespace.update(record_measure=record_measure,record_clear=record_clear)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 str(Path(__file__)), 'exec'),namespace)
    return namespace['macro']


_MACRO = _compile_macro()


def transition(state,task,world,budget):
    budget.transition()
    successor = copy.deepcopy(state)
    api = PredictionAPI(successor,world,budget)
    trace = _MACRO(successor,api,task)
    return successor, api.cost_s, dict(events=api.events,
        optical_blocks=sum(row['phase'] == 'optical_plan' for row in trace))


def plan_window(state, depth=2, *, transition_cap=MAX_TRANSITIONS):
    require(depth in (1,2),'invalid rollout depth')
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    before = public_digest(state)
    original = baseline_task(state)
    roots = window_tasks(state,original)
    budget = PredictionBudget(cap=transition_cap)
    log = dict(phase='window_rollout_prediction',model=MODEL_ID,depth=depth,
        root_digest=before,root_tasks=[list(t) for t in roots],baseline_task=original,
        selected=original,applied=False,reason='window_not_eligible',rows=[],
        world_digest=None,world_count=0,model_assumption='Finite work worlds, not official probability.',
        successor_policy='Original open_route rule on surviving FIXED ROOT tasks, using public feedback.',
        terminal='Fixed-root-task route/optical/current-unknown proxy, not global remaining cost.',
        new_discovery_scope='Real/hypothetical beliefs update normally; new service tasks enter next real window.')
    try:
        if not state.beliefs or not state.cover.needed_stations(unknown(state)) or len(roots)<2:
            return original,log
        sample = sample_worlds(state.beliefs,state.cover,state.discovered,state.cleared,
                               state.ledger,count=WORLD_COUNT)
        worlds = sample['worlds']
        log['sampling'] = sample['diagnostics']
        log['world_count'] = len(worlds)
        log['world_digest'] = digest(worlds)
        if len(worlds) != WORLD_COUNT:
            log['reason'] = 'inadequate_predictive_support_original_policy'
            return original,log
        require(len(roots)*WORLD_COUNT*depth <= transition_cap,'insufficient budget for complete root round')
        selected, minimum = original, math.inf
        for root in roots:
            samples = []
            for index,world in enumerate(worlds):
                first,cost1,event1 = transition(state,root,world,budget)
                second_task, cost2, event2 = None, 0., None
                first_digest = public_digest(first)
                if depth == 2:
                    # Replanning depends on feedback, NEVER on world identity/position.
                    second_task = baseline_task(first,roots)
                    if second_task is not None:
                        end,cost2,event2 = transition(first,second_task,world,budget)
                    else:
                        end = first
                else:
                    end = first
                budget.terminal_evaluations += 1
                terminal = terminal_value(end,roots)
                samples.append(dict(world_index=index,first_cost_s=cost1,second_cost_s=cost2,
                    first_public_digest=first_digest,second_task=second_task,
                    first_events=event1,second_events=event2,terminal=terminal,
                    total_s=float(cost1+cost2+terminal['total_s'])))
            value = float(np.mean([row['total_s'] for row in samples]))
            log['rows'].append(dict(task=root,mean_s=value,samples=samples))
            if value < minimum-1e-8:
                selected,minimum = root,value
        log.update(selected=selected,applied=selected != original,reason='complete_root_comparison')
        return selected,log
    except (ValueError,RuntimeError,FloatingPointError) as error:
        # Partial rows are diagnostic only. Never choose the best completed subset.
        log.update(reason='prediction_rejected_original_policy',error=dict(
            type=type(error).__name__,message=str(error)),selected=original,applied=False)
        return original,log
    finally:
        require(before == public_digest(state),'real public state mutated by planner')
        log.update(budget=budget.row(),wall_s=time.perf_counter()-start_wall,
                   cpu_s=time.process_time()-start_cpu)


def _compile_solver():
    function = _source_function()
    _add_public_ledger(function)
    first_def = next(i for i,n in enumerate(function.body) if isinstance(n,ast.FunctionDef))
    function.body[first_def:first_def] = ast.parse('planner_ledger, planner_calls = [], 0').body
    loop = next(n for n in function.body if isinstance(n,ast.For))
    assignment = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
                      and ast.unparse(n.targets[0]) == '(task, index)')
    hook = ast.parse('''
if planner_calls < MAX_WINDOW_DECISIONS and beliefs and pending:
    state = PublicState(np.asarray(api.position).copy(),int(api.channel),beliefs,local_steps,
        cover,discovered,cleared,planner_ledger,max_active_measures)
    selected, prediction = plan_window(state,depth=rollout_depth)
    planner_calls += 1
    prediction['window_call'] = planner_calls
    prediction['event_prefix'] = calls
    trace.append(prediction)
    if selected is not None:
        task,index = selected
''').body
    loop.body[assignment+1:assignment+1] = hook
    namespace = baseline.solve_multi.__globals__.copy()
    namespace.update(record_measure=record_measure,record_clear=record_clear,
        PublicState=PublicState,plan_window=plan_window,MAX_WINDOW_DECISIONS=MAX_WINDOW_DECISIONS)
    # A closure-free depth argument keeps simultaneous d1/d2 runs independent.
    function.args.args.append(ast.arg(arg='rollout_depth'))
    function.args.defaults.append(ast.Constant(value=2))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(Path(__file__)), 'exec'),namespace)
    return namespace['solve_multi']


_SOLVER = _compile_solver()


def solve_multi(api,variant='share25',trace=None,max_active_measures=6,depth=2):
    require(variant == 'share25','this research pilot uses the frozen share25-prune candidate')
    require(depth in (1,2),'depth must be 1 or 2')
    return _SOLVER(api,variant=variant,trace=trace,max_active_measures=max_active_measures,
                   rollout_depth=depth)
