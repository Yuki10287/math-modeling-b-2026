"""One joint source/scan decision using history-compatible known sources."""
import copy
import hashlib
import json
import math
import time
import numpy as np

from closed_preview import state_key, decision
from conditional_worlds import conditional_worlds, clone_conditional, LedgerModel
from rollout_solver import continue_lean


def joint_candidates(tasks, route):
    selected = [route[0]]
    # Ensure the comparison can change between searching and servicing.
    other_kind = next((t for t in route if t['kind'] != route[0]['kind']), None)
    if other_kind is not None:
        selected.append(other_kind)
    for task in route[1:]:
        if len(selected) == 3:
            break
        if (task['kind'], task['key']) not in {(t['kind'], t['key']) for t in selected}:
            selected.append(task)
    return selected


def conditional_key(model, iteration, candidates, count):
    extra = json.dumps(dict(beliefs=model.beliefs, history=model.history), sort_keys=True,
                       default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
    return hashlib.sha256((state_key(model, iteration, candidates, count)+extra).encode()).hexdigest()


def conditional_forecast(model, iteration, candidates, count=6):
    worlds = conditional_worlds(model, count=count)
    if not worlds:
        return None
    started = time.perf_counter()
    rows = []
    for task in candidates:
        costs, errors, details = [], [], []
        for index, world in enumerate(worlds):
            field = ('smooth', 'extreme', 'hash')[index % 3]
            try:
                clone, arena = clone_conditional(model, world, field=field, scenario_index=index)
                outcome = continue_lean(clone, start_iteration=iteration, first_task=task)
                if not outcome['complete'] or not arena.evaluation()['all_cleared']:
                    raise RuntimeError(f'incomplete scenario: {outcome["status"]}')
                cost = float(arena.time_s)
                if not math.isfinite(cost):
                    raise ValueError('nonfinite scenario cost')
                costs.append(cost)
                errors.append(None)
                details.append(dict(counts=arena.counts.copy(), final_position=arena.position.tolist(), field=field))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                costs.append(None)
                errors.append(f'{type(exc).__name__}: {exc}')
                details.append(None)
        rows.append(dict(task=copy.deepcopy(task), costs_s=costs, errors=errors, scenarios=details))
    return dict(candidates=rows, scenario_count=len(worlds),
                world_counts=[len(w) for w in worlds], forecast_runtime_s=time.perf_counter()-started)


class ConditionalSelector:
    def __init__(self, *, scenario_count=6, risk=.2, margin_s=5., cache=None):
        self.scenario_count, self.risk, self.margin_s = scenario_count, risk, margin_s
        self.used = False
        self.cache = {} if cache is None else cache

    def __call__(self, model, iteration, tasks, route):
        if self.used or not model.beliefs or len(tasks) < 2:
            return None
        self.used = True
        candidates = joint_candidates(tasks, route)
        key = conditional_key(model, iteration, candidates, self.scenario_count)
        cached = key in self.cache
        if not cached:
            self.cache[key] = conditional_forecast(model, iteration, candidates, self.scenario_count)
        prediction = self.cache[key]
        chosen, scores = decision(prediction, self.risk, self.margin_s)
        model.trace.append(dict(phase='conditional_forecast', iteration=iteration,
            state_key=key, cached=cached, prediction=prediction, scores=scores,
            changed=chosen != 0, chosen_index=chosen, risk=self.risk, margin_s=self.margin_s,
            known_channels=sorted(model.beliefs), unavailable=prediction is None))
        return candidates[chosen] if chosen else None


def solve_conditional(api, *, trace=None, reference=False, selector=None):
    model = LedgerModel(api, [] if trace is None else trace, range_cuts=True, search_fraction=.30)
    result = continue_lean(model, selector=None if reference else (selector or ConditionalSelector()))
    result['experiment_variant'] = 'ledger_reference' if reference else 'conditional'
    return result
