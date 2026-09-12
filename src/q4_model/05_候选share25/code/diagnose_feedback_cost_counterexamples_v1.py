"""Prefix-only diagnosis of five saved joint-view counterexamples.

This is conditional model inspection, not a new solver run, performance test,
or feedback replay at alternative points. Predictor inputs contain only actual
preceding observations. Saved source truth is neither inspected nor passed on.
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
RESULTS = HERE.parent/'results'
sys.path.insert(0, str(Q4))
sys.path.insert(0, str(HERE))
from localization import Belief, choose_measure, optical_plan, samples
import ordered_optical_prune as baseline
import station_joint_view_experiment as old_model

INPUTS = (
    ('development', 54100, RESULTS/'station_joint_view_dev54000_v1/54100-boundary-hashed-station_joint_view.json'),
    ('holdout', 55101, RESULTS/'station_joint_view_holdout55000_v1/55101-boundary-hashed-station_joint_view.json'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    return dict(path=path.relative_to(PROJECT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def state_dict(belief):
    return dict(polygon=belief.P.tolist(), positives=[p.tolist() for p in belief.positives],
                negatives=[p.tolist() for p in belief.negatives], measured=[p.tolist() for p in belief.measured],
                failed_clears=[p.tolist() for p in belief.failed_clears],
                near=None if belief.near is None else belief.near.tolist())


def restore_prefix(events, trace, index):
    """Rebuild actual feedback state, then bind it to every recorded belief row."""
    prefix = trace[:index]
    actual = [t for t in prefix if t.get('phase') == 'actual_action']
    require([t['event'] for t in actual] == list(range(1, len(actual)+1)), 'nonconsecutive action prefix')
    by_event, count = {}, 0
    for row in prefix:
        if row.get('phase') == 'actual_action':
            count = row['event']
        elif row.get('phase') == 'belief':
            by_event.setdefault(count, []).append(row)
    beliefs = {}
    negatives = {c: [] for c in range(1, 21)}
    active = {c: 0 for c in range(1, 21)}
    position, current = np.zeros(2), 1
    matched = 0
    for number, logged in enumerate(actual, 1):
        event = events[number-1]
        result_key = 'measure_result' if event['action'] == 'measure' else 'clear_result'
        require((logged['action'], logged['channel'], logged['position'], logged['result']) ==
                (event['action'], event['channel'], event['position'], event[result_key]),
                'trace action differs from independent accepted event ledger')
        c, q = event['channel'], np.asarray(event['position'], float)
        if event['action'] == 'measure':
            current = c
            feedback = {k: event[k] for k in ('measure_result', 'svd_deg') if k in event}
            if feedback['measure_result'] == 'no_signal':
                negatives[c].append(q.copy())
                if c in beliefs:
                    beliefs[c].update(q, feedback)
            elif c not in beliefs:
                beliefs[c] = Belief(q, feedback)
                beliefs[c].negatives = [p.copy() for p in negatives[c]]
            else:
                beliefs[c].update(q, feedback)
            if logged['reason'] in ('source_measure', 'joint_source_measure'):
                active[c] += 1
        elif event[result_key] == 'success':
            beliefs.pop(c, None)
        elif c in beliefs:
            beliefs[c].failed_clears.append(q.copy())
        position = q.copy()
        for row in by_event.get(number, []):
            state = state_dict(beliefs[row['channel']])
            require(all(state[k] == row[k] for k in ('polygon', 'positives', 'negatives')),
                    'reconstructed belief differs from historical recorded geometry')
            matched += 1
    return dict(beliefs=beliefs, cover=SimpleNamespace(negative=negatives), position=position,
                current_channel=current, active=active, events_before=len(actual), belief_rows_matched=matched)


def selected_states():
    output = []
    for split, seed, path in INPUTS:
        payload = json.loads(path.read_text(encoding='utf-8'))
        # Deliberately project to public records before any diagnosis. No source,
        # hidden total, generator, arena or environment object reaches a helper.
        events, trace = payload['events'], payload['trace']
        del payload
        selected_count = 0
        for index, prediction in enumerate(trace):
            if prediction.get('phase') != 'station_joint_prediction' or prediction['selected'] is None:
                continue
            state = restore_prefix(events, trace, index)
            c = prediction['channel']
            b = state['beliefs'][c]
            plan = optical_plan(b.P, state['position'])
            action = choose_measure(b, state['position'], c, state['current_channel']) if state['active'][c] < 6 else None
            require(('measure' if action is not None and action['score'] < plan['score'] else 'optical') ==
                    prediction['original_action'], 'original action not recovered')
            reproductions = []
            selection = old_model.select_joint_measure(b, state['position'], c, state['current_channel'],
                action, plan, prediction['frozen_tasks'], prediction['frozen_points'],
                prediction['unknown_channels'], state['cover'], reproductions)
            require(reproductions == [prediction], 'entire historical prediction failed exact reproduction')
            actual = next(t for t in trace[index+1:] if t.get('phase') == 'actual_action')
            require(actual['reason'] == 'joint_source_measure' and actual['channel'] == c and
                    actual['event'] == state['events_before']+1 and actual['position'] == selection['q'].tolist(),
                    'selected q not bound to next accepted action')
            feedback_event = events[actual['event']-1]
            require(feedback_event['measure_result'] == actual['result'] == 'no_signal', 'not the intended negative counterexample')
            output.append(dict(split=split, seed=seed, input_path=path, trace_index=index,
                state=state, channel=c, belief=b, original_plan=plan, original_action=action,
                prediction=prediction, joint=selection, actual=actual,
                after_actual_belief=next(t for t in trace[index+1:] if t.get('phase') == 'belief' and t['channel'] == c)))
            selected_count += 1
        require(selected_count == (1 if split == 'development' else 4), 'counterexample inventory changed')
    return output


def independent_tail(forecast, start, P):
    """Scalar distance/cost reconstruction, not the predictor's vector routine."""
    if forecast['kind'] != 'ordered_optical':
        value = math.dist(start, forecast['clearpoint'])/5+5
        require(abs(value-forecast['mean_s']) < 1e-7 and abs(value-forecast['worst_s']) < 1e-7,
                'single-clear forecast fee mismatch')
        return dict(passed=True, kind=forecast['kind'], independently_recomputed_mean_s=value,
                    independently_recomputed_worst_s=value)
    path = [forecast['original_path'][i] for i in forecast['kept_indices']]
    distances = [math.dist(start if i == 0 else path[i-1], q)/5 for i, q in enumerate(path)]
    cumulative = [math.fsum(distances[:i+1]) for i in range(len(path))]
    worst = cumulative[-1]+3*(len(path)-1)+5
    # The candidate's finite position measure is deliberately held fixed here;
    # this checks its arithmetic, not whether it is the official distribution.
    hull = baseline.exact_hull(P)
    quadrature = [g for g in samples(np.asarray(forecast['conditional_polygon']))
                  if baseline.distance_squared(g, hull) == 0]
    costs, misses = [], 0
    for g in quadrature:
        first = next((i for i, q in enumerate(path) if math.dist(g, q) <= 20), None)
        if first is None:
            misses += 1
        else:
            costs.append(cumulative[first]+3*first+5)
    fallback = not quadrature or bool(misses)
    mean = worst if fallback else math.fsum(costs)/len(costs)
    require(abs(mean-forecast['mean_s']) < 1e-7 and abs(worst-forecast['worst_s']) < 1e-7,
            'independent optical tail cost mismatch')
    require(fallback == forecast['mean_fallback_to_worst'] and misses == forecast['sampled_uncovered'],
            'quadrature fallback mismatch')
    return dict(passed=True, kind='ordered_optical', independently_recomputed_mean_s=mean,
                independently_recomputed_worst_s=worst, quadrature_positions=[g.tolist() for g in quadrature],
                sampled_first_success_costs_s=costs, mean_fallback_to_worst=fallback)


def check_measure_formula(score, model):
    require(score is not None, 'fixed measured point unexpectedly inadmissible')
    positive = math.fsum(b['work_mass']*b['mean_s'] for b in score['positive_branches'])
    worsts = [b['worst_s'] for b in score['positive_branches']]
    negative = score['negative_forecast']
    if negative is not None:
        positive += score['negative_work_mass']*negative['mean_s']
        worsts.append(negative['worst_s'])
    risk = max(worsts)
    value = score['immediate_s']+model.MEAN_WEIGHT*positive+model.RISK_WEIGHT*risk
    require(abs(value-score['score']) < 1e-7 and abs(positive-score['expected_tail_s']) < 1e-7
            and abs(risk-score['risk_tail_s']) < 1e-7, 'branch aggregation mismatch')
    return dict(passed=True, positive_expected_mean_s=positive-(score['negative_work_mass']*negative['mean_s']
                if negative is not None else 0.), independently_aggregated_expected_tail_s=positive,
                independently_aggregated_risk_tail_s=risk, independently_aggregated_score_s=value)


def diagnose(row, model):
    before = state_dict(row['belief'])
    b, state = row['belief'], row['state']
    c, q = row['channel'], np.asarray(row['joint']['q'])
    prepared = b.scenarios()
    started = time.perf_counter()
    joint_score = model.score_measure(b, state['position'], q, c, state['current_channel'],
                                      prepared=prepared, diagnostics=True)
    joint_formula = check_measure_formula(joint_score, model)
    require(not joint_score['receipt_certificate']['certified'], 'actual negative contradicts certified receipt')
    require(joint_score['negative_in_risk'] and joint_score['negative_forecast'] is not None,
            'actual negative is still missing from risk support')
    current_optical = model.forecast_clear(b, state['position'], plan=row['original_plan'])
    current_audit = independent_tail(current_optical, state['position'], b.P)
    if row['prediction']['original_action'] == 'measure':
        original = model.score_measure(b, state['position'], row['original_action']['q'], c,
                                       state['current_channel'], prepared=prepared, diagnostics=True)
        original_audit = check_measure_formula(original, model)
    else:
        original, original_audit = current_optical, current_audit
    require(state_dict(b) == before, 'prediction mutated original belief')
    # Only now apply the observed feedback at the actual historical q. No
    # alternative-point reading is invented or copied from this observation.
    after = copy.deepcopy(b)
    after.update(q, {'measure_result': 'no_signal'})
    actual_after = state_dict(after)
    require(all(actual_after[k] == row['after_actual_belief'][k] for k in ('polygon', 'positives', 'negatives')),
            'actual immediate negative update does not match next trace belief')
    fixed_plan = optical_plan(b.P, q)
    tail_before = model.forecast_clear(b, q, plan=fixed_plan)
    tail_after = model.forecast_clear(after, q, plan=fixed_plan)
    fixed_before_audit = independent_tail(tail_before, q, b.P)
    fixed_after_audit = independent_tail(tail_after, q, after.P)
    negative = joint_score['negative_forecast']
    for key, value in model.compact_forecast(tail_after).items():
        require(negative[key] == value, 'predicted negative branch differs from actual-prefix conditional fallback')
    same_path = tail_before.get('original_path') == tail_after.get('original_path')
    require(same_path, 'negative-only comparison changed the original optical path')
    risk_without_negative = max(b['worst_s'] for b in joint_score['positive_branches'])
    extra_risk = model.RISK_WEIGHT*(joint_score['risk_tail_s']-risk_without_negative)
    return dict(split=row['split'], seed=row['seed'], channel=c, trace_index=row['trace_index'],
        event=row['actual']['event'], accepted_prefix_events=state['events_before'],
        input=binding(row['input_path']), before_position=state['position'].tolist(),
        before_radio_channel=state['current_channel'], active_views_used=state['active'][c],
        before_belief=before, prefix_audit=dict(passed=True,
            exact_belief_rows=state['belief_rows_matched'], entire_old_prediction_exactly_reproduced=True,
            next_negative_belief_exactly_reproduced=True, old_state_not_mutated=True),
        actual_joint_action=row['actual'], old_joint_prediction=row['joint'],
        old_original_action_kind=row['prediction']['original_action'],
        old_original_action=row['original_action'], old_original_prediction=row['prediction']['original'],
        new_joint_score=joint_score, new_original_action_score=original,
        new_current_optical_forecast=current_optical,
        new_joint_minus_original_local_score_s=joint_score['score']-original['score'],
        local_comparison_excludes_station_scan_and_future_route=True,
        new_score_formula_audits=dict(joint=joint_formula, original=original_audit),
        current_optical_cost_audit=current_audit,
        actual_negative_branch_diagnosis=dict(
            empirical_actual_no_signal=True, sampled_signal_mass=joint_score['sampled_signal_mass'],
            negative_work_mass=joint_score['negative_work_mass'], negative_retained_in_risk=True,
            negative_tail_s=negative['worst_s'], largest_positive_tail_s=risk_without_negative,
            weighted_score_increase_from_including_negative_in_max_s=extra_risk,
            negative_is_risk_max=negative['worst_s'] == joint_score['risk_tail_s'],
            caution='This maximum includes the observed negative branch, but .2 weighting and finite positive readings do not constitute a worst-case completion-time guarantee.'),
        same_fixed_original_path_at_actual_q=same_path,
        fixed_path_before_actual_negative=tail_before, fixed_path_after_actual_negative=tail_after,
        fixed_path_cost_audits=dict(before=fixed_before_audit, after=fixed_after_audit),
        diagnostic_wall_s=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS/'feedback_cost_counterexamples_v1.json')
    parser.add_argument('--prefix-only', action='store_true', help='check old inputs without importing new model or writing evidence')
    args = parser.parse_args()
    if not args.prefix_only:
        require(not args.out.exists(), 'refuse to overwrite existing evidence')
    states = selected_states()
    if args.prefix_only:
        print(json.dumps([dict(seed=r['seed'], channel=r['channel'], event=r['actual']['event'],
                              old_signal_mass=r['joint']['signal_mass'],
                              belief_rows_matched=r['state']['belief_rows_matched']) for r in states]))
        return
    model = importlib.import_module('feedback_cost_experiment')
    paths = [Path(__file__), Path(old_model.__file__), Path(model.__file__),
             HERE/'receive_guarantee_v1.py', Q4/'ordered_optical_prune.py', Q4/'negative_region.py',
             Q4/'localization.py', Q4/'shared.py', PROJECT/'src/q3_model_v2/geometry.py']
    hashes = [binding(path) for path in paths]
    rows = []
    for state in states:
        row = diagnose(state, model)
        rows.append(row)
        print(json.dumps(dict(seed=row['seed'], channel=row['channel'],
            actual_negative_in_risk=row['actual_negative_branch_diagnosis']['negative_retained_in_risk'],
            wall_s=row['diagnostic_wall_s']), ensure_ascii=False), flush=True)
    require(hashes == [binding(path) for path in paths], 'diagnostic source changed during execution')
    report = dict(
        evidence_type='Five old actual-prefix conditional cost diagnoses; no new whole-game execution or official test.',
        scope='Five accepted joint q points in development 54100 and holdout 55101. Old source truth never enters prediction. Alternative original action scores are forecasts only, with no reused feedback.',
        limitations=['These are selected known counterexamples, not independent performance samples.',
                     'The new online candidate uses its original ten local points, not these five joint stations; scoring a joint q here is a mechanism diagnostic only.',
                     'Local action comparison excludes future route and station scan cost. It does not reconstruct or claim a new joint-policy decision.',
                     'Position/type/heading/radius quadrature and .8/.2 coefficients are work-model assumptions, not official probabilities or robust guarantees.',
                     'Finite-position mean after negative feedback is re-sampled in the retained polygon; it is not an exact Bayesian conditional mean.'],
        model_id=model.MODEL_ID, source_bindings=hashes, rows=rows,
        summary=dict(cases=len(rows), old_p1_actual_negative=sum(r['old_joint_prediction']['signal_mass'] == 1 for r in rows),
            actual_negatives_in_new_risk=sum(r['new_joint_score']['negative_in_risk'] for r in rows),
            negative_is_maximum=sum(r['actual_negative_branch_diagnosis']['negative_is_risk_max'] for r in rows),
            all_prefix_and_cost_checks_passed=True))
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=serial)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.out), **report['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
