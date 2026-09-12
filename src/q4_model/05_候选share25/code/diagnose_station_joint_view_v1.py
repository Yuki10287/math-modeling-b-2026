"""Read saved v1 cases; preserve a completed bounded predictor microtest record.

This diagnostic does not import a solver, read source truth, obtain feedback,
or repeat the 885-second game. The microtest numbers below are observations
from the separately completed read-only process, not timings of this script.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
ROOT = HERE.parents[3]
MODEL_SHA = 'f83999d7a38316930bccfee6d7304aff35a4117569f8347fe55ab816919118cc'
MICRO_INPUT_SHA = '81f58ee39e43d0fa42403dba3bc757e1501281b227b48da759299c108199708e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def without_clock(event):
    return {k: v for k, v in event.items() if k != 'time_s'}


def actions(case):
    return [t for t in case['trace'] if t.get('phase') == 'actual_action']


def discoveries(case):
    found = {}
    for event, row in enumerate(case['events'], 1):
        if row['action'] == 'measure' and row['measure_result'] != 'no_signal':
            found.setdefault(row['channel'], {'event': event, 'position': row['position'],
                                              'time_s': row['time_s']})
    return found


def optical_blocks(case):
    aa = actions(case)
    output = {}
    for row in case['trace']:
        if row.get('phase') != 'ordered_optical_prune':
            continue
        channel = row['channel']
        block = [a for a in aa if a['channel'] == channel and a['reason'] == 'optical_cover']
        # The four cases each contain at most one optical block per channel.
        assert channel not in output
        event_costs = []
        for action in block:
            index = action['event'] - 1
            prior = case['events'][index - 1]['time_s'] if index else 0
            event_costs.append(case['events'][index]['time_s'] - prior)
        output[channel] = dict(
            plan=row, actual=block, total_s=sum(event_costs),
            actions=len(block), failures=sum(a['result'] != 'success' for a in block),
            move_s=sum(a['move_s'] for a in block))
    return output


def compare(pair_directory, seed):
    prefix = f'{seed}-boundary-hashed-'
    paths = [RESULTS / pair_directory / (prefix + suffix + '.json') for suffix in
             ('baseline_share25_prune', 'station_joint_view')]
    old, new = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    first = next(i for i, (a, b) in enumerate(zip(old['events'], new['events']))
                 if without_clock(a) != without_clock(b))
    old_blocks, new_blocks = optical_blocks(old), optical_blocks(new)
    blocks = []
    for channel in sorted(set(old_blocks) | set(new_blocks)):
        a, b = old_blocks.get(channel), new_blocks.get(channel)
        assert a is not None and b is not None
        pa, pb = a['plan'], b['plan']
        na, nb = set(map(tuple, pa['negatives'])), set(map(tuple, pb['negatives']))
        blocks.append(dict(
            channel=channel,
            baseline={k: a[k] for k in ('actions', 'failures', 'move_s', 'total_s')},
            candidate={k: b[k] for k in ('actions', 'failures', 'move_s', 'total_s')},
            delta_s=b['total_s']-a['total_s'],
            original_polygon_exactly_same=pa['original_polygon'] == pb['original_polygon'],
            original_point_set_exactly_same=sorted(pa['original_path']) == sorted(pb['original_path']),
            original_order_exactly_same=pa['original_path'] == pb['original_path'],
            original_points=[len(pa['original_path']), len(pb['original_path'])],
            removed_planned=[len(pa['removed_indices']), len(pb['removed_indices'])],
            first_actual_points=[a['actual'][0]['position'], b['actual'][0]['position']],
            successful_clearpoint_exactly_same=a['actual'][-1]['position'] == b['actual'][-1]['position'],
            baseline_only_negative_points=sorted(na-nb), candidate_only_negative_points=sorted(nb-na)))
    od, nd = discoveries(old), discoveries(new)
    channels = list(od)
    assert channels == list(nd)
    assert all(od[c]['position'] == nd[c]['position'] for c in channels)
    oa, na = actions(old), actions(new)
    joint = []
    for trace_index, row in enumerate(new['trace']):
        if row.get('phase') != 'station_joint_prediction' or row['selected'] is None:
            continue
        action = next(t for t in new['trace'][trace_index+1:] if t.get('phase') == 'actual_action')
        assert action['reason'] == 'joint_source_measure'
        selection = next(c for c in row['candidates'] if c['station'] == row['selected'])
        joint.append(dict(trace_index=trace_index, actual_action=action, original_action=row['original_action'],
                          original_forecast=row['original'], selected_forecast=selection,
                          expected_saving_s=row['original']['score']-selection['score'],
                          original_prune_diagnostic=row['original_prune_diagnostic']))
    delta = new['summary']['total_s']-old['summary']['total_s']
    last_old, last_new = od[channels[-1]]['event'], nd[channels[-1]]['event']
    suffix_same = [without_clock(e) for e in old['events'][last_old:]] == [
        without_clock(e) for e in new['events'][last_new:]]
    return dict(
        inputs=[binding(p) for p in paths],
        baseline_summary=old['summary'], candidate_summary=new['summary'],
        total_delta_s=delta, relative_regression_percent=100*delta/old['summary']['total_s'],
        component_delta_s={k: new['summary']['time_parts_s'][k]-v
                           for k, v in old['summary']['time_parts_s'].items()},
        reason_counts=[dict(Counter(a['reason'] for a in aa)) for aa in (oa, na)],
        first_behavior_divergence=dict(event=first+1, baseline=old['events'][first], candidate=new['events'][first]),
        selected_joint_actions=joint, optical_blocks=blocks,
        optical_extra_s=sum(b['delta_s'] for b in blocks),
        optical_share_of_total_regression=sum(b['delta_s'] for b in blocks)/delta,
        discoveries=[dict(channel=c, baseline=od[c], candidate=nd[c],
                          candidate_delay_s=nd[c]['time_s']-od[c]['time_s']) for c in channels],
        discovery_order_and_positions_exactly_same=True,
        suffix_after_last_discovery=dict(baseline_start_after_event=last_old,
                                        candidate_start_after_event=last_new,
                                        baseline_actions=len(old['events'])-last_old,
                                        candidate_actions=len(new['events'])-last_new,
                                        identical_except_virtual_clock=suffix_same))


def completed_microtest():
    """Record the already-completed child process, without rerunning it."""
    rows = [
        (351,324,17,7,16,8,1,None,.2576610999822151,.265625),
        (356,325,17,7,17,8,1,None,.05551760000525974,.046875),
        (386,348,20,5,17,7,1,None,.07300509998458438,.078125),
        (403,354,14,5,19,7,2,19,.4994474000122864,.5),
        (475,416,8,5,19,5,1,None,.18611469998722896,.171875),
        (480,417,8,5,20,5,1,None,.05678970000008121,.0625),
        (520,446,19,4,21,3,0,None,.39112730001215823,.375),
        (529,449,18,8,22,3,0,None,.09008369999355637,.09375),
        (549,462,4,7,22,2,1,None,.5587745999800973,.546875),
        (554,463,4,6,22,2,1,None,.6564035000046715,.625),
        (562,465,4,6,23,2,0,None,.08514359997934662,.078125),
        (587,482,16,4,12,1,0,None,.3466891999996733,.34375),
    ]
    keys = ('trace_index', 'accepted_events_before', 'channel', 'polygon_vertices',
            'negative_points', 'remaining_scan_tasks', 'effective_station_candidates',
            'selected_station', 'wall_s', 'cpu_s')
    records = [dict(zip(keys, row), prediction_exactly_equal_to_saved=True) for row in rows]
    inp = RESULTS / 'station_joint_view_dev54000_v1/54100-boundary-hashed-station_joint_view.json'
    assert sha(inp) == MICRO_INPUT_SHA
    trace = json.loads(inp.read_text(encoding='utf-8'))['trace']
    for record in records:
        row = trace[record['trace_index']]
        assert row['phase'] == 'station_joint_prediction'
        assert row['channel'] == record['channel'] and row['selected'] == record['selected_station']
        assert len(row['candidates']) == record['effective_station_candidates']
    return dict(
        record_type='retrospective_record_of_completed_bounded_microtest',
        rerun_by_this_diagnostic_script=False, input=binding(inp),
        python_command='D:\\Python\\python.exe -B -X utf8 -',
        method=[
            'Reconstruct each of the 12 selector inputs from preceding accepted actions and latest belief trace only; never read saved sources.',
            'Restore polygon, positives and negatives; measured points start at the first positive for that channel; restore failed clears, last actual position and last measured channel.',
            'Restore coverage negative ledgers from actual no_signal measurements; reuse recorded fixed tasks, points and unknown channels.',
            'Recompute original choose_measure and optical_plan; call the unchanged select_joint_measure; compare the entire produced prediction dictionary by exact Python equality.',
            'Measure perf_counter wall and process_time CPU; inclusive wrappers instrument optical_plan, optical_forecast and measure_forecast in that child process only.',
            '55-second deadline checked before wrapped calls; process finished around 4.7 seconds, without timeout; no whole-game replay, file mutation, or new feedback.'
        ],
        predictions=records, exact_predictions=12,
        total_timed_wall_s=3.268361400027061, total_timed_cpu_s=3.1875,
        inclusive_profile_not_additive={
            'optical_plan': dict(calls=439, wall_s=.35015629994450137, cpu_s=.3125),
            'optical_forecast': dict(calls=444, wall_s=.5235208996164147, cpu_s=.53125),
            'measure_forecast': dict(calls=60, wall_s=.6235535000450909, cpu_s=.609375)},
        original_game_runtime_s=885.4039062000229,
        original_game_cpu_or_segment_times_available=False,
        conclusion='The 12 identical predictions did not reproduce a function slowdown. Process scheduling or machine suspension remains possible; the original log lacks CPU and segment timings, so the 885-second cause is not diagnosed. Candidate count alone does not explain it.')


def main():
    model = HERE / 'station_joint_view_experiment.py'
    assert sha(model) == MODEL_SHA
    dev = compare('station_joint_view_dev54000_v1', 54100)
    holdout = compare('station_joint_view_holdout55000_v1', 55101)
    report = dict(
        scope='Read-only analysis of saved local development and independent holdout cases; no official test or new solver experiment.',
        diagnostic_source=binding(Path(__file__)), frozen_model=binding(model),
        development_54100=dev, holdout_55101=holdout, bounded_predictor_microtest=completed_microtest(),
        findings={
            'development': 'A single joint negative at station 19 replaces c14 original negative view. The finite forecast assigns signal mass 1, but actual response is negative. Same 98-point optical plan then removes 36 instead of 60 points and executes 32 instead of 20; optical extra 117.97 seconds explains most of 136.39 seconds regression.',
            'holdout': 'Four selected joint measurements are all negative, including two forecast signal masses of 1. Earlier c16/c19 discoveries are overtaken by optical entry/order and negative-witness changes. Five affected blocks add 131 failed clears and 1328.11 seconds, with unchanged polygons, original point sets and success points. The whole 1690.43-second difference exists by the last discovery; subsequent 49 actions are identical.',
            'interpretation': 'The scores are finite work-model forecasts, not probabilities guaranteed by the task. They do not reliably value negative witness placement or early success order. More certified deletions alone do not ensure earlier success: c11 removes more planned points but executes 31 instead of 1.',
            'decision': 'Do not promote frozen v1 or adjust it against this counterexample. Preserve its performance and audit evidence as a rejected local candidate.'},
        exact_selection_preserving_cost_pruning_notes={
            'status': 'Mathematical possibility only; not implemented or tested; microtest does not establish a current performance need.',
            'lower_bound': 'For a candidate use its immediate move/measurement/switch and current scan fee, plus fixed residual internal-route movement and frozen scan fees. Other expected optical and exit-connector costs are nonnegative; their omission gives a conservative lower bound.',
            'branch_bound': 'Already accumulated nonnegative weighted optical/connector terms can strengthen this bound; uncomputed terms may be bounded by zero. Stop only when a conservative bound cannot beat the incumbent under the exact existing tie policy.',
            'numerics': 'A float implementation must round conservatively and account for score comparison tolerance; no claim that naive float summation preserves exact selection.',
            'limitation': 'Such pruning preserves the present proxy selection, not actual run quality; it does not repair the holdout optical-entry regressions.'})
    output = RESULTS / 'station_joint_view_v1_diagnostics.json'
    if output.exists():
        raise FileExistsError('Refuse to overwrite existing evidence: ' + str(output))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), development_delta_s=dev['total_delta_s'],
                          holdout_delta_s=holdout['total_delta_s'],
                          holdout_optical_extra_s=holdout['optical_extra_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()
