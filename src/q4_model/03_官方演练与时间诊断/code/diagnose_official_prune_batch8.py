"""Offline cost and route diagnostics supplementing the independent batch-8 audit.

No solver/client is imported or executed. No raw identifiers are exported.
The scan skeleton and direct optical-success paths use hindsight; neither is
an executable alternative policy or an estimated official competition score.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def diagnose(archive, checked):
    raw = archive.read(checked['input_member'])
    raw_trace = archive.read(checked['trace_member'])
    require(sha(raw) == checked['input_log_sha256'], 'log changed')
    require(sha(raw_trace) == checked['input_trace_sha256'], 'trace changed')
    records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    trace = json.loads(raw_trace.decode('utf-8-sig'))
    summary = records[-1]
    markers = [r for r in trace if r.get('phase') == 'actual_action']
    calls = [r for r in records if r.get('path') in ('/measure', '/clear')]
    require(len(markers) == len(calls) == checked['actual_actions'], 'action count differs')
    position, tuned, elapsed = [0., 0.], 1, 0.
    actions = []
    for number, (call, marker) in enumerate(zip(calls, markers), 1):
        req, res = call['request'], call['response']
        point, channel = [req['position']['x'], req['position']['y']], req['channel']
        action = call['path'][1:]
        kind = res['measure_result' if action == 'measure' else 'clear_result']
        require(res['accepted'] is True and marker['event'] == number, 'invalid accepted action')
        require(marker['position'] == point and marker['result'] == kind, 'trace differs from response')
        switch = int(tuned != channel) if action == 'measure' else 0
        move = math.dist(position, point)/5
        fee = 5+switch if action == 'measure' else (5 if kind == 'success' else 3)
        item = dict(event=number, action=action, channel=channel, position=point,
                    result=kind, reason=marker['reason'], move_s=move, fee_s=fee,
                    start_s=elapsed, time_s=res['virtual_time_s'], duration_s=move+fee)
        actions.append(item)
        elapsed += move+fee
        require(abs(elapsed-res['virtual_time_s']) < 1e-4, 'timing inconsistency')
        position = point
        if action == 'measure':
            tuned = channel
    require(abs(elapsed-checked['total_virtual_s']) < 1e-4, 'audit total differs')
    scan = [a for a in actions if a['reason'] == 'scan']
    scan_stops = []
    for a in scan:
        if not scan_stops or scan_stops[-1] != a['position']:
            scan_stops.append(a['position'])
    skeleton = [[0., 0.]]+scan_stops+[actions[-1]['position']]
    actual_move_s = length([[0., 0.]]+[a['position'] for a in actions])/5
    skeleton_s = length(skeleton)/5
    require(skeleton_s <= actual_move_s+1e-7, 'skeleton exceeds real route')
    cleared = set(checked['cleared_channels'])
    absent_measures = [a for a in actions if a['action'] == 'measure' and a['channel'] not in cleared]
    reasons = {}
    for reason in sorted({a['reason'] for a in actions}):
        rows = [a for a in actions if a['reason'] == reason]
        reasons[reason] = dict(actions=len(rows), move_s=sum(a['move_s'] for a in rows),
            fee_s=sum(a['fee_s'] for a in rows), total_s=sum(a['duration_s'] for a in rows),
            positive=sum(a['result'] in ('direction', 'near') for a in rows),
            negative=sum(a['result'] == 'no_signal' for a in rows),
            failed_clear=sum(a['result'] == 'no_target_in_range' for a in rows))
    channels = []
    for c in sorted(cleared):
        rows = [a for a in actions if a['channel'] == c]
        first = next(a for a in rows if a['result'] in ('direction', 'near'))
        success = next(a for a in rows if a['result'] == 'success')
        channels.append(dict(channel=c, first_detected_s=first['time_s'], cleared_s=success['time_s'],
            discovery_scan_stop=sum(a['time_s'] <= first['time_s'] and (i == 0 or a['position'] != scan[i-1]['position'])
                                    for i, a in enumerate(scan)),
            clear_fail=sum(a['result'] == 'no_target_in_range' for a in rows),
            measure_counts=dict(Counter(a['reason'] for a in rows if a['action'] == 'measure')),
            non_scan_destination_cost_s=sum(a['duration_s'] for a in rows if a['reason'] != 'scan')))
    optical = []
    for p in checked['optical_plan_details']:
        first_index = p['preceding_actual_event']
        rows = actions[first_index:first_index+p['actual_clear_calls']]
        require(all(a['reason'] == 'optical_cover' and a['channel'] == p['channel'] for a in rows), 'optical slice differs')
        require(rows[-1]['result'] == 'success', 'optical slice lacks success')
        entry = actions[first_index-1]['position'] if first_index else [0., 0.]
        direct = math.dist(entry, rows[-1]['position'])/5+5
        cost = sum(a['duration_s'] for a in rows)
        require(abs(cost-p['actual_block_cost_s']['total_s']) < 1e-6, 'optical audit disagrees')
        optical.append(dict(channel=p['channel'], actual_points=len(rows),
            original_points=p['original_plan_points'], actual_cost_s=cost,
            direct_to_observed_success_cost_s=direct, hindsight_excess_s=cost-direct,
            already_saved_by_prune_s=p['proved_avoided_cost_s']))
    last_clear = max(a['time_s'] for a in actions if a['result'] == 'success')
    return dict(case=checked['case'], cleared=checked['cleared'],
        total_virtual_s=checked['total_virtual_s'], per_source_virtual_s=checked['per_source_virtual_s'],
        user_type_counts=dict(omni=checked['omni_from_user'], directional=checked['directional_from_user']),
        counts=checked['action_counts'], costs_s=checked['costs_s'], by_reason=reasons,
        scan_stop_visits=len(scan_stops), distinct_scan_stops=len({tuple(p) for p in scan_stops}),
        scan_stations_configured=len(summary['stations']) if 'stations' in summary else None,
        last_clear_s=last_clear, post_last_clear_s=checked['total_virtual_s']-last_clear,
        post_last_clear_actions=sum(a['start_s'] >= last_clear-1e-4 for a in actions),
        absent_channel_measures=len(absent_measures), absent_channel_measure_and_switch_s=sum(a['fee_s'] for a in absent_measures),
        actual_move_s=actual_move_s, same_order_scan_skeleton_s=skeleton_s,
        service_insertion_extra_move_s=actual_move_s-skeleton_s,
        scan_skeleton_definition='Polyline retains the actual ordered scan stops and final position, removes all intervening service points. Hindsight geometric reference only, not a feasible source-clearing route.',
        optical_hindsight_excess_s=sum(p['hindsight_excess_s'] for p in optical),
        optical_hindsight_definition='Within each actual optical block, compare with going directly to its observed success point. Uses hindsight: an upper bound on savings restricted to these blocks and endpoints, not an achievable policy.',
        optical_blocks=optical, channels=channels, actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'refuse to overwrite evidence')
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    require(audit['passed'] and sha(args.zip.read_bytes()) == audit['archive_sha256'], 'unverified input')
    with zipfile.ZipFile(args.zip) as archive:
        cases = [diagnose(archive, row) for row in audit['cases']]
    total = sum(c['total_virtual_s'] for c in cases)
    costs = dict(sum((Counter(c['costs_s']) for c in cases), Counter()))
    by_reason = {reason: dict(sum((Counter(c['by_reason'].get(reason, {})) for c in cases), Counter()))
                 for reason in sorted(set().union(*(c['by_reason'] for c in cases)))}
    aggregate = dict(runs=len(cases), cleared=sum(c['cleared'] for c in cases),
        total_virtual_s=total, mean_total_s=total/len(cases),
        pooled_per_source_s=total/sum(c['cleared'] for c in cases),
        mean_case_per_source_s=sum(c['per_source_virtual_s'] for c in cases)/len(cases),
        costs_s=costs, cost_percent={k:v/total*100 for k, v in costs.items()}, by_reason=by_reason)
    for key in ('post_last_clear_s', 'absent_channel_measures', 'absent_channel_measure_and_switch_s',
                'actual_move_s', 'same_order_scan_skeleton_s', 'service_insertion_extra_move_s', 'optical_hindsight_excess_s'):
        aggregate[key] = sum(c[key] for c in cases)
    result = dict(archive_sha256=audit['archive_sha256'], audit_sha256=sha(args.audit.read_bytes()),
        analysis_script_sha256=sha(Path(__file__).read_bytes()), cases=cases, aggregate=aggregate,
        local_only=True, source_policy_modified=False, formal_results_modified=False)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    for c in cases:
        print(json.dumps({k:v for k,v in c.items() if k not in ('actions','channels','optical_blocks','scan_skeleton_definition','optical_hindsight_definition')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
