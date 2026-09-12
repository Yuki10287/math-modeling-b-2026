"""Audit saved official Q4 shared logs without opening any simulator connection.

Does not infer an official source total from the client's own completion flag.
Reads ZIP members in memory; output excludes identity and request identifiers.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_audit():
    root = next(p for p in Path(__file__).resolve().parents if (p/'src/studies.json').is_file())
    registry = json.loads((root/'src/studies.json').read_text(encoding='utf-8'))
    paths = {r['logical']: root/r['path'] for r in registry['files']}
    base = root/'tmp/official_archive_audit'
    base.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='run-', dir=base))
    # Preserve historical relative imports in a disposable local execution view.
    for logical, source in paths.items():
        if logical.startswith(('src/q4_model/', 'src/q3_model_v2/')) and logical.endswith('.py'):
            dest = runtime/logical
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
    sys.path.insert(0, str(runtime/'src/q4_model'))
    import analyze_official_results as legacy
    return root, paths, legacy


def analyze_member(z, name, paths, legacy):
    raw = z.read(name)
    trace_raw = z.read(name+'.beliefs.json')
    records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    traces = json.loads(trace_raw.decode('utf-8-sig'))
    starts = [r for r in records if r.get('type') == 'session_start']
    summaries = [r for r in records if r.get('type') == 'session_summary']
    assert len(starts) == len(summaries) == 1
    start, summary = starts[0], summaries[0]
    assert start['problem'] == summary['problem'] == 4
    assert start['variant'] == summary['variant'] == 'shared'
    assert summary['complete'] and summary['exit_accepted']
    errors = sum(r.get('type') == 'client_error' for r in records)
    assert errors == 0
    hashes = start['source_sha256']
    assert hashes == summary['source_sha256']
    hash_matches = {name: sha(paths['src/'+name].read_bytes()) == value for name, value in hashes.items()}
    assert all(hash_matches.values())
    calls = [r for r in records if r.get('path')]
    assert calls[0]['path'] == '/enter' and calls[-1]['path'] == '/exit'
    assert all(r['response']['accepted'] for r in calls)
    assert len({r['request']['request_id'] for r in calls}) == len(calls)
    assert calls[0]['response']['virtual_time_s'] == 0
    assert calls[-1]['response']['exit_reason'] == 'user_exit'
    position, tuned, elapsed = np.zeros(2), 1, 0.
    counts, actions, cleared, discovered, errors_s = Counter(), [], set(), {}, []
    markers = [r for r in traces if r['phase'] == 'actual_action']
    assert len(markers) == len(calls)-2
    for row, marker in zip(calls[1:-1], markers):
        req, response = row['request'], row['response']
        action, c = row['path'][1:], req['channel']
        q = np.array([req['position']['x'], req['position']['y']], float)
        assert action in ('measure', 'clear') and type(c) is int and 1 <= c <= 20
        assert np.isfinite(q).all() and np.max(np.abs(q)) <= 2_000_000
        move_s = float(np.linalg.norm(q-position))/5
        switch = int(tuned != c) if action == 'measure' else 0
        key = 'measure_result' if action == 'measure' else 'clear_result'
        kind = response[key]
        if action == 'measure':
            assert kind in ('no_signal', 'direction', 'near')
            fee = 5+switch
            counts['measure'] += 1
            counts['switch'] += switch
            counts[kind] += 1
            tuned = c
            if kind != 'no_signal':
                discovered.setdefault(c, response['virtual_time_s'])
        else:
            assert kind in ('success', 'no_target_in_range')
            fee = 5 if kind == 'success' else 3
            counts['clear_success' if kind == 'success' else 'clear_fail'] += 1
            if kind == 'success':
                assert c not in cleared
                cleared.add(c)
        item = dict(index=len(actions)+1, action=action, channel=c, position=q.tolist(),
                    move_s=move_s, fee_s=fee, duration_s=move_s+fee, start_s=elapsed,
                    time_s=response['virtual_time_s'], reason=marker['reason'],
                    switch=switch, response=response, **{key:kind})
        assert marker['event'] == item['index'] and marker['action'] == action
        assert marker['channel'] == c and marker['result'] == kind
        assert np.linalg.norm(np.asarray(marker['position'])-q) <= 1e-7
        assert abs(marker['move_s']-move_s) <= 1e-7
        elapsed += move_s+fee
        errors_s.append(abs(elapsed-response['virtual_time_s']))
        assert errors_s[-1] <= 1e-4
        actions.append(item)
        position = q
    total = summary['total_virtual_s']
    assert abs(elapsed-total) <= 1e-4
    assert abs(total-calls[-1]['response']['virtual_time_s']) <= 1e-4
    assert len(cleared) == summary['successful_clear_count']
    assert cleared == set(summary['cleared']) == set(discovered)
    geometry = legacy.independent_geometry(traces, actions, summary)
    # Strict replay only: never feed a saved response to an alternative action.
    replay = legacy.FeedbackReplay(actions)
    replay_trace = []
    try:
        result = legacy.solve_multi(replay, variant='shared', trace=replay_trace)
        assert replay.cursor == len(actions)
        legacy.assert_tree_close(replay_trace, traces)
        legacy.assert_tree_close(json.loads(json.dumps(result)), {k:summary[k] for k in result})
        replay_audit = dict(strict=True, consumed_actions=replay.cursor)
    except AssertionError as exc:
        replay_audit = dict(strict=False, consumed_actions=replay.cursor, mismatch=str(exc)[:600])
    costs = dict(move=sum(a['move_s'] for a in actions), measure=5*counts['measure'],
                 switch=counts['switch'], clear_success=5*counts['clear_success'],
                 clear_fail=3*counts['clear_fail'])
    by_reason = {}
    for reason in sorted({a['reason'] for a in actions}):
        rows = [a for a in actions if a['reason'] == reason]
        by_reason[reason] = dict(actions=len(rows), time_s=sum(a['duration_s'] for a in rows),
            move_s=sum(a['move_s'] for a in rows), fee_s=sum(a['fee_s'] for a in rows))
    last_clear = max(a['time_s'] for a in actions if a.get('clear_result') == 'success')
    scan_points = list(dict.fromkeys(tuple(a['position']) for a in actions if a['reason'] == 'scan'))
    pts = np.array(scan_points)
    station_floor = float(minimum_spanning_tree(np.linalg.norm(pts[:, None]-pts, axis=2)).sum())/5
    absent = set(summary['certificate']['absent'])
    return dict(case=Path(name).stem, input_sha256=sha(raw), trace_sha256=sha(trace_raw),
        variant=summary['variant'], source_files_checked=len(hashes), source_hashes_match=hash_matches,
        total_s=total, cleared=len(cleared), average_s=total/len(cleared), absent=len(absent),
        official_source_total=None, official_type_composition=None,
        observed_positive_channels_all_cleared=True, completion_geometry_passed=geometry['passed'],
        official_result_sheet_crosscheck=False, exit_accepted=True, client_errors=errors,
        rejected_actions=0, unique_accepted_requests=len(calls), counts=dict(counts), costs_s=costs,
        by_reason=by_reason, post_last_clear_s=total-last_clear, last_clear_s=last_clear,
        no_source_channel_measures=sum(a['action'] == 'measure' and a['channel'] in absent for a in actions),
        scan_stops=len(scan_points), geometry_audit=geometry, replay_audit=replay_audit,
        max_cumulative_timing_error_s=max(errors_s),
        fixed_station_mst_lower_bound_s=station_floor,
        fixed_station_and_observed_fees_floor_s=station_floor+sum(v for k,v in costs.items() if k != 'move'),
        local_session_wall_time_s=summary['local_session_wall_time_s'],
        official_program_runtime_s=summary['official_program_runtime_s'],
        actions=[{k:v for k,v in a.items() if k != 'response'} for a in actions])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root, paths, legacy = prepare_audit()
    with zipfile.ZipFile(args.zip) as z:
        cases = []
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl')):
            case = analyze_member(z, name, paths, legacy)
            cases.append(case)
            print(json.dumps({k:case[k] for k in ('case','variant','total_s','cleared','average_s','replay_audit')}, ensure_ascii=False), flush=True)
    total = sum(r['total_s'] for r in cases)
    cleared = sum(r['cleared'] for r in cases)
    counts = dict(sum((Counter(r['counts']) for r in cases), Counter()))
    costs = {k:sum(r['costs_s'][k] for r in cases) for k in cases[0]['costs_s']}
    aggregate = dict(runs=len(cases), total_s=total, mean_total_s=total/len(cases), cleared=cleared,
        pooled_average_s=total/cleared, mean_case_average_s=float(np.mean([r['average_s'] for r in cases])),
        counts=counts, costs_s=costs, cost_shares_pct={k:100*v/total for k,v in costs.items()},
        post_last_clear_s=sum(r['post_last_clear_s'] for r in cases),
        scan_time_s=sum(r['by_reason']['scan']['time_s'] for r in cases),
        scan_measures=sum(r['by_reason']['scan']['actions'] for r in cases),
        no_source_channel_measures=sum(r['no_source_channel_measures'] for r in cases),
        max_cumulative_timing_error_s=max(r['max_cumulative_timing_error_s'] for r in cases),
        strict_replay_runs=sum(r['replay_audit']['strict'] for r in cases),
        mean_fixed_station_and_observed_fees_floor_s=float(np.mean([r['fixed_station_and_observed_fees_floor_s'] for r in cases])),
        target5000_required_reduction_pct=100*(1-5000*len(cases)/total),
        target5000_mean_gap_s=total/len(cases)-5000)
    old_path = paths['src/q4_model/results/official_q4_20260911_analysis.json']
    old = json.loads(old_path.read_text(encoding='utf-8-sig'))['aggregate']
    report = dict(archive_name=args.zip.name, archive_sha256=sha(args.zip.read_bytes()),
        analysis_script_sha256=sha(Path(__file__).read_bytes()),
        official_simulator_contacted=False, original_solver_modified=False,
        cases=cases, aggregate=aggregate, previous_batch=dict(input_sha256=sha(old_path.read_bytes()),
        runs=old['runs'], mean_total_s=old['mean_total_s'], pooled_average_s=old['pooled_average_s']),
        limitations=['Official total counts and type composition not supplied; successful clear count is not independent ground truth.',
        'Completion certificates audited from actual saved feedback under the problem assumptions.',
        'No true source locations, official result sheet or encrypted official activity log supplied.',
        'Different official layouts are not a paired experiment. All four records are shared, not share25.',
        '5000 seconds is an unverified user-observed external reference, not a confirmed equal-condition benchmark.',
        'The fixed-station floor holds only if the 25 stops and observed action counts are retained.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
