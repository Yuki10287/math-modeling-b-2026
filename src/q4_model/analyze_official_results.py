"""Offline Q4 log audit and deterministic feedback replay; no HTTP access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from joint_solver import solve_multi
import joint_solver
from route_planning import route_length
from polar_cover import PolarCover
from shared import ROOT


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def assert_tree_close(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            assert_tree_close(a[k], b[k])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_close(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-7), (a, b)
    else:
        assert a == b, (a, b)


class FeedbackReplay:
    """Consume saved replies only if the unchanged policy requests the same action."""
    def __init__(self, actions):
        self.actions = actions
        self.cursor = 0
        self.position = np.zeros(2)
        self.channel = 1

    def action(self, kind, q, c):
        assert self.cursor < len(self.actions), 'replay requested an extra action'
        row = self.actions[self.cursor]
        assert row['action'] == kind and row['channel'] == c
        assert np.linalg.norm(np.array(row['position'])-q) <= 1e-7, (self.cursor, q, row['position'])
        self.position = np.asarray(q).copy()
        if kind == 'measure':
            self.channel = c
        self.cursor += 1
        return dict(row['response'])

    def measure(self, q, c):
        return self.action('measure', q, c)

    def clear(self, q, c):
        return self.action('clear', q, c)


def replay_logged_actions(actions):
    api = FeedbackReplay(actions)
    trace = []
    try:
        result = solve_multi(api, variant='shared', trace=trace)
        assert api.cursor == len(actions)
        return result, trace, dict(strict=True, consumed_actions=api.cursor, equivalent_route_reversals=[])
    except AssertionError as exc:
        strict_failure = dict(consumed_actions=api.cursor, detail=str(exc))
    # A symmetric open path and its reversal can differ at the last floating-point bit.
    # This second pass is explicitly conditioned on the logged choice, not a strict replay.
    api = FeedbackReplay(actions)
    trace, ties = [], []
    original_route = joint_solver.open_route
    def logged_equal_route(points, start):
        order = original_route(points, start)
        if api.cursor >= len(actions) or actions[api.cursor]['reason'] != 'scan':
            return order
        q = np.asarray(actions[api.cursor]['position'])
        if np.linalg.norm(points[order[0]]-q) <= 1e-7:
            return order
        reverse = order[::-1]
        if np.linalg.norm(points[reverse[0]]-q) > 1e-7:
            return order
        forward_value = route_length(points, start, order)
        reverse_value = route_length(points, start, reverse)
        if abs(forward_value-reverse_value) > 1e-8:
            return order
        ties.append(dict(event=api.cursor+1, original_first_point=points[order[0]].tolist(),
            logged_first_point=q.tolist(), forward_length_m=forward_value,
            reverse_length_m=reverse_value, absolute_difference_m=abs(forward_value-reverse_value)))
        return reverse
    joint_solver.open_route = logged_equal_route
    try:
        result = solve_multi(api, variant='shared', trace=trace)
        assert api.cursor == len(actions) and ties
    finally:
        joint_solver.open_route = original_route
    return result, trace, dict(strict=False, strict_failure=strict_failure,
        consumed_actions=api.cursor, equivalent_route_reversals=ties,
        note='Replay conditions only equal-cost route reversals on the logged choice; it is not a strict deterministic replay.')


def independent_geometry(traces, actions, summary):
    mesh = PolarCover()
    cert = summary['certificate']
    assert cert['basis'] == 'directional_triangle_cover'
    assert np.allclose(cert['stations'], mesh.stations, rtol=0, atol=1e-7)
    assert np.array_equal(cert['triangles'], mesh.indices)
    cleared = {a['channel'] for a in actions if a.get('clear_result') == 'success'}
    detected = {a['channel'] for a in actions if a['action'] == 'measure' and a['measure_result'] != 'no_signal'}
    absent = set(cert['absent'])
    assert cleared == detected == set(cert['cleared']) == set(summary['cleared'])
    assert absent | cleared == set(range(1, 21)) and not absent & cleared
    negative = {c: [a['position'] for a in actions if a['channel'] == c and a.get('measure_result') == 'no_signal']
                for c in absent}
    triangles = 0
    for c in absent:
        points = np.asarray(negative[c])
        claimed = cert['channels'][str(c)]
        assert claimed['complete'] and claimed['covered'] == claimed['triangles'] == 36
        recorded_points = np.asarray(claimed['negative_points'])
        assert all(np.min(np.linalg.norm(points-p, axis=1)) <= 1e-7 for p in recorded_points)
        assert set(map(int, claimed['triangle_witnesses'])) == set(range(36))
        for t, indices in claimed['triangle_witnesses'].items():
            T = mesh.triangles[int(t)]
            witnesses = recorded_points[indices]
            assert np.max(np.linalg.norm(T[:, None, :]-witnesses, axis=2)) <= 1000-1e-7
            hull = ConvexHull(witnesses)
            assert np.max(T @ hull.equations[:, :2].T + hull.equations[:, 2]) <= 1e-7
            triangles += 1
    polygons, optical, safe_clear = 0, 0, 0
    latest_polygons = {}
    for row in traces:
        if row['phase'] == 'belief':
            P = np.asarray(row['polygon'])
            assert len(P) and np.isfinite(P).all()
            latest_polygons[row['channel']] = P
            polygons += 1
        elif row['phase'] == 'optical_plan':
            P, R, origin = map(np.asarray, (row['polygon'], row['rotation'], row['origin']))
            lo, hi = map(np.asarray, (row['lower'], row['upper']))
            nx, ny = row['cells']
            assert np.allclose(R.T @ R, np.eye(2), atol=1e-10)
            local = (P-origin) @ R
            assert np.all(local >= lo-1e-7) and np.all(local <= hi+1e-7)
            assert np.linalg.norm((hi-lo)/[nx, ny])/2 < 20-1e-6
            expected = np.array([[lo[0]+(i+.5)*(hi[0]-lo[0])/nx, lo[1]+(j+.5)*(hi[1]-lo[1])/ny]
                                 for i in range(nx) for j in range(ny)]) @ R.T + origin
            path = np.asarray(row['path'])
            assert len(path) == nx*ny
            assert np.max(np.min(np.linalg.norm(expected[:, None, :]-path, axis=2), axis=1)) <= 1e-7
            optical += 1
        elif row['phase'] == 'actual_action' and row['reason'] in ('certified_clear', 'shared_clear'):
            P = latest_polygons[row['channel']]
            assert np.max(np.linalg.norm(P-row['position'], axis=1)) <= 20+1e-6
            assert row['result'] == 'success'
            safe_clear += 1
    return dict(passed=True, triangle_certificates=triangles, belief_polygons=polygons,
                optical_cover_plans=optical, single_position_clear_proofs=safe_clear,
                hidden_truth_containment_checked=False)


def analyze_case(path, total_count, omni, directional):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    traces = read(Path(str(path)+'.beliefs.json'))
    starts = [r for r in records if r.get('type') == 'session_start']
    summaries = [r for r in records if r.get('type') == 'session_summary']
    assert len(starts) == len(summaries) == 1
    summary = summaries[0]
    assert summary['problem'] == starts[0]['problem'] == 4
    assert summary['variant'] == starts[0]['variant'] == 'shared'
    assert summary['complete'] and summary['exit_accepted']
    assert not [r for r in records if r.get('type') == 'client_error']
    hashes = summary['source_sha256']
    assert hashes == starts[0]['source_sha256']
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    accepted = [r for r in records if r.get('path')]
    assert accepted[0]['path'] == '/enter' and accepted[-1]['path'] == '/exit'
    assert all(r['response']['accepted'] is True for r in accepted)
    assert accepted[0]['response']['virtual_time_s'] == 0
    assert accepted[-1]['response']['exit_reason'] == 'user_exit'
    assert len({r['request']['request_id'] for r in accepted}) == len(accepted)
    position, tuned, elapsed, distance = np.zeros(2), 1, 0., 0.
    counts = Counter()
    actions = []
    removed, discovered, near = set(), {}, {}
    repeat = {}
    max_error = 0.
    for record in accepted[1:-1]:
        req, response = record['request'], record['response']
        action = record['path'][1:]
        assert action in ('measure', 'clear')
        q = np.array([req['position']['x'], req['position']['y']], float)
        assert np.isfinite(q).all() and (np.abs(q) <= 2_000_000).all()
        c = req['channel']
        assert type(c) is int and 1 <= c <= 20
        movement = float(np.linalg.norm(q-position))
        switch = int(c != tuned) if action == 'measure' else 0
        before = elapsed
        if action == 'measure':
            kind = response['measure_result']
            assert kind in ('direction', 'near', 'no_signal')
            if kind == 'direction':
                assert 0 <= response['svd_deg'] < 360
            if kind != 'no_signal':
                discovered.setdefault(c, response['virtual_time_s'])
            if kind == 'near':
                near[c] = q.copy()
            key = (c, tuple(q), c in removed)
            reading = (kind, response.get('svd_deg'))
            assert key not in repeat or reading == repeat[key]
            repeat[key] = reading
            counts['measure'] += 1
            counts['switch'] += switch
            counts[kind] += 1
            fee = 5+switch
            tuned = c
        else:
            kind = response['clear_result']
            assert kind in ('success', 'no_target_in_range')
            success = kind == 'success'
            counts['clear_success' if success else 'clear_fail'] += 1
            fee = 5 if success else 3
            if success:
                assert c not in removed
                removed.add(c)
        elapsed += movement/5+fee
        distance += movement
        max_error = max(max_error, abs(elapsed-response['virtual_time_s']))
        # Official movement is accumulated in microseconds; local Euclidean lengths are unrounded.
        assert abs(elapsed-response['virtual_time_s']) <= 1e-4
        item = dict(index=len(actions)+1, action=action, channel=c, position=q.tolist(),
            from_position=position.tolist(), movement_m=movement, move_s=movement/5,
            fee_s=fee, switch=switch, duration_s=movement/5+fee,
            start_s=before, time_s=response['virtual_time_s'], response=response,
            **{('measure_result' if action == 'measure' else 'clear_result'):kind})
        actions.append(item)
        position = q
    assert abs(elapsed-summary['total_virtual_s']) <= 1e-4
    assert len(removed) == summary['successful_clear_count'] == total_count == omni+directional
    markers = [r for r in traces if r['phase'] == 'actual_action']
    assert len(markers) == len(actions)
    for marker, event in zip(markers, actions):
        assert marker['event'] == event['index'] and marker['action'] == event['action']
        assert marker['channel'] == event['channel']
        assert np.linalg.norm(np.asarray(marker['position'])-event['position']) <= 1e-7
        assert abs(marker['move_s']-event['move_s']) <= 1e-7
        assert marker['result'] == event.get('measure_result', event.get('clear_result'))
        event['reason'] = marker['reason']
        if marker['reason'] == 'near_clear':
            assert np.linalg.norm(near[event['channel']]-event['position']) <= 1e-7
            assert marker['result'] == 'success'
    replay_result, replay_trace, replay_audit = replay_logged_actions(actions)
    assert_tree_close(replay_trace, traces)
    # JSON serializes integer witness-map keys as strings in the saved certificate.
    assert_tree_close(json.loads(json.dumps(replay_result)), {k:summary[k] for k in replay_result})
    proofs = independent_geometry(traces, actions, summary)
    costs = dict(move=distance/5, measure=5*counts['measure'], switch=counts['switch'],
                 clear_success=5*counts['clear_success'], clear_fail=3*counts['clear_fail'])
    by_reason = {}
    for reason in sorted({a['reason'] for a in actions}):
        rows = [a for a in actions if a['reason'] == reason]
        by_reason[reason] = dict(actions=len(rows), time_s=sum(a['duration_s'] for a in rows),
            move_s=sum(a['move_s'] for a in rows), fee_s=sum(a['fee_s'] for a in rows),
            positive=sum(a.get('measure_result') in ('near', 'direction') for a in rows),
            negative=sum(a.get('measure_result') == 'no_signal' for a in rows),
            clear_fail=sum(a.get('clear_result') == 'no_target_in_range' for a in rows))
    last_clear = max(a['time_s'] for a in actions if a.get('clear_result') == 'success')
    last_discovery = max(discovered.values())
    scan_points = list(dict.fromkeys(tuple(a['position']) for a in actions if a['reason'] == 'scan'))
    per_channel = {}
    for c in sorted(removed):
        rows = [a for a in actions if a['channel'] == c]
        clear = next(a for a in rows if a.get('clear_result') == 'success')
        per_channel[str(c)] = dict(first_detected_s=discovered[c], cleared_s=clear['time_s'],
            clear_position=clear['position'], clear_fail=sum(a.get('clear_result') == 'no_target_in_range' for a in rows),
            local_time_s=sum(a['duration_s'] for a in rows if a['reason'] != 'scan'),
            measured=sum(a['action'] == 'measure' for a in rows))
    predictions = []
    for i, row in enumerate(traces):
        if row['phase'] not in ('decision', 'shared_prediction'):
            continue
        following = next(r for r in traces[i+1:] if r['phase'] == 'actual_action')
        assert following['action'] == 'measure' and following['channel'] == row['channel']
        predictions.append(dict(kind=row['phase'], signal_mass=row['signal_mass'],
                                signal_received=following['result'] != 'no_signal'))
    return dict(case=path.stem, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(Path(str(path)+'.beliefs.json').read_bytes()).hexdigest(),
        total_sources_from_user=total_count, omni_from_user=omni, directional_from_user=directional,
        cleared=len(removed), all_cleared_against_user_total=True, absent=len(summary['certificate']['absent']),
        total_s=summary['total_virtual_s'], average_s=summary['total_virtual_s']/len(removed),
        counts=dict(counts), costs_s=costs, distance_m=distance, by_reason=by_reason,
        passed=True, source_files_checked=len(hashes), source_hashes_match_current=True,
        replay_actions_match=True, replay_trace_matches=True, replay_certificate_matches=True,
        replay_audit=replay_audit,
        geometry_audit=proofs, max_cumulative_timing_error_s=max_error,
        client_errors=0, rejected_actions=0, unique_accepted_requests=len(accepted),
        scan_stops=len(scan_points), scan_points=[list(p) for p in scan_points],
        last_discovered_s=last_discovery, last_clear_s=last_clear,
        post_last_clear_s=summary['total_virtual_s']-last_clear,
        after_last_discovery_s=summary['total_virtual_s']-last_discovery,
        solver_wall_time_s=summary['solver_wall_time_s'],
        local_session_wall_time_s=summary['local_session_wall_time_s'],
        server_timestamp_span_s=(accepted[-1]['response']['real_timestamp_ms']-accepted[0]['response']['real_timestamp_ms'])/1000,
        official_program_runtime_s=summary['official_program_runtime_s'],
        per_channel=per_channel, predictions=predictions,
        actions=[{k:v for k,v in a.items() if k != 'response'} for a in actions])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read(args.input_dir/'manifest.json')
    cases = []
    for i, (n, o, d) in enumerate(zip(manifest['user_source_counts'], manifest['user_omni_counts'],
                                     manifest['user_directional_counts']), 1):
        row = analyze_case(args.input_dir/f'practice-q4-{i:02d}.jsonl', n, o, d)
        cases.append(row)
        print(f'{row["case"]}: {row["cleared"]}/{n}, {row["total_s"]:.6f}s, '
              f'{row["average_s"]:.2f}s/source, replay and certificates passed', flush=True)
    total = sum(r['total_s'] for r in cases)
    sources = sum(r['cleared'] for r in cases)
    cost_totals = {k:sum(r['costs_s'][k] for r in cases) for k in cases[0]['costs_s']}
    reason_totals = {}
    for reason in sorted({k for r in cases for k in r['by_reason']}):
        reason_totals[reason] = {k:sum(r['by_reason'].get(reason, {}).get(k,0) for r in cases)
                                for k in ('actions','time_s','move_s','fee_s','positive','negative','clear_fail')}
    aggregate = dict(runs=len(cases), all_cleared_runs=len(cases), cleared=sources,
        omni_from_user=sum(r['omni_from_user'] for r in cases),
        directional_from_user=sum(r['directional_from_user'] for r in cases),
        total_s=total, mean_total_s=total/len(cases), pooled_average_s=total/sources,
        mean_case_average_s=float(np.mean([r['average_s'] for r in cases])),
        costs_s=cost_totals, cost_shares_pct={k:100*v/total for k,v in cost_totals.items()},
        counts=dict(sum((Counter(r['counts']) for r in cases), Counter())),
        by_reason=reason_totals, post_last_clear_s=sum(r['post_last_clear_s'] for r in cases),
        mean_solver_wall_time_s=float(np.mean([r['solver_wall_time_s'] for r in cases])),
        mean_local_session_wall_time_s=float(np.mean([r['local_session_wall_time_s'] for r in cases])),
        audits={k:sum(r['geometry_audit'][k] for r in cases) for k in
                ('triangle_certificates','belief_polygons','optical_cover_plans','single_position_clear_proofs')},
        max_cumulative_timing_error_s=max(r['max_cumulative_timing_error_s'] for r in cases))
    local = read(Path(__file__).parent/'results/joint_holdout_analysis.json')['groups']['shared']
    report = dict(scope='Five user-supplied official Q4 logs; counts and type composition supplied by user.',
        official_simulator_contacted_by_analysis=False, new_official_test_started=False,
        raw_archive_sha256=manifest['archive_sha256'], cases=cases, aggregate=aggregate,
        local_reference=local,
        cross_cohort_total_difference_pct=100*(aggregate['mean_total_s']/local['mean_total_s']-1),
        cross_cohort_pooled_difference_pct=100*(aggregate['pooled_average_s']/local['pooled_average_s']-1),
        cross_cohort_note='Different layouts and type compositions; not a paired optimization effect.',
        limitations=['No official result sheet, encrypted official log, hidden source coordinates, radii, or headings supplied.',
                     'Cannot verify true-source containment, per-channel type, official runtime, optimality or ranking.',
                     'Offline replay validates the logged policy only; it cannot score unobserved alternative actions.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
