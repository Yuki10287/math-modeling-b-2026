"""Read completed official logs; never connects to a simulator or runs a policy.

Produces aggregate, de-identified evidence from accepted responses and trace
markers. No hidden source coordinates or unobserved feedback are reconstructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_holdout_replays import all_channel_evidence_audit
from validate_model import independent_cells, independent_disk_cover

BASE = Path(__file__).resolve().parent


def visited_stop_diagnostic(case, centers):
    """Hindsight geometric capacity of all stops reached by the last clear.

    Failure of the whole-cell test alone does not prove a physical hole. A
    witness strictly inside the target disk and farther than 1000 m from every
    visited stop does prove such a hole for minimum-range sources.
    """
    stops = list(dict.fromkeys(tuple(e['position']) for e in case['actions']
                              if e['time_s'] <= case['last_clear_s'] + 1e-7))
    covered = np.zeros(len(centers), bool)
    nearest = np.full(len(centers), np.inf)
    for point in stops:
        distance = np.linalg.norm(centers-point, axis=1)
        nearest = np.minimum(nearest, distance)
        covered |= distance + 10*np.sqrt(2) <= 1000
    inside = np.linalg.norm(centers, axis=1) <= 1800
    candidate_indices = np.flatnonzero(inside)
    index = candidate_indices[int(np.argmax(nearest[inside]))]
    witness = dict(position=centers[index].tolist(), nearest_visited_distance_m=float(nearest[index]))
    return dict(visited_positions=len(stops), covered_cells=int(covered.sum()),
        total_cells=len(centers), complete=bool(covered.all()),
        physical_coverage_hole_proved=bool(nearest[index]>1000+1e-8),
        farthest_examined_in_domain_point=witness,
        interpretation='All previous stops considered, regardless of channels actually measured; not a rerun of a changed policy.')


def read_case(path, centers, source_dir=None):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    traces = json.loads(Path(str(path) + '.beliefs.json').read_text(encoding='utf-8-sig'))
    summaries = [r for r in records if r.get('type') == 'session_summary']
    assert len(summaries) == 1
    summary = summaries[0]
    assert not [r for r in records if r.get('type') == 'client_error']
    assert summary['complete'] and summary['status'] == 'complete' and summary['exit_accepted']
    assert summary['fallback_count'] == 0
    hashes = summary['source_sha256']
    source_dir = Path(source_dir) if source_dir is not None else BASE
    mismatches = [name for name, value in hashes.items()
                  if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != value]
    assert not mismatches, mismatches
    accepted = [r for r in records if r.get('path')]
    assert accepted[0]['path'] == '/enter' and accepted[-1]['path'] == '/exit'
    assert all(r['response']['accepted'] is True for r in accepted)
    assert accepted[0]['response']['virtual_time_s'] == 0
    actions = []
    position, tuned, elapsed, distance = np.zeros(2), 1, 0., 0.
    seen_requests = set()
    counts = Counter()
    max_error = 0.
    removed = set()
    discovered = {}
    repeated = {}
    per_channel = defaultdict(lambda: dict(measures=0, negatives=0, failures=0, positive=0,
        charged_movement_m=0., charged_time_s=0.))
    for record in accepted:
        request, response = record['request'], record['response']
        request_id = request['request_id']
        assert request_id not in seen_requests, 'Duplicate successful request in log'
        seen_requests.add(request_id)
        if record['path'] not in ('/measure', '/clear'):
            continue
        q = np.array([request['position']['x'], request['position']['y']], float)
        assert np.isfinite(q).all() and (np.abs(q) <= 2_000_000).all()
        channel = request['channel']
        assert 1 <= channel <= 20
        leg = float(np.linalg.norm(q - position))
        action = record['path'][1:]
        before = elapsed
        switch = int(channel != tuned) if action == 'measure' else 0
        outcome = response.get('measure_result', response.get('clear_result'))
        if action == 'measure':
            assert outcome in ('direction', 'near', 'no_signal')
            fee = 5 + switch
            counts['measures'] += 1
            counts['switches'] += switch
            counts[outcome] += 1
            tuned = channel
            per_channel[channel]['measures'] += 1
            per_channel[channel]['negatives' if outcome == 'no_signal' else 'positive'] += 1
            if outcome != 'no_signal':
                discovered.setdefault(channel, (len(actions), before + leg / 5 + fee))
            key = (channel, tuple(q), channel in removed)
            reading = (outcome, response.get('svd_deg'))
            assert key not in repeated or repeated[key] == reading
            counts['repeated_measure_positions'] += int(key in repeated)
            repeated[key] = reading
        else:
            assert outcome in ('success', 'no_target_in_range')
            success = outcome == 'success'
            fee = 5 if success else 3
            counts['clear_success' if success else 'clear_fail'] += 1
            if success:
                assert channel not in removed
                removed.add(channel)
                per_channel[channel]['clear_action_index'] = len(actions)
                per_channel[channel]['successful_clear_position'] = q.tolist()
            else:
                per_channel[channel]['failures'] += 1
        duration = leg / 5 + fee
        elapsed += duration
        distance += leg
        max_error = max(max_error, abs(elapsed - response['virtual_time_s']))
        assert abs(elapsed - response['virtual_time_s']) < 1e-4
        event = dict(index=len(actions), action=action, channel=channel, position=q.tolist(),
            from_position=position.tolist(), movement_m=leg, fee_s=fee, switch=switch,
            start_s=before, time_s=response['virtual_time_s'], duration_s=duration,
            tag='scan' if action == 'measure' else 'unmatched_clear')
        event['measure_result' if action == 'measure' else 'clear_result'] = outcome
        if outcome == 'direction':
            assert 0 <= response['svd_deg'] < 360
            event['svd_deg'] = response['svd_deg']
        actions.append(event)
        per_channel[channel]['charged_movement_m'] += leg
        per_channel[channel]['charged_time_s'] += duration
        position = q
    assert max_error < 1e-4
    assert abs(elapsed - summary['total_virtual_s']) < 1e-4
    assert len(removed) == summary['successful_clear_count'] == summary['cleared']
    assert removed == set(discovered)
    evidence = all_channel_evidence_audit(SimpleNamespace(events=actions), summary, centers)
    cert = summary['completion_certificate']
    absent = set(cert['absent_channels'])
    absent_by_count = set(cert.get('absent_by_count_channels', []))
    assert not (absent & removed)
    assert not (absent_by_count & (absent | removed))
    assert absent | absent_by_count | removed == set(range(1, 21))
    if absent_by_count:
        assert cert['basis'] == 'count_upper_bound' and len(removed) == 16
        assert absent_by_count == set(range(1, 21)) - removed
    assert not cert['unresolved_channels']
    assert all(evidence['per_channel'][str(c)]['complete'] for c in absent)
    cursor = 0
    radius = {}
    proofs = Counter()
    for trace in traces:
        phase = trace['phase']
        if phase == 'localize':
            radius[trace['channel']] = trace['radius_m']
        if phase not in ('local_action', 'shared_bearing', 'opportunity_search', 'shared_clear'):
            continue
        kind = trace['action'] if phase == 'local_action' else 'clear' if phase == 'shared_clear' else 'measure'
        hits = [i for i in range(cursor, len(actions)) if actions[i]['action'] == kind
            and actions[i]['channel'] == trace['channel']
            and np.linalg.norm(np.asarray(actions[i]['position']) - trace['position']) < 1e-7]
        assert hits, (path.name, trace)
        i = hits[0]
        cursor = i + 1
        event = actions[i]
        event['tag'] = ('local_measure' if kind == 'measure' else 'local_clear') if phase == 'local_action' else phase
        event['rationale'] = trace.get('rationale')
        if phase == 'local_action':
            event['prior_radius_m'] = radius.get(trace['channel'])
            event['estimated_s'] = trace.get('estimated_s')
            if kind == 'measure':
                assert event['measure_result'] != 'no_signal', 'Guaranteed reception failed'
                proofs['guaranteed_local_reception_success'] += 1
        if trace.get('certified') and kind == 'clear':
            assert event['clear_result'] == 'success'
            P = np.array(trace['polygon'], float)
            assert np.linalg.norm(P - event['position'], axis=1).max() <= 20 + 1e-5
            proofs['single_clear_cover'] += 1
        if trace.get('optical_path'):
            assert independent_disk_cover(trace['polygon'], trace['optical_path'])['complete']
            proofs['optical_cover_plans'] += 1
    assert not [e for e in actions if e['tag'] == 'unmatched_clear']
    by_tag = {}
    for tag in sorted({e['tag'] for e in actions}):
        group = [e for e in actions if e['tag'] == tag]
        by_tag[tag] = dict(actions=len(group), movement_m=sum(e['movement_m'] for e in group),
            time_s=sum(e['duration_s'] for e in group), fee_s=sum(e['fee_s'] for e in group))
    scan_batches = []
    for e in actions:
        if e['tag'] != 'scan':
            continue
        if not scan_batches or np.linalg.norm(np.array(scan_batches[-1]['position'])-e['position']) > 1e-7:
            scan_batches.append(dict(position=e['position'], start_index=e['index'], end_index=e['index'],
                start_s=e['start_s'], end_s=e['time_s'], movement_m=0., channels=[], discoveries=[]))
        batch = scan_batches[-1]
        batch['end_index'], batch['end_s'] = e['index'], e['time_s']
        batch['movement_m'] += e['movement_m']
        batch['channels'].append(e['channel'])
        if discovered.get(e['channel'], (-1,))[0] == e['index']:
            batch['discoveries'].append(e['channel'])
    last_clear = max(e['time_s'] for e in actions if e.get('clear_result') == 'success')
    for channel in removed:
        item = per_channel[channel]
        clear_event = actions[item['clear_action_index']]
        item['first_detected_s'] = discovered[channel][1]
        item['cleared_s'] = clear_event['time_s']
        item['discovery_to_clear_s'] = item['cleared_s'] - item['first_detected_s']
        own = [e for e in actions if e['channel'] == channel and e['tag'].startswith('local_')]
        item['local_action_count'] = len(own)
        item['local_action_time_s'] = sum(e['duration_s'] for e in own)
    costs = dict(movement=distance/5, measure=5*counts['measures'], switch=counts['switches'],
                 clear_fail=3*counts['clear_fail'], clear_success=5*counts['clear_success'])
    return dict(case=path.stem, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        passed=True, version_matches_frozen=True, source_files_checked=len(hashes),
        accepted_actions=len(actions), cleared=len(removed), absent=len(absent),
        total_s=summary['total_virtual_s'], average_s=summary['average_virtual_s_per_cleared'],
        distance_m=distance, counts=dict(counts), costs_s=costs,
        local_session_wall_s=summary['local_session_wall_time_s'], solver_wall_s=summary['solver_wall_time_s'],
        server_timestamp_span_s=(accepted[-1]['response']['real_timestamp_ms']-accepted[0]['response']['real_timestamp_ms'])/1000,
        official_program_runtime_s=summary['official_program_runtime_s'],
        max_cumulative_timing_error_s=max_error, independent_completion=evidence,
        geometry_proofs=dict(proofs), fallback_count=summary['fallback_count'],
        trace_phase_counts=dict(Counter(t['phase'] for t in traces)),
        by_action_tag=by_tag, scan_batches=scan_batches, per_channel=dict(per_channel),
        first_discovered_s=min(x[1] for x in discovered.values()),
        last_discovered_s=max(x[1] for x in discovered.values()),
        last_clear_s=last_clear, scan_tail_s=summary['total_virtual_s']-last_clear,
        long_legs=sorted(actions,key=lambda e:e['movement_m'],reverse=True)[:8], actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    centers = independent_cells()
    cases = [read_case(path, centers) for path in sorted(args.input_dir.glob('*.jsonl'))]
    assert cases
    for case in cases:
        case['hindsight_pre_last_clear_visited_stop_cover'] = visited_stop_diagnostic(case, centers)
    n = len(cases)
    aggregate = dict(cases=n, cleared=sum(c['cleared'] for c in cases),
        mean_total_s=sum(c['total_s'] for c in cases)/n,
        mean_case_average_s=sum(c['average_s'] for c in cases)/n,
        pooled_average_s=sum(c['total_s'] for c in cases)/sum(c['cleared'] for c in cases),
        mean_scan_tail_s=sum(c['scan_tail_s'] for c in cases)/n,
        accepted_actions=sum(c['accepted_actions'] for c in cases),
        costs_s={k:sum(c['costs_s'][k] for c in cases)/n for k in cases[0]['costs_s']},
        max_timing_error_s=max(c['max_cumulative_timing_error_s'] for c in cases))
    report = dict(analysis_kind='read_only_audit_of_user_supplied_official_logs',
        official_simulator_contacted=False, solver_rerun=False, hidden_truth_available=False,
        all_effective_tests_included='Confirmed by user; 04 and 06 are skipped identifiers.',
        diagnostic_boundary='Visited-stop geometry is hindsight on the original path, not an online-policy improvement claim.',
        aggregate=aggregate, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(aggregate, ensure_ascii=False))
    for c in cases:
        print(json.dumps({k:v for k,v in c.items() if k in ('case','cleared','absent','total_s','average_s',
            'counts','scan_tail_s','last_discovered_s','last_clear_s','by_action_tag','scan_batches',
            'geometry_proofs','max_cumulative_timing_error_s')},ensure_ascii=False))


if __name__ == '__main__':
    main()
