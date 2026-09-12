"""Independently audit the uploaded fifth-case prune batch, entirely offline.

Reads original ZIP members in memory. Output contains hashes, aggregate costs
and proof locations, never robot identifiers, request identifiers or raw logs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
ROOT = Q4.parents[1]
sys.path.insert(0, str(Q4))
sys.path.insert(0, str(Q4/'05_候选share25/code'))
from prune_proof import verify_trace
from refined_cover_audit import exact_certificate_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def optical_cost(start, path):
    previous, move = start, 0.
    for point in path:
        move += math.dist(previous, point)/5
        previous = point
    return dict(move_s=move, failed_clear_s=3*(len(path)-1), successful_clear_s=5,
                total_s=move+3*(len(path)-1)+5)


def audit_case(archive, name, total, omni, directional, release):
    raw, raw_trace = archive.read(name), archive.read(name+'.beliefs.json')
    records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    trace = json.loads(raw_trace.decode('utf-8-sig'))
    starts = [r for r in records if r.get('type') == 'session_start']
    summaries = [r for r in records if r.get('type') == 'session_summary']
    require(len(starts) == len(summaries) == 1, 'incomplete or duplicate session envelope')
    start, summary = starts[0], summaries[0]
    require(records[0] is start and records[-1] is summary, 'session envelope is out of order')
    require(start['problem'] == summary['problem'] == 4, 'wrong problem')
    require(start['variant'] == summary['variant'] == 'share25_prune', 'wrong variant')
    require(start['base_variant'] == summary['base_variant'] == 'share25', 'wrong base variant')
    require(start['source_sha256'] == summary['source_sha256'] == release['source_sha256'],
            'session fingerprint does not match frozen release')
    require(len(start['source_sha256']) == 15, 'incomplete runtime fingerprint')
    matches = {key: sha((ROOT/'src'/key).read_bytes()) == value
               for key, value in start['source_sha256'].items()}
    require(all(matches.values()), 'current runtime differs from recorded version')
    calls = [r for r in records if 'path' in r]
    require(len(records) == len(calls)+2, 'nonstandard session or error record')
    require(calls[0]['path'] == '/enter' and calls[-1]['path'] == '/exit', 'missing enter or exit')
    require(all(r['response']['accepted'] is True for r in calls), 'rejected request')
    require(len({r['request']['request_id'] for r in calls}) == len(calls), 'duplicate accepted request ID')
    require(calls[0]['response']['virtual_time_s'] == 0, 'nonzero initial virtual clock')
    require(calls[-1]['response']['exit_reason'] == 'user_exit', 'unexpected exit')
    require(summary['complete'] is True and summary['exit_accepted'] is True, 'session not complete')
    markers = [r for r in trace if r['phase'] == 'actual_action']
    require(len(markers) == len(calls)-2, 'actual action trace omits recorded requests')
    events, accounting, cleared, discovered = [], [], set(), set()
    previous, tuned, elapsed, reported_previous = [0., 0.], 1, 0., 0.
    errors, local_errors, counts = [], [], Counter()
    repetition = {}
    for number, (record, marker) in enumerate(zip(calls[1:-1], markers), 1):
        request, response = record['request'], record['response']
        action, channel = record['path'][1:], request['channel']
        require(action in ('measure', 'clear'), 'unknown action')
        require(type(channel) is int and 1 <= channel <= 20, 'invalid channel')
        point = [request['position']['x'], request['position']['y']]
        require(all(math.isfinite(v) and abs(v) <= 2_000_000 for v in point), 'invalid coordinates')
        kind_key = 'measure_result' if action == 'measure' else 'clear_result'
        kind = response[kind_key]
        move = math.dist(previous, point)/5
        switch = int(channel != tuned) if action == 'measure' else 0
        if action == 'measure':
            require(kind in ('direction', 'near', 'no_signal'), 'invalid measure feedback')
            if kind == 'direction':
                require(math.isfinite(response['svd_deg']) and 0 <= response['svd_deg'] < 360,
                        'invalid observed bearing')
            key = (channel, tuple(point), channel in cleared)
            value = (kind, response.get('svd_deg'))
            require(key not in repetition or repetition[key] == value, 'inconsistent repeated observation')
            repetition[key] = value
            if kind != 'no_signal':
                discovered.add(channel)
            fee, tuned = 5+switch, channel
            counts['measure'] += 1
            counts['switch'] += switch
            counts[kind] += 1
        else:
            require(kind in ('success', 'no_target_in_range'), 'invalid clear feedback')
            fee = 5 if kind == 'success' else 3
            counts['clear_success' if kind == 'success' else 'clear_fail'] += 1
            if kind == 'success':
                require(channel not in cleared, 'duplicate successful clearance')
                cleared.add(channel)
        require(marker['event'] == number and marker['action'] == action and marker['channel'] == channel,
                'trace action differs from accepted HTTP request')
        require(marker['position'] == point and marker['result'] == kind, 'trace position or feedback differs')
        require(abs(marker['move_s']-move) <= 1e-7, 'trace movement differs')
        event = dict(action=action, channel=channel, position=point, time_s=response['virtual_time_s'],
                     **{kind_key: kind})
        if kind == 'direction':
            event['svd_deg'] = response['svd_deg']
        events.append(event)
        accounting.append(dict(event=number, reason=marker['reason'], move_s=move, fee_s=fee,
                               duration_s=move+fee, switch=switch, channel=channel, result=kind))
        elapsed += move+fee
        errors.append(abs(elapsed-event['time_s']))
        local_errors.append(abs(event['time_s']-reported_previous-move-fee))
        require(errors[-1] < 1e-4 and local_errors[-1] <= 1.01e-6, 'official virtual clock disagrees with rules')
        previous, reported_previous = point, event['time_s']
    require(len(cleared) == summary['successful_clear_count'] == total == omni+directional,
            'actual successful channels disagree with reported source total')
    require(cleared == set(summary['cleared']) == discovered, 'cleared and discovered channels disagree')
    require(abs(elapsed-summary['total_virtual_s']) < 1e-4, 'summary total clock disagrees')
    require(summary['total_virtual_s'] == calls[-1]['response']['virtual_time_s'] == events[-1]['time_s'],
            'summary or exit clock differs from last action')
    proof_audit = verify_trace(trace, events)
    require(proof_audit == summary['prune_audit'], 'independent prune audit disagrees with self-report')
    require(proof_audit['skipped_before_success'] == summary['optical_prune_skipped_before_success'],
            'skipped-count summary mismatch')
    require(proof_audit['removed_planned'] == summary['optical_prune_removed_planned'],
            'planned-count summary mismatch')
    completion = exact_certificate_audit(summary, events)
    plans = []
    for index, proof in enumerate(trace):
        if proof['phase'] != 'ordered_optical_prune':
            continue
        actual = []
        for row in trace[index+2:]:
            require(row['phase'] == 'actual_action' and row['reason'] == 'optical_cover', 'broken optical block')
            actual.append(row)
            if row['result'] == 'success':
                break
        require(actual and actual[-1]['result'] == 'success', 'optical block has no success')
        first_success = proof['kept_indices'][len(actual)-1]
        skipped = [i for i in proof['removed_indices'] if i < first_success]
        actual_path = [row['position'] for row in actual]
        original_prefix = proof['original_path'][:first_success+1]
        entry = events[proof['event']-1]['position'] if proof['event'] else [0., 0.]
        actual_cost, original_cost = optical_cost(entry, actual_path), optical_cost(entry, original_prefix)
        saved = original_cost['total_s']-actual_cost['total_s']
        require(saved >= -1e-8, 'pruning increased restored prefix cost')
        require(len(skipped) == len(original_prefix)-len(actual_path), 'restored prefix contains extra change')
        plans.append(dict(trace_index=index, preceding_actual_event=proof['event'], channel=proof['channel'],
            original_plan_points=len(proof['original_path']), planned_removed=len(proof['removed_indices']),
            original_polygon_removed=len(proof['original_removed_indices']),
            additional_contraction_removed=len(set(proof['removed_indices'])-set(proof['original_removed_indices'])),
            contraction_applied=proof['info']['applied'], actual_clear_calls=len(actual),
            successful_original_index=first_success, actual_skipped_before_success=len(skipped),
            actual_skipped_original_indices=skipped, removed_after_success=len(proof['removed_indices'])-len(skipped),
            actual_block_cost_s=actual_cost, restored_original_prefix_cost_s=original_cost,
            proved_avoided_cost_s=saved,
            proof_bound_to_raw_trace_and_preceding_actual_measurements=True))
    require(sum(r['actual_skipped_before_success'] for r in plans) == proof_audit['skipped_before_success'],
            'independent successful-prefix reconstruction mismatch')
    by_reason = {}
    for reason in sorted({a['reason'] for a in accounting}):
        items = [a for a in accounting if a['reason'] == reason]
        by_reason[reason] = dict(actions=len(items), move_s=sum(a['move_s'] for a in items),
                                fee_s=sum(a['fee_s'] for a in items), total_s=sum(a['duration_s'] for a in items))
    last_success = max(i for i, e in enumerate(events) if e.get('clear_result') == 'success')
    missing_channel_measurements = sum(e['action'] == 'measure' and e['channel'] not in cleared for e in events)
    costs = dict(move=sum(a['move_s'] for a in accounting), measure=5*counts['measure'],
                 switch=counts['switch'], clear_success=5*counts['clear_success'], clear_fail=3*counts['clear_fail'])
    return dict(case=Path(name).stem, input_log_sha256=sha(raw), input_trace_sha256=sha(raw_trace),
        input_member=name, trace_member=name+'.beliefs.json', passed=True,
        total_sources_from_user=total, omni_from_user=omni, directional_from_user=directional,
        source_total_basis='User-provided official end-screen count, independently cross-checked against unique successful clear feedback; never inferred from time.',
        cleared=len(cleared), cleared_channels=sorted(cleared), clear_fraction=len(cleared)/total,
        variant=summary['variant'], base_variant=summary['base_variant'], source_files_checked=len(matches),
        hashes_match_release_and_current=all(matches.values()), source_hash_matches=matches,
        unique_accepted_requests=len(calls), actual_actions=len(events), client_errors=0, rejected_requests=0,
        total_virtual_s=summary['total_virtual_s'], per_source_virtual_s=summary['total_virtual_s']/total,
        independently_reconstructed_virtual_s=elapsed, max_cumulative_rounding_error_s=max(errors),
        max_per_action_rounding_error_s=max(local_errors), action_counts=dict(counts), costs_s=costs,
        by_reason=by_reason, final_absence_tail_s=summary['total_virtual_s']-events[last_success]['time_s'],
        final_absence_tail_actions=len(events)-last_success-1,
        measurements_on_eventually_absent_channels=missing_channel_measurements,
        proof_audit=proof_audit, exact_completion_audit=completion, optical_plan_details=plans,
        proved_avoided_cost_s=sum(p['proved_avoided_cost_s'] for p in plans),
        avoided_cost_interpretation='Geometric proof and actual successful prefix restore only the original ordered optical points that were guaranteed to fail. This is conditional cost accounting, not a newly observed official share25 score.',
        official_program_runtime_s=summary['official_program_runtime_s'],
        observed_server_enter_to_exit_s=(calls[-1]['response']['real_timestamp_ms']-calls[0]['response']['real_timestamp_ms'])/1000,
        client_local_session_wall_s=summary['local_session_wall_time_s'],
        client_solver_wall_s=summary['solver_wall_time_s'],
        client_pruning_proof_wall_s=summary['pruning_proof_check_wall_time_s'],
        hidden_truth_positions_available=False, hidden_truth_containment_independently_checked=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'run frozen exact cover auditor with assertions enabled')
    require(not args.out.exists(), 'refuse to overwrite any existing result')
    release_path = Q4/'05_候选share25/results/share25_prune_client_release.json'
    release = json.loads(release_path.read_text(encoding='utf-8'))
    archive_hash = sha(args.zip.read_bytes())
    rows = []
    with zipfile.ZipFile(args.zip) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.jsonl'))
        require(len(names) == 5, 'expected exactly five practice cases')
        expected = [(15, 2, 13), (12, 11, 1), (16, 11, 5), (14, 1, 13), (15, 9, 6)]
        for number, (name, population) in enumerate(zip(names, expected), 1):
            require(name.endswith(f'-{number:02}.jsonl'), 'case numbering does not match supplied order')
            row = audit_case(archive, name, *population, release)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('case', 'passed', 'cleared', 'total_virtual_s',
                               'proof_audit', 'proved_avoided_cost_s')}, ensure_ascii=False), flush=True)
    require(archive_hash == sha(args.zip.read_bytes()), 'input ZIP changed during audit')
    result = dict(passed=True, archive_filename=args.zip.name, archive_sha256=archive_hash,
        cases=rows, cases_checked=5, cleared_total=sum(r['cleared'] for r in rows),
        supplied_source_total=sum(r['total_sources_from_user'] for r in rows),
        total_virtual_s=sum(r['total_virtual_s'] for r in rows),
        average_case_virtual_s=sum(r['total_virtual_s'] for r in rows)/5,
        pooled_per_source_virtual_s=sum(r['total_virtual_s'] for r in rows)/sum(r['cleared'] for r in rows),
        equal_weight_case_per_source_s=sum(r['per_source_virtual_s'] for r in rows)/5,
        optical_plans=sum(r['proof_audit']['plans'] for r in rows),
        planned_removed=sum(r['proof_audit']['removed_planned'] for r in rows),
        actual_skipped_before_success=sum(r['proof_audit']['skipped_before_success'] for r in rows),
        proved_avoided_cost_s=sum(r['proved_avoided_cost_s'] for r in rows),
        release_manifest_sha256=sha(release_path.read_bytes()),
        auditors_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes()) for p in (
            Path(__file__), Q4/'prune_proof.py', Q4/'negative_region_proof.py',
            Q4/'05_候选share25/code/refined_cover_audit.py', Q4/'polar_cover.py', Q4/'directional_cover.py')},
        no_official_simulator_contacted=True, input_members_read_in_memory_only=True,
        original_zip_modified=False, original_source_or_results_modified=False,
        formal_test_ids_and_official_runtime_available=False,
        scope='Official practice log audit with user-reported totals. New practice outcomes are separated from conditional restored-prefix cost accounting and from any prior batch.')
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
