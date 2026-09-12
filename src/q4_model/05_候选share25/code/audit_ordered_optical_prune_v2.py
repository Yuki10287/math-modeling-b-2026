"""Add complete plan/action accounting and certificate-removal challenges.

Version 1 exact geometry is unchanged and retained at its frozen hash. This
wrapper also rejects omission of whole proof records, rather than validating
only whichever records a purported candidate trace happens to supply.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_ordered_optical_prune_v1 import verify_case as verify_geometry, self_tests


def verify_case(case, replay=None):
    trace = case['trace']
    plans = [i for i, row in enumerate(trace) if row['phase'] == 'optical_plan']
    declared = [i for i, row in enumerate(trace) if row['phase'] == 'ordered_optical_prune']
    if replay is None:
        assert [i+1 for i in declared] == plans, 'each original optical plan requires its adjacent pruning proof'
    else:
        assert not declared, 'restricted replay must use the original baseline trace'
        assert len(replay['proofs']) == len(plans), 'each original plan requires one replay proof'
    assigned = []
    for index in plans:
        plan = trace[index]
        succeeded = False
        for action in trace[index+1:]:
            assert action['phase'] == 'actual_action', 'optical block abandoned without a successful clear'
            assert action['reason'] == 'optical_cover' and action['action'] == 'clear'
            assert action['channel'] == plan['channel']
            assigned.append(action['event'])
            if action['result'] == 'success':
                succeeded = True
                break
            assert action['result'] == 'no_target_in_range'
        assert succeeded
    expected = [row['event'] for row in trace if row['phase'] == 'actual_action' and row['reason'] == 'optical_cover']
    assert assigned == expected, 'an actual optical action has no proof, or belongs to duplicate proofs'
    assert len(set(assigned)) == len(assigned)
    result = verify_geometry(case, replay)
    assert result['plans'] == len(plans)
    result.update(all_optical_plans_audited=True, all_actual_optical_actions_accounted=True,
                  actual_optical_actions=len(assigned), auditor_version=2)
    return result


def corruption_challenges(original):
    verify_case(original)
    index = next(i for i, row in enumerate(original['trace']) if row['phase'] == 'ordered_optical_prune')
    trials = []
    case = copy.deepcopy(original)
    case['trace'].pop(index)
    trials.append(('omit_entire_pruning_proof', case))
    case = copy.deepcopy(original)
    del case['trace'][index:index+2]
    trials.append(('omit_proof_and_original_cover_record', case))
    case = copy.deepcopy(original)
    case['trace'].insert(index, copy.deepcopy(case['trace'][index]))
    trials.append(('duplicate_pruning_proof', case))
    case = copy.deepcopy(original)
    p = case['trace'][index]
    p['original_distances_squared'][0] = '999999999999999'
    trials.append(('forge_exact_distance', case))
    case = copy.deepcopy(original)
    p = case['trace'][index]
    p['kept_indices'].reverse()
    trials.append(('reverse_retained_order', case))
    case = copy.deepcopy(original)
    p = case['trace'][index]
    p['original_polygon'][0][0] += 100
    trials.append(('change_source_region_without_external_premise', case))
    case = copy.deepcopy(original)
    p = case['trace'][index]
    p['original_path'][0][0] += 100
    trials.append(('move_original_clearing_point', case))
    case = copy.deepcopy(original)
    p = case['trace'][index]
    p['negatives'].append([123456., 654321.])
    # Change the claimed belief too so the observed-events ledger, not merely
    # comparison to that row, has to reject the invented negative premise.
    prior = [r for r in case['trace'][:index] if r['phase'] == 'belief' and r['channel'] == p['channel']][-1]
    prior['negatives'].append([123456., 654321.])
    trials.append(('invent_unobserved_negative_in_proof_and_belief', case))
    passed = []
    for name, case in trials:
        try:
            verify_case(case)
        except (AssertionError, ValueError):
            passed.append(dict(name=name, forged_record_rejected=True))
        else:
            raise AssertionError('Forgery passed: '+name)
    return dict(passed=True, challenges=len(passed), rows=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    results = Path(__file__).resolve().parent.parent/'results'
    rows = []
    candidate_paths = sorted((results/'ordered_optical_prune_dev48000').glob('*-ordered_prune.json'))
    assert len(candidate_paths) == 5
    for p in candidate_paths:
        case = json.loads(p.read_text(encoding='utf-8'))
        rows.append(dict(suite='candidate48000', file=p.name, input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         audit=verify_case(case)))
    for p in sorted((results/'late_negative_optical_dev41000').glob('*-share25.json')):
        r = results/'ordered_optical_prune_replay41000'/(p.stem+'-replay.json')
        case, replay = json.loads(p.read_text(encoding='utf-8')), json.loads(r.read_text(encoding='utf-8'))
        assert replay['input_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(dict(suite='replay41000', file=p.name, input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         audit=verify_case(case, replay)))
    raw_audit = results/'ordered_optical_prune_independent_official_batch6_audit.json'
    assert json.loads(raw_audit.read_text(encoding='utf-8'))['passed']
    for number in range(1, 5):
        folder = results/'ordered_optical_prune_official_batch6_restricted'
        p, r = folder/f'case-{number:02}-sanitized-input.json', folder/f'case-{number:02}-restricted-replay.json'
        case, replay = json.loads(p.read_text(encoding='utf-8')), json.loads(r.read_text(encoding='utf-8'))
        rows.append(dict(suite='official_batch6_restricted', file=p.name,
                         input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         replay_sha256=hashlib.sha256(r.read_bytes()).hexdigest(), audit=verify_case(case, replay)))
    result = dict(passed=True, cases=len(rows), plans=sum(row['audit']['plans'] for row in rows), rows=rows,
        raw_official_binding_audit_sha256=hashlib.sha256(raw_audit.read_bytes()).hexdigest(),
        exact_distance_tests=self_tests(),
        corruption_challenges=corruption_challenges(json.loads(candidate_paths[0].read_text(encoding='utf-8'))),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_verifier_sha256=hashlib.sha256(Path(__file__).with_name('audit_ordered_optical_prune_v1.py').read_bytes()).hexdigest(),
        old_verifier_unchanged=True, no_solver_rerun=True, official_simulator_contacted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, cases=result['cases'], plans=result['plans'],
                         corruption_challenges=result['corruption_challenges']['challenges']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
