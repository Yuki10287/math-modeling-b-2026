"""Independent complete prune audit of the frozen 50000 three-way experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_ordered_optical_prune_v2 import verify_case
from audit_official_ordered_prune_v1 import charges


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def without_time(events):
    return [{k: v for k, v in row.items() if k != 'time_s'} for row in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    summary = load(args.batch/'summary.json')
    assert summary['passed'] and summary['source_snapshot_stable']
    paths = sorted(args.batch.glob('*-station22_prune.json'))
    assert len(paths) == 10
    rows = []
    for path in paths:
        prefix = path.name.removesuffix('-station22_prune.json')
        base_path = args.batch/(prefix+'-station22.json')
        replay_path = args.batch/(prefix+'-homomorphism.json')
        candidate, baseline, replay = load(path), load(base_path), load(replay_path)
        audit_actual, audit_replay = verify_case(candidate), verify_case(baseline, replay)
        assert without_time(candidate['events']) == without_time(replay['events'])
        assert candidate['result'] == baseline['result'], 'source result or absence proof changed'
        old_times, new_times = charges(baseline['events']), charges(candidate['events'])
        assert max(abs(t-e['time_s']) for t, e in zip(old_times, baseline['events'])) < 1e-7
        assert max(abs(t-e['time_s']) for t, e in zip(new_times, candidate['events'])) < 1e-7
        assert abs(old_times[-1]-replay['original_s']) < 1e-7
        assert abs(new_times[-1]-replay['candidate_s']) < 1e-7
        assert new_times[-1] <= old_times[-1]+1e-7
        rows.append(dict(file=path.name, actual=audit_actual, replay=audit_replay,
            baseline_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            replay_sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            same_retained_actions_and_feedback=True, identical_completion_certificate=True,
            independently_recomputed_saved_s=old_times[-1]-new_times[-1]))
    here = Path(__file__).parent
    result = dict(passed=True, pairs=len(rows), rows=rows,
        candidate_plans=sum(r['actual']['plans'] for r in rows),
        replay_plans=sum(r['replay']['plans'] for r in rows),
        all_optical_actions_accounted=True,
        auditor_sha256={name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
            (Path(__file__).name, 'audit_ordered_optical_prune_v1.py', 'audit_ordered_optical_prune_v2.py',
             'audit_official_ordered_prune_v1.py', 'audit_negative_region_boxes_v1.py')},
        run_manifest_sha256=hashlib.sha256((args.batch/'manifest.json').read_bytes()).hexdigest(),
        official_simulator_contacted=False, no_solver_rerun=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, pairs=len(rows), candidate_plans=result['candidate_plans'],
                         replay_plans=result['replay_plans']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
