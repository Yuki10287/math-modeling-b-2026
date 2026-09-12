"""Independent v2 audit of twenty saved share25 pruning controls; no solve."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_ordered_optical_prune_v2 import verify_case
from audit_official_ordered_prune_v1 import charges


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    here = Path(__file__).resolve().parent
    project = here.parents[3]
    results = here.parent/'results'
    summary = load(args.controls/'summary.json')
    assert summary['passed'] and summary['code_and_inputs_stable']
    rows, batches = [], {}
    for label, baseline_folder, expected_layouts in (
            ('holdout50000', 'station22_ordered_prune_holdout50000', 10),
            ('pressure53000', 'station22_ordered_prune_pressure53000', 5)):
        paths = sorted((args.controls/label).glob('*-share25-replay.json'))
        assert len(paths) == 10
        batch_rows, seeds = [], set()
        for path in paths:
            replay = load(path)
            input_path = (project/replay['input_path']).resolve()
            assert input_path.parent == (results/baseline_folder).resolve()
            assert input_path.name.endswith('-share25.json')
            assert replay['input_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
            case = load(input_path)
            assert case['summary']['variant'] == 'share25'
            seeds.add(case['summary']['seed'])
            audit = verify_case(case, replay)
            old_times, new_times = charges(case['events']), charges(replay['events'])
            assert max(abs(t-e['time_s']) for t, e in zip(old_times, case['events'])) < 1e-7
            assert max(abs(t-e['time_s']) for t, e in zip(new_times, replay['events'])) < 1e-7
            assert abs(old_times[-1]-replay['original_s']) < 1e-7
            assert abs(new_times[-1]-replay['candidate_s']) < 1e-7
            saved = old_times[-1]-new_times[-1]
            assert abs(saved-replay['saved_s']) < 1e-7
            assert saved >= 3*audit['removed_events']-1e-7
            assert case['events'][-1]['position'] == replay['events'][-1]['position']
            row = dict(batch=label, input_path=replay['input_path'], input_sha256=replay['input_sha256'],
                replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seed=case['summary']['seed'], field=case['summary']['field'],
                audit=audit, original_s=old_times[-1], candidate_s=new_times[-1],
                saved_s=saved, retained_feedback_identical=True, original_first_success_retained=True)
            rows.append(row)
            batch_rows.append(row)
        assert len(seeds) == expected_layouts
        before = sum(r['original_s'] for r in batch_rows)/10
        after = sum(r['candidate_s'] for r in batch_rows)/10
        declared = summary['batches'][label]
        assert abs(before-declared['means_s']['share25']) < 1e-7
        assert abs(after-declared['means_s']['share25_prune']) < 1e-7
        batches[label] = dict(condition_pairs=10, independent_layouts=len(seeds),
            plans=sum(r['audit']['plans'] for r in batch_rows),
            removed_events=sum(r['audit']['removed_events'] for r in batch_rows),
            original_mean_s=before, candidate_mean_s=after,
            faster=sum(r['saved_s'] > 1e-7 for r in batch_rows),
            slower=sum(r['saved_s'] < -1e-7 for r in batch_rows))
    result = dict(passed=True, control_replays=len(rows), batches=batches, rows=rows,
        all_optical_plans_and_actions_audited=True,
        input_summary_sha256=hashlib.sha256((args.controls/'summary.json').read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256((args.controls/'manifest.json').read_bytes()).hexdigest(),
        auditor_sha256={name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in (
            Path(__file__).name, 'audit_ordered_optical_prune_v1.py', 'audit_ordered_optical_prune_v2.py',
            'audit_official_ordered_prune_v1.py', 'audit_negative_region_boxes_v1.py')},
        no_solver_rerun=True, official_simulator_contacted=False,
        interpretation='Restricted pruning of saved original share25 trajectories, not twenty new solver runs.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, control_replays=len(rows), batches=batches), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
