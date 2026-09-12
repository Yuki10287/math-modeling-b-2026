"""Replay declared local stress layouts with the same public-interface audits."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_exploration import hashes, run_case
from benchmark_joint import summary
from validate_model import independent_cells, stress_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--counterexamples', required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.counterexamples.read_text(encoding='utf-8'))
    cases = stress_cases() + payload['cases']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'cases').mkdir()
    before = hashes()
    manifest = dict(local_only=True, official_simulator_used=False, truth_hidden=True,
                    split='stress', cases=cases, source_sha256=before,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    constructed_input_sha256=hashlib.sha256(args.counterexamples.read_bytes()).hexdigest(),
                    note='Constructed counterexamples depend on development trajectories; not a holdout.')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    centers = independent_cells()
    for case in cases:
        for variant in ('lean', 'probe'):
            detail = run_case(case, variant, centers)
            row = detail['result']
            rows.append(row)
            (args.out / 'cases' / f'{case["name"]}-{variant}.json').write_text(
                json.dumps(detail), encoding='utf-8')
            (args.out / 'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps({k: row[k] for k in ('name', 'schedule', 'passed', 'total_s', 'probe_outcomes')}), flush=True)
    stable = before == hashes()
    result = dict(**summary(rows), source_snapshot_stable=stable)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if stable and all(r['passed'] for r in rows) else 1)


if __name__ == '__main__':
    main()
