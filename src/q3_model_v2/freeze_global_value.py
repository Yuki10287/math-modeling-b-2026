"""Select on development and predeclare the untouched local holdout."""
import json
from datetime import datetime, timezone
from pathlib import Path
from benchmark_value import hashes, summary

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for kind in ('uniform', 'sparse', 'boundary'):
        folder = ROOT / f'results/value-dev-{kind}'
        assert json.loads((folder/'summary.json').read_text())['source_snapshot_stable']
        assert json.loads((folder/'manifest.json').read_text())['source_sha256'] == hashes()
        rows += json.loads((folder/'rows.json').read_text())
    assert len(rows) == 60 and all(r['passed'] for r in rows)
    development = summary(rows)
    selected = min(development['comparisons_to_original'],
        key=lambda v: development['groups'][v]['mean_total_s'])
    result = dict(frozen_at=datetime.now(timezone.utc).isoformat(), selected=selected, baseline='original',
        baseline_definition='Previous route_interleave candidate, not lean or the rejected bounded-cost variant.',
        selection_basis='Smallest development mean among four variants; no performance claim from development.',
        development=development, source_sha256=hashes(), independent_layouts=20, paired_conditions=60,
        holdout_plan=[dict(population=p, start_seed=s, seeds=5, fields=['smooth','hash','extreme']) for p, s in
                      [('uniform',10000), ('sparse10',10100), ('boundary10',10200), ('one_side10',10300)]],
        stress_cases=9, official_simulator_used=False,
        decision_rule='Keep code and parameters fixed on holdout; report all pairs and clustered uncertainty; do not change official entry automatically.')
    target = ROOT / 'results/value_selection.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(selected=selected, development=development['comparisons_to_original'],
        new_layouts=20, new_conditions=60, source_files=len(hashes()))))


if __name__ == '__main__':
    main()
