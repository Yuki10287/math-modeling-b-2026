"""Assemble immutable local holdout evidence and export scientific figures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from benchmark import summarize, code_hashes

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(rows, reference, candidate):
    base = {(r['seed'], r['field']): r for r in rows if r['schedule'] == reference}
    new = {(r['seed'], r['field']): r for r in rows if r['schedule'] == candidate}
    assert base.keys() == new.keys()
    keys = sorted(base)
    a = np.array([base[k]['total_s'] for k in keys])
    b = np.array([new[k]['total_s'] for k in keys])
    delta = a-b
    seeds = sorted({k[0] for k in keys})
    groups = [[i for i, k in enumerate(keys) if k[0] == seed] for seed in seeds]
    source_base = np.array([a[g].sum() for g in groups])
    source_new = np.array([b[g].sum() for g in groups])
    # Resample layouts as clusters; the three fields sharing a layout are not
    # falsely counted as three independent source layouts.
    rng = np.random.default_rng(77219)
    indices = rng.integers(0, len(seeds), size=(5000, len(seeds)))
    bootstrap = 100*(1-source_new[indices].sum(axis=1)/source_base[indices].sum(axis=1))
    order = np.argsort(delta)
    return dict(reference=reference, candidate=candidate, conditions=len(keys), layouts=len(seeds),
        aggregate_reduction_pct=float(100*delta.sum()/a.sum()), mean_saved_s=float(delta.mean()),
        faster=int(np.sum(delta>1e-6)), slower=int(np.sum(delta < -1e-6)), equal=int(np.sum(abs(delta)<=1e-6)),
        layout_cluster_bootstrap_95pct_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
        worst_regression_s=float(max(0, -delta.min())),
        worst_cases=[dict(seed=keys[i][0], field=keys[i][1], reference_s=float(a[i]),
                         candidate_s=float(b[i]), saved_s=float(delta[i]),
                         saved_pct=float(100*delta[i]/a[i])) for i in order[:5]],
        by_field={field: dict(reduction_pct=float(100*sum(base[k]['total_s']-new[k]['total_s'] for k in keys if k[1]==field)
                          /sum(base[k]['total_s'] for k in keys if k[1]==field)))
                  for field in ('smooth', 'hash', 'extreme')})


def main():
    selection = load(ROOT/'results'/'selection.json')
    chosen = selection['chosen_schedule']
    rows, manifests = [], []
    for field in ('smooth', 'hash', 'extreme'):
        path = ROOT/'results'/f'holdout-{field}'
        summary, manifest = load(path/'summary.json'), load(path/'manifest.json')
        assert summary['code_hashes_unchanged']
        assert manifest['code_hashes'] == code_hashes()
        assert manifest['first_seed'] == 4000 and manifest['last_seed'] == 4019
        rows.extend(load(path/'rows.json'))
        manifests.append(manifest)
    assert len(rows) == 180
    assert len({(r['seed'], r['field'], r['schedule']) for r in rows}) == 180
    assert all(r['all_cleared'] and r['certificate_valid'] and not r['error']
               and not r['belief_violations'] and r['max_event_time_error_s'] < 1e-6 for r in rows)
    result = summarize(rows)
    result.update(chosen_schedule=chosen, local_only=True, independent_layouts=20,
        paired_conditions=60, total_solver_runs=180, code_hashes=code_hashes(),
        comparisons={ref: compare(rows, ref, chosen) for ref in ('baseline', 'v1')},
        uncertainty_note='Bootstrap describes this self-defined layout/error population only; no official distribution claim.')
    (ROOT/'results'/'holdout_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'results'/'holdout_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
