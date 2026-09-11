"""Aggregate the predeclared global-value holdout by independent layout."""
import json
from pathlib import Path
import numpy as np
from benchmark_value import hashes, summary

ROOT = Path(__file__).resolve().parent


def main():
    selection = json.loads((ROOT/'results/value_selection.json').read_text())
    rows, populations, manifests = [], {}, []
    for kind in ('uniform', 'sparse', 'boundary', 'one-side'):
        folder = ROOT/f'results/value-holdout-{kind}'
        part = json.loads((folder/'rows.json').read_text())
        populations[kind] = summary(part)
        rows.extend(dict(r, population=kind) for r in part)
        manifests.append(json.loads((folder/'manifest.json').read_text()))
    original = {r['name']: r for r in rows if r['schedule'] == 'original'}
    selected = selection['selected']
    pairs = [dict(name=r['name'], seed=r['seed'], field=r['field'], population=r['population'],
        source_count=r['source_count'], original_s=original[r['name']]['total_s'], selected_s=r['total_s'],
        saved_s=original[r['name']]['total_s']-r['total_s']) for r in rows if r['schedule']==selected]
    rng, boot = np.random.default_rng(449307), np.zeros(10000)
    layout_rows = []
    for kind in populations:
        part = [p for p in pairs if p['population']==kind]
        seeds = sorted({p['seed'] for p in part})
        means = [float(np.mean([p['saved_s'] for p in part if p['seed']==s])) for s in seeds]
        for seed, mean in zip(seeds, means):
            layout_rows.append(dict(population=kind, seed=seed, mean_saved_s=mean))
        boot += rng.choice(means, size=(10000,len(seeds)), replace=True).mean(axis=1)*len(part)/len(pairs)
    stress = json.loads((ROOT/'results/value-stress/rows.json').read_text())
    manifests.append(json.loads((ROOT/'results/value-stress/manifest.json').read_text()))
    frozen = selection['source_sha256']==hashes() and all(m['source_sha256']==hashes() for m in manifests)
    assert frozen and len(pairs)==selection['paired_conditions'] and len(layout_rows)==selection['independent_layouts']
    assert len(stress)==2*selection['stress_cases']
    results = dict(**summary(rows), populations=populations, pairs=pairs, independent_layouts=layout_rows,
        stress=summary(stress), all_passed=all(r['passed'] for r in rows+stress),
        source_snapshot_stable=frozen, paired_saved_s_ci95=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap=dict(unit='layout; keep three error fields together', stratified=True, resamples=10000, seed=449307),
        selected=selected, baseline=selection['baseline_definition'], official_simulator_used=False)
    (ROOT/'results/value_holdout_summary.json').write_text(json.dumps(results,indent=2), encoding='utf-8')
    print(json.dumps({k:results[k] for k in ('groups','comparisons_to_original','paired_saved_s_ci95','all_passed','source_snapshot_stable')}))


if __name__ == '__main__':
    main()
