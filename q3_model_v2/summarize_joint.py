"""Summarize the predeclared, paired holdout; fields share layout clusters."""
import json
from pathlib import Path
import numpy as np
from benchmark_joint import hashes, summary

ROOT=Path(__file__).resolve().parent


def main():
    names=['uniform','sparse','boundary','one-side']
    rows=[]
    populations={}
    manifests=[]
    for name in names:
        folder=ROOT/'results'/f'joint-holdout-{name}'
        rs=json.loads((folder/'rows.json').read_text(encoding='utf-8'))
        populations[name]=summary(rs)
        rows.extend(dict(r,population=name) for r in rs)
        manifests.append(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    baseline={r['name']:r for r in rows if r['schedule']=='lean'}
    pairs=[]
    for r in rows:
        if r['schedule']=='lean':continue
        b=baseline[r['name']]
        pairs.append(dict(name=r['name'],seed=r['seed'],field=r['field'],population=r['population'],
            source_count=r['source_count'],baseline_s=b['total_s'],candidate_s=r['total_s'],
            saved_s=b['total_s']-r['total_s'],saved_pct=100*(b['total_s']-r['total_s'])/b['total_s'],
            baseline_tail_s=b['tail_s'],candidate_tail_s=r['tail_s'],passed=b['passed'] and r['passed']))
    rng=np.random.default_rng(128462)
    boot=np.zeros(10000)
    for name in names:
        selected=[p for p in pairs if p['population']==name]
        layouts=sorted({p['seed'] for p in selected})
        deltas=np.array([np.mean([p['saved_s'] for p in selected if p['seed']==s]) for s in layouts])
        boot+=rng.choice(deltas,size=(len(boot),len(deltas)),replace=True).mean(axis=1)*len(selected)/len(pairs)
    selection=json.loads((ROOT/'results'/'joint_selection.json').read_text(encoding='utf-8'))
    stress=json.loads((ROOT/'results'/'joint-stress'/'rows.json').read_text(encoding='utf-8'))
    data=dict(**summary(rows),populations=populations,pairs=pairs,stress=summary(stress),
        independent_layouts=len({(r['population'],r['seed']) for r in pairs}),
        paired_saved_seconds_ci95=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap=dict(resamples=10000,seed=128462,unit='layout; all fields kept together',
            stratified_by_population=True,interpretation='uncertainty under the four self-defined layout generators only'),
        source_snapshot_stable=selection['source_sha256']==hashes() and all(m['source_sha256']==hashes() for m in manifests),
        all_passed=all(r['passed'] for r in rows+stress),official_simulator_used=False,
        expected_conditions=selection['holdout_conditions'],actual_conditions=len(pairs))
    assert data['source_snapshot_stable'] and len(pairs)==60
    (ROOT/'results'/'joint_holdout_summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ('groups','comparisons_to_lean','paired_saved_seconds_ci95','all_passed','source_snapshot_stable')}))


if __name__=='__main__':main()
