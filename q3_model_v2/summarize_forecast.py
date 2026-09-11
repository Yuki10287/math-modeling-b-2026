"""Frozen cost-model holdout summary with layout-cluster uncertainty."""
import json
from pathlib import Path
import numpy as np
from benchmark_forecast import hashes,summary

ROOT=Path(__file__).resolve().parent


def main():
    rows=[];populations={};manifests=[]
    for kind in ('uniform','sparse','boundary'):
        folder=ROOT/'results'/f'cost-holdout-{kind}'
        part=json.loads((folder/'rows.json').read_text())
        populations[kind]=summary(part)
        rows.extend(dict(r,population=kind) for r in part)
        manifests.append(json.loads((folder/'manifest.json').read_text()))
    base={r['name']:r for r in rows if r['schedule']=='original'}
    pairs=[]
    for r in rows:
        if r['schedule']=='original':continue
        b=base[r['name']]
        pairs.append(dict(name=r['name'],seed=r['seed'],field=r['field'],population=r['population'],
            original_s=b['total_s'],bounded_s=r['total_s'],saved_s=b['total_s']-r['total_s'],
            source_count=r['source_count'],original_runtime_s=b['runtime_s'],bounded_runtime_s=r['runtime_s']))
    rng=np.random.default_rng(235907)
    boot=np.zeros(10000)
    for kind in populations:
        part=[p for p in pairs if p['population']==kind]
        seeds=sorted({p['seed'] for p in part})
        ds=np.array([np.mean([p['saved_s'] for p in part if p['seed']==s]) for s in seeds])
        boot+=rng.choice(ds,size=(10000,len(ds)),replace=True).mean(axis=1)*len(part)/len(pairs)
    selection=json.loads((ROOT/'results/cost_selection.json').read_text())
    stress=json.loads((ROOT/'results/cost-stress/rows.json').read_text())
    audit=json.loads((ROOT/'results/cost_polygon_audit.json').read_text())
    manifests.append(json.loads((ROOT/'results/cost-stress/manifest.json').read_text()))
    frozen=selection['source_sha256']==hashes() and all(m['source_sha256']==hashes() for m in manifests)
    result=dict(**summary(rows),populations=populations,pairs=pairs,stress=summary(stress),
        original_audit_all_passed=all(r['passed'] for r in rows+stress),
        all_completed=all(r['all_cleared'] for r in rows+stress),
        supplemental_audit_all_passed=audit['supplemental_audit_passed']==len(rows+stress),
        supplemental_audit_file='cost_polygon_audit.json',source_snapshot_stable=frozen,
        independent_layouts=10,conditions=len(pairs),paired_saved_s_ci95=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap=dict(unit='layout; keep three error fields together',stratified=True,resamples=10000,seed=235907),
        baseline='prior route_interleave candidate; not lean',official_simulator_used=False)
    assert frozen and len(pairs)==selection['paired_conditions']
    (ROOT/'results/cost_holdout_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('groups','comparisons_to_original','paired_saved_s_ci95',
        'original_audit_all_passed','all_completed','supplemental_audit_all_passed','source_snapshot_stable')}))


if __name__=='__main__':main()
