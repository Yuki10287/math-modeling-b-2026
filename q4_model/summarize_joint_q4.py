"""Summarize complete Q4 paired results; bootstrap independent layouts, not fields."""
import json
import statistics
from pathlib import Path
import numpy as np
from benchmark_joint import hashes


def main():
    root=Path(__file__).parent/'results'
    selection=json.loads((root/'joint_selection.json').read_text(encoding='utf-8'))
    rows=json.loads((root/'joint-holdout/rows.json').read_text(encoding='utf-8'))
    assert len(rows)==48 and all(r['audit']['passed'] and r['all_cleared'] for r in rows)
    manifest=json.loads((root/'joint-holdout/manifest.json').read_text(encoding='utf-8'))
    assert hashes()==selection['code_hashes']==manifest['code_hashes']
    assert json.loads((root/'joint-holdout/summary.json').read_text())['source_snapshot_stable']
    groups={}
    for variant in ('baseline','shared'):
        rr=[r for r in rows if r['variant']==variant]
        groups[variant]=dict(runs=len(rr),all_cleared=sum(r['all_cleared'] for r in rr),
            source_instances=sum(r['source_count'] for r in rr),
            mean_total_s=statistics.mean(r['total_s'] for r in rr),
            mean_case_average_s=statistics.mean(r['average_s'] for r in rr),
            pooled_average_s=sum(r['total_s'] for r in rr)/sum(r['cleared'] for r in rr),
            mean_runtime_s=statistics.mean(r['runtime_s'] for r in rr),max_runtime_s=max(r['runtime_s'] for r in rr),
            costs_s={k:statistics.mean(r['time_parts_s'][k] for r in rr) for k in rr[0]['time_parts_s']})
    pairs=[]
    for a in [r for r in rows if r['variant']=='shared']:
        b=next(r for r in rows if r['variant']=='baseline' and all(r[k]==a[k] for k in ('seed','population','field')))
        assert a['source_count']==b['source_count']
        pairs.append({k:a[k] for k in ('seed','population','field')}|dict(baseline_s=b['total_s'],candidate_s=a['total_s'],saved_s=b['total_s']-a['total_s']))
    populations={}
    for pop in ('uniform','boundary','cluster','all'):
        pp=[r for r in pairs if pop=='all' or r['population']==pop]
        old=statistics.mean(r['baseline_s'] for r in pp);new=statistics.mean(r['candidate_s'] for r in pp)
        populations[pop]=dict(conditions=len(pp),baseline_mean_s=old,candidate_mean_s=new,mean_saved_s=old-new,
            reduction_pct=100*(old-new)/old,faster=sum(r['saved_s']>1e-6 for r in pp),slower=sum(r['saved_s']< -1e-6 for r in pp))
    rng=np.random.default_rng(70604);old_boot=[];new_boot=[]
    for pop in ('uniform','boundary','cluster'):
        pp=[r for r in pairs if r['population']==pop];seeds=sorted(set(r['seed'] for r in pp))
        assert len(seeds)==4
        old=np.array([statistics.mean(r['baseline_s'] for r in pp if r['seed']==seed) for seed in seeds])
        new=np.array([statistics.mean(r['candidate_s'] for r in pp if r['seed']==seed) for seed in seeds])
        draw=rng.integers(0,4,size=(10000,4))
        old_boot.append(old[draw].mean(axis=1));new_boot.append(new[draw].mean(axis=1))
    old=np.mean(old_boot,axis=0);new=np.mean(new_boot,axis=0)
    development=[r for folder in ('joint-development','joint-development-2') for r in json.loads((root/folder/'rows.json').read_text())]
    audits=[r['audit'] for r in rows+development]
    # Complete trajectories allow after-last-clear tail and reception diagnostics.
    diagnostics={}
    for variant in ('baseline','shared'):
        tail=[];received=0;predicted=0.;count=0;scan_move=[]
        for r in [r for r in rows if r['variant']==variant]:
            path=root/'joint-holdout'/f'{r["seed"]}-{r["population"]}-{r["field"]}-{variant}.json'
            case=json.loads(path.read_text(encoding='utf-8'));events=case['events']
            last_clear=max(i for i,e in enumerate(events) if e['action']=='clear' and e['clear_result']=='success')
            tail.append(events[-1]['time_s']-events[last_clear]['time_s'])
            known=set();position=np.zeros(2);travel=0.
            for event in events:
                q=np.array(event['position']);c=event['channel']
                if event['action']=='measure':
                    if c not in known:travel+=float(np.linalg.norm(q-position))/5
                    if event['measure_result']!='no_signal':known.add(c)
                position=q
            scan_move.append(travel)
        diagnostics[variant]=dict(mean_tail_s=statistics.mean(tail),mean_scan_approach_move_s=statistics.mean(scan_move))
    result=dict(selected='shared',baseline='previous_q4_active_31_stations',conditions=24,independent_layouts=12,
        groups=groups,populations=populations,pairs=pairs,
        bootstrap=dict(replicates=10000,seed=70604,unit='layout; both fields stay together; stratified by population',
            mean_saved_s_ci95=np.quantile(old-new,[.025,.975]).tolist(),reduction_pct_ci95=np.quantile(100*(old-new)/old,[.025,.975]).tolist()),
        worst_regression_s=max(0.,-min(r['saved_s'] for r in pairs)),smallest_saving=min(pairs,key=lambda r:r['saved_s']),
        audits=dict(runs=len(audits),source_instances=sum(r['source_count'] for r in rows+development),
            events=sum(a['events'] for a in audits),polygons=sum(a['polygons'] for a in audits),
            optical_plans=sum(a['optical_plans'] for a in audits),triangle_certificates=sum(a['triangle_certificates'] for a in audits)),
        diagnostics=diagnostics,source_snapshot_stable=True,all_passed=all(a['passed'] for a in audits),
        official_simulator_used=False,local_only=True)
    (root/'joint_holdout_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
