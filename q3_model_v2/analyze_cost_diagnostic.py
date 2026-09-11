"""Compare cost rankings on fixed diagnostic alternatives, without retraining."""
import json
from pathlib import Path
import numpy as np
from forecast_policy import measure_score
from local_policy import area_scenarios

ROOT=Path(__file__).resolve().parent


def main():
    rows=json.loads((ROOT/'results/cost-diagnostic-full/rows.json').read_text())
    totals=dict(branches=0,future_measures=0,uncertified_reception=0,repeated_position=0)
    ranks={v:[] for v in ('original','legal','bounded','policy')}
    for row in rows:
        public=row['public_snapshot']
        P,s=np.asarray(public['polygon']),np.asarray(public['position'])
        samples=area_scenarios(P)
        actions=row['alternatives']
        for action in actions:
            check=action.get('forecast_checks')
            if check:
                for key in totals:totals[key]+=check[key]
            action['predictions']={'original':action['predicted_local_s']}
            for variant in ('legal','bounded','policy'):
                action['predictions'][variant]=(action['predicted_local_s'] if action['kind']=='clear' else
                    float(measure_score(P,s,np.asarray(action['position']),samples,public['observed_positions'],variant=variant)
                          +int(public['current_channel']!=row['channel'])))
        for variant in ranks:
            selected=min(actions,key=lambda a:a['predictions'][variant])
            ranks[variant].append(dict(case=row['case'],category=row['category'],chosen=selected['label'],
                local_regret_s=selected['actual_local_s']-min(a['actual_local_s'] for a in actions),
                global_regret_s=selected['actual_remaining_s']-min(a['actual_remaining_s'] for a in actions)))
    result=dict(local_only=True,forecast_checks=totals,
        interpretation='Hindsight regret over a limited candidate set in selected local states; not achievable perfect-information performance and not a causal estimate of approximation error.',
        ranking={v:dict(mean_local_regret_s=float(np.mean([r['local_regret_s'] for r in rs])),
                        mean_global_regret_s=float(np.mean([r['global_regret_s'] for r in rs])),
                        states_with_local_regret_over_1s=sum(r['local_regret_s']>1 for r in rs)) for v,rs in ranks.items()},
        rank_rows=ranks,states=rows)
    (ROOT/'results/cost_diagnostic_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('forecast_checks','ranking')}))


if __name__=='__main__':main()
