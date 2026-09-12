"""Post-hoc known-source continuation audit; truth never reaches the selector."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np

import bootstrap
from validate_model import BoundedArena, PublicAPI
from scenario_model import _copy_coverage
from diagnose_forecasts import SnapshotAPI
from conditional_worlds import LedgerModel, clone_conditional
from conditional_preview import conditional_key, joint_candidates
from rollout_solver import continue_lean


def snapshot(model):
    clone = LedgerModel(SnapshotAPI(model.position,model.channel),[],
        use_negatives=model.use_negatives,range_cuts=model.range_cuts,search_fraction=model.search_fraction)
    clone.coverage = _copy_coverage(model.coverage)
    clone.positions = {c:[np.asarray(p).copy() for p in ps] for c,ps in model.positions.items()}
    clone.beliefs = copy.deepcopy(model.beliefs)
    clone.cleared,clone.discovered = set(model.cleared),set(model.discovered)
    clone.history = copy.deepcopy(model.history)
    return clone


def analyze(path):
    saved=json.loads(path.read_text(encoding='utf-8'))
    case=saved['result']
    records=[r for r in saved['trace'] if r.get('phase')=='conditional_forecast']
    arena=BoundedArena(saved['sources'],case['seed'],case['field'],max_actions=3000)
    model=LedgerModel(PublicAPI(arena),[],range_cuts=True,search_fraction=.30)
    captured=[]
    def replay(model,iteration,tasks,route):
        record=next((r for r in records if r['iteration']==iteration),None)
        if record is None: return None
        candidates=joint_candidates(tasks,route)
        assert conditional_key(model,iteration,candidates,6)==record['state_key']
        captured.append((snapshot(model),iteration,candidates,record))
        return candidates[record['chosen_index']] if record['changed'] else None
    result=continue_lean(model,selector=replay)
    assert result['complete'] and arena.events==saved['events'],'recorded conditional replay diverged'
    output=[]
    for model,iteration,candidates,record in captured:
        if record['prediction'] is None:
            output.append(dict(iteration=iteration,prediction_unavailable=True))
            continue
        world={s['channel']:(np.asarray(s['position']),s['radius'])
               for s in saved['sources'] if s['channel'] not in model.cleared}
        costs=[]
        for task in candidates:
            clone,sim=clone_conditional(model,world,field=case['field'],scenario_index=case['seed'])
            outcome=continue_lean(clone,start_iteration=iteration,first_task=task)
            assert outcome['complete'] and sim.evaluation()['all_cleared']
            costs.append(float(sim.time_s))
        prediction=np.asarray([r['costs_s'] for r in record['prediction']['candidates']],dtype=float)
        alternatives=[]
        for i in range(1,len(candidates)):
            actual=costs[i]-costs[0]
            predicted=float(np.mean(prediction[i]-prediction[0]))
            alternatives.append(dict(index=i,actual_delta_s=actual,predicted_delta_s=predicted,
                absolute_delta_error_s=abs(actual-predicted),correct_five_second_decision=bool((actual < -5)==(predicted < -5))))
        output.append(dict(iteration=iteration,known_channels=sorted(model.beliefs),
            candidates=candidates,actual_continuation_s=costs,alternatives=alternatives,
            online_chosen_index=record['chosen_index'],chosen_actual_delta_s=costs[record['chosen_index']]-costs[0]))
    return dict(name=case['name'],exact_online_replay=True,checkpoints=output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',required=True,nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    assert not args.out.exists()
    rows=[]
    for path in args.cases:
        rows.append(analyze(path))
        print(json.dumps(dict(diagnosed=str(path))),flush=True)
    alternatives=[a for row in rows for c in row['checkpoints'] for a in c.get('alternatives',[])]
    summary=dict(comparisons=len(alternatives),mean_absolute_delta_error_s=float(np.mean([
        a['absolute_delta_error_s'] for a in alternatives])) if alternatives else None,
        correct_five_second_decisions=sum(a['correct_five_second_decision'] for a in alternatives))
    args.out.write_text(json.dumps(dict(local_only=True,diagnostic_only=True,official_simulator_used=False,
        hidden_truth_used_only_in_counterfactual_feedback=True,cases=rows,summary=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
