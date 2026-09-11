"""Re-rank prior controlled branches using the full public global state.

The 42 recorded branch outcomes are reused, not rerun or treated as holdout.
Baseline replay must match exactly. No truth is exposed to ranking.
"""
import copy
import json
from pathlib import Path
import numpy as np
import geometry as core
import value_solver
from diagnose_cost import CASES, options
from global_value_policy import GlobalContext, candidates
from validate_model import BoundedArena, PublicAPI

ROOT = Path(__file__).resolve().parent


def main():
    old = json.loads((ROOT/'results/cost-diagnostic-full/rows.json').read_text())
    rows = []
    for kind, name in CASES:
        data = json.loads((ROOT/f'results/joint-holdout-{kind}/cases/{name}-route_interleave.json').read_text())
        arena = BoundedArena(data['sources'], data['result']['seed'], data['result']['field'], max_actions=3000)
        seen = set()
        def capture(model, channel):
            radius = core.mec(model.beliefs[channel]['P'])[1]
            category = 'broad' if radius > 180 else 'optical' if radius > 20.01 else None
            if category is None or category in seen or model.beliefs[channel]['state'] is not None:
                return
            seen.add(category)
            saved = next(r for r in old if r['case'] == name and r['category'] == category)
            assert len(arena.events) == saved['prefix_actions']
            before = model.coverage._excluded.copy()
            beliefs_before = copy.deepcopy(model.beliefs)
            actions = options(model, channel)
            context = GlobalContext(model, channel)
            actual = {a['label']: a for a in saved['alternatives']}
            feature_rows = []
            for action in actions:
                previous = actual[action['label']]
                assert np.allclose(previous['position'], action['q'], atol=1e-9, rtol=0)
                terms = {v: context.correction(action, v) for v in ('route', 'coverage', 'shared')}
                feature_rows.append(dict(label=action['label'], kind=action['kind'], position=previous['position'],
                    local_score_s=action['estimated_s'], global_terms=terms,
                    actual_local_s=previous['actual_local_s'], actual_remaining_s=previous['actual_remaining_s']))
            # Check that the shortlist still contains the exact old local choice.
            first = candidates(model, channel)[0]
            assert first['kind'] == actions[0]['kind'] and np.allclose(first['q'], actions[0]['q'], atol=1e-9, rtol=0)
            assert np.array_equal(before, model.coverage._excluded)
            for c in beliefs_before:
                assert np.array_equal(beliefs_before[c]['P'], model.beliefs[c]['P'])
            rankings = {}
            for variant in ('original', 'route', 'coverage', 'shared'):
                chosen = min(feature_rows, key=lambda a: a['local_score_s'] +
                    (0 if variant == 'original' else a['global_terms'][variant]['correction_s']))
                rankings[variant] = dict(chosen=chosen['label'],
                    local_regret_s=chosen['actual_local_s']-min(a['actual_local_s'] for a in feature_rows),
                    global_regret_s=chosen['actual_remaining_s']-min(a['actual_remaining_s'] for a in feature_rows))
            rows.append(dict(case=name, category=category, channel=channel, alternatives=feature_rows, rankings=rankings))
        outcome = value_solver.solve_multi(PublicAPI(arena), decision_hook=capture)
        assert outcome['complete'] and arena.events == data['events']
        print(json.dumps(dict(case=name, states=len(seen), baseline_replay_exact=True)), flush=True)
    result = dict(local_only=True, official_simulator_used=False, rows=rows,
        states=len(rows), branches=sum(len(r['alternatives']) for r in rows),
        baseline_replay_exact=True, shortlist_retains_local_optimum=True, evidence_unchanged_by_scoring=True,
        interpretation='Selected former holdout states are now development; hindsight over a finite set, not attainable oracle value.',
        rankings={v: dict(mean_global_regret_s=float(np.mean([r['rankings'][v]['global_regret_s'] for r in rows])),
            mean_local_regret_s=float(np.mean([r['rankings'][v]['local_regret_s'] for r in rows])),
            changed=sum(r['rankings'][v]['chosen'] != r['rankings']['original']['chosen'] for r in rows))
            for v in ('original', 'route', 'coverage', 'shared')})
    (ROOT/'results/value_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
