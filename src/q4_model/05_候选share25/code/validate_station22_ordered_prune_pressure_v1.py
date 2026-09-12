"""Frozen combination pressure check: five new layouts, two error conditions.

Reuse the unchanged 50000 combination. Only the local environment error field
and its result label are supplied as a parameter. These are ten conditions on
five layouts, not ten independent new layouts. No network or official calls.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import benchmark_station22_ordered_prune_v1 as frozen

HERE = Path(__file__).resolve().parent
FREEZE = HERE.parent/'results/station22_ordered_prune_holdout50000/manifest.json'
FIELDS = ('constant','extreme')


def field_runner():
    tree = ast.parse(Path(frozen.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    class ParameterizeField(ast.NodeTransformer):
        count = 0
        def visit_Constant(self,node):
            if node.value == 'smooth':
                self.count += 1
                return ast.copy_location(ast.Name(id='field',ctx=ast.Load()),node)
            return node
    visitor = ParameterizeField()
    function = visitor.visit(function)
    assert visitor.count == 2
    function.args.args.append(ast.arg(arg='field'))
    private = frozen.run_case.__globals__.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(Path(__file__)),'exec'),private)
    return private['run_case']


def check_freeze():
    frozen.check_frozen()
    assert frozen.hashes() == json.loads(FREEZE.read_text(encoding='utf-8'))['code_hashes']


def comparison(pairs):
    totals = {v:sum(p[v+'_s'] for p in pairs) for v in frozen.VARIANTS}
    a,b,c = [totals[v] for v in frozen.VARIANTS]
    return dict(condition_pairs=len(pairs),independent_layouts=len({p['seed'] for p in pairs}),
        means_s={v:totals[v]/len(pairs) for v in frozen.VARIANTS},
        combined_reduction_percent=100*(a-c)/a,station22_reduction_percent=100*(a-b)/a,
        prune_increment_percent=100*(b-c)/b,
        faster_vs25=sum(p['combined_saved_s']>1e-7 for p in pairs),
        slower_vs25=sum(p['combined_saved_s']< -1e-7 for p in pairs),
        ties_vs25=sum(abs(p['combined_saved_s'])<=1e-7 for p in pairs),
        faster_vs22=sum(p['prune_increment_s']>1e-7 for p in pairs),
        slower_vs22=sum(p['prune_increment_s']< -1e-7 for p in pairs),
        removed_actual=sum(p['removed_actual'] for p in pairs),
        worst_pair=min(pairs,key=lambda p:p['combined_reduction_percent']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    check_freeze()
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runner,combined = field_runner(),frozen.make_combination()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    frozen.prune.write(out/'manifest.json',dict(code_hashes=frozen.hashes(),driver_sha256=driver_hash,
        combination_manifest_sha256=hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
        variants=frozen.VARIANTS,populations=frozen.POPULATIONS,fields=FIELDS,
        seeds=[53000+100*i for i in range(5)],independent_layouts=5,condition_pairs=10,
        runner_changes='Only two original smooth constants (LocalArena field and summary label) parameterized.',
        no_parameter_tuning=True,models_frozen_before_run=True,official_contacted=False))
    rows,pairs,invariants = [],[],[]
    for i,population in enumerate(frozen.POPULATIONS):
        seed = 53000+100*i
        for field in FIELDS:
            cases = {}
            for variant in frozen.VARIANTS:
                case = runner(seed,population,variant,combined,field)
                path = out/f'{seed}-{population}-{field}-{variant}.json'
                frozen.prune.write(path,case)
                invariants.append(frozen.scan_audit(path))
                rows.append(case['summary'])
                cases[variant] = case
                print(f'{seed} {population} {field} {variant}: {case["summary"]["total_s"]:.3f}s; '
                      f'{case["summary"]["cleared"]}/{case["summary"]["source_count"]}; audit=True',flush=True)
            original,new = cases['station22'],cases['station22_prune']
            replay = frozen.prune.constrained_replay(original)
            replay['independent_exact_prune'] = frozen.exact_prune_audit(original,replay)
            same = [frozen.prune.event_without_time(e) for e in replay['events']] == [frozen.prune.event_without_time(e) for e in new['events']]
            assert same
            error = max(abs(a['time_s']-b['time_s']) for a,b in zip(replay['events'],new['events']))
            assert error < 1e-6 and original['result'] == new['result']
            frozen.prune.write(out/f'{seed}-{population}-{field}-homomorphism.json',replay)
            a,b,c = [cases[v]['summary']['total_s'] for v in frozen.VARIANTS]
            pairs.append(dict(seed=seed,population=population,field=field,share25_s=a,station22_s=b,station22_prune_s=c,
                combined_saved_s=a-c,combined_reduction_percent=100*(a-c)/a,station22_saved_s=a-b,
                prune_increment_s=b-c,prune_reduction_percent=100*(b-c)/b,identical_retained_actions=same,
                identical_completion_certificate=True,time_error_s=error,removed_actual=replay['removed_actual'],
                removed_original_only=replay['removed_original_only'],additional_from_contraction=replay['additional_from_contraction']))
    check_freeze()
    stable = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == driver_hash
    summary = dict(passed=stable and all(r['audit']['passed'] for r in rows),source_snapshot_stable=stable,
        official_contacted=False,pairs=pairs,overall=comparison(pairs),
        by_field={f:comparison([p for p in pairs if p['field']==f]) for f in FIELDS},
        groups={v:dict(source_condition_instances=sum(r['source_count'] for r in rows if r['variant']==v),
            cleared=sum(r['cleared'] for r in rows if r['variant']==v),
            passes=sum(r['audit']['passed'] for r in rows if r['variant']==v),
            mean_wall_s=float(np.mean([r['runtime_s'] for r in rows if r['variant']==v]))) for v in frozen.VARIANTS},
        warning='Five new layouts each evaluated twice; do not count as ten independent layouts or pool with smooth holdout as twenty layouts.')
    frozen.prune.write(out/'rows.json',rows)
    frozen.prune.write(out/'scan_invariant_audit.json',dict(passed=True,rows=invariants))
    frozen.prune.write(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
