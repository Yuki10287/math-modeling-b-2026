"""Extraction-equivalence and optimized-Python checks for runtime proof code."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
sys.path.insert(0, str(Q4))
from prune_proof import verify_trace
import audit_ordered_optical_prune_v2 as historical_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ExplicitChecks(ast.NodeTransformer):
    def visit_Assert(self, node):
        self.generic_visit(node)
        message = node.msg or ast.Constant(value='Proof obligation failed: '+ast.unparse(node.test)[:180])
        return ast.copy_location(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=[ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError', ctx=ast.Load()),
                args=[message], keywords=[]), cause=None)], orelse=[]), node)


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text(encoding='utf-8-sig')).body
            if isinstance(node, ast.FunctionDef)}


def check_extraction():
    runtime_negative = functions(Q4/'negative_region_proof.py')
    original_negative = functions(HERE/'audit_negative_region_boxes_v1.py')
    runtime_optical = functions(Q4/'prune_proof.py')
    original_optical = functions(HERE/'audit_ordered_optical_prune_v1.py')
    count = 0
    for name, actual in runtime_negative.items():
        expected = ExplicitChecks().visit(copy.deepcopy(original_negative[name]))
        require(ast.dump(actual) == ast.dump(expected), 'negative checker extraction changed '+name)
        count += 1
    for name, actual in runtime_optical.items():
        if name == 'verify_trace':
            continue
        if name == '_verify_complete':
            expected = functions(HERE/'audit_ordered_optical_prune_v2.py')['verify_case']
            for node in ast.walk(expected):
                if isinstance(node, ast.Name) and node.id == 'verify_geometry':
                    node.id = '_verify_geometry'
        else:
            expected = copy.deepcopy(original_optical['verify_case' if name == '_verify_geometry' else name])
        expected.name = name
        expected = ExplicitChecks().visit(expected)
        require(ast.dump(actual) == ast.dump(expected), 'optical checker extraction changed '+name)
        count += 1
    imports = []
    for name in ('negative_region_proof.py', 'prune_proof.py'):
        tree = ast.parse((Q4/name).read_text(encoding='utf-8'))
        require(not any(isinstance(n, ast.Assert) for n in ast.walk(tree)), 'optimized-away assertion remains')
        imports.extend(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    require(not any('audit_' in name or 'experiment' in name or 'producer' in name for name in imports), 'research import in runtime')
    return dict(passed=True, equivalent_functions=count, all_assertions_explicit=True, imports=imports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'existing result would be overwritten')
    extraction = check_extraction()
    files = []
    results = HERE.parent/'results'
    for folder, pattern in (('ordered_optical_prune_dev48000', '*-ordered_prune.json'),
                            ('station22_ordered_prune_holdout50000', '*-station22_prune.json'),
                            ('station22_ordered_prune_pressure53000', '*-station22_prune.json')):
        files.extend(sorted((results/folder).glob(pattern)))
    require(len(files) == 25, 'expected 25 already validated historical cases')
    rows = []
    for path in files:
        case = json.loads(path.read_text(encoding='utf-8'))
        audit = verify_trace(case['trace'], case['events'])
        historical = historical_audit.verify_case(case)
        require(audit['passed'] and audit['plans'] == historical['plans'], 'historical audit mismatch')
        require(audit['removed_planned'] == sum(r['removed'] for r in historical['rows']), 'planned deletion mismatch')
        rows.append(dict(input_path=path.relative_to(Q4.parent.parent).as_posix(),
                         input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), audit=audit))
    # Rebind the old adversarial harness privately. Its adversarial mutations
    # now call the NEW production checker, including when Python uses -O.
    namespace = historical_audit.corruption_challenges.__globals__.copy()
    namespace['verify_case'] = lambda case: verify_trace(case['trace'], case['events'])
    challenges = FunctionType(historical_audit.corruption_challenges.__code__, namespace)(
        json.loads(files[0].read_text(encoding='utf-8')))
    example = json.loads(files[0].read_text(encoding='utf-8'))
    rejected = []
    for name in ('different_feedback', 'missing_accepted_event', 'unaccepted_event'):
        ledger = copy.deepcopy(example['events'])
        if name == 'different_feedback':
            ledger[0]['measure_result'] = 'near' if ledger[0]['measure_result'] == 'no_signal' else 'no_signal'
        elif name == 'missing_accepted_event':
            ledger.pop()
        else:
            ledger[0]['accepted'] = False
        try:
            verify_trace(example['trace'], ledger)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('Independent accepted ledger forgery passed '+name)
    summary = dict(passed=True, optimization_level=sys.flags.optimize, extraction=extraction,
        historical_cases=len(rows), optical_plans=sum(r['audit']['plans'] for r in rows), rows=rows,
        corruption_challenges=challenges, independent_ledger_challenges_rejected=rejected,
        source_sha256={name: hashlib.sha256((Q4/name).read_bytes()).hexdigest() for name in
                       ('negative_region_proof.py', 'prune_proof.py')},
        no_new_solver_runs=True, official_simulator_contacted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, optimization_level=sys.flags.optimize, equivalent_functions=extraction['equivalent_functions'],
        historical_cases=len(rows), optical_plans=summary['optical_plans'],
        forged_proof_challenges=challenges['challenges'], ledger_challenges=len(rejected)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
