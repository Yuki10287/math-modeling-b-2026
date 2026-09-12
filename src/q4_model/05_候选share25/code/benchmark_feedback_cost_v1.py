"""Frozen local pairs for feedback-branch cost prediction only.

Baseline is the production ordered optical prune policy, not the failed joint
station experiment. Its old harness contributes only unchanged API and audit
helpers. Current physical dependencies resolve through the study index.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import benchmark_station_joint_view_v1 as frozen

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
INDEX = PROJECT/'src/studies.json'
RELEASE = HERE.parent/'results/share25_prune_client_release.json'
POPULATIONS = frozen.POPULATIONS
FIELDS = frozen.FIELDS
VARIANTS = ('baseline_share25_prune', 'feedback_cost')
require, write, digest = frozen.require, frozen.write, frozen.digest


def resolve_logical(logical):
    direct = PROJECT/logical
    if direct.is_file():
        return direct.resolve()
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    matches = [PROJECT/r['path'] for r in index['files'] if r['logical'] == logical]
    require(len(matches) == 1 and matches[0].is_file(), 'missing current physical source '+logical)
    return matches[0].resolve()


def support_paths():
    return {key:resolve_logical(logical) for key, logical in dict(
        environment='src/q3_model_v2/environment.py', validation='src/q4_model/validation.py',
        cases='src/q4_model/benchmark.py', populations='src/q4_model/benchmark_cells.py').items()}


def load_support():
    sys.path.insert(0, str(Q4))
    baseline = importlib.import_module('ordered_optical_prune')
    candidate = importlib.import_module('feedback_cost_experiment')
    require(Path(baseline.__file__).resolve() == Q4/'ordered_optical_prune.py', 'wrong baseline module')
    require(Path(candidate.__file__).resolve() == HERE/'feedback_cost_experiment.py', 'wrong candidate module')
    paths = support_paths()
    env = frozen.load_module('_feedback_cost_local_arena', paths['environment'])
    validation = frozen.load_module('_feedback_cost_post_validation', paths['validation'])
    namespace = dict(np=np, math=math)
    for path, names in ((paths['cases'], {'make_case'}), (paths['populations'], {'sources_for'})):
        nodes = [n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == names, 'missing frozen source generator')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    proof = importlib.import_module('prune_proof')
    cover = importlib.import_module('refined_cover_audit')
    polar = importlib.import_module('polar_cover')
    return dict(baseline=baseline.solve_multi, candidate=candidate.solve_multi,
        LocalArena=env.LocalArena, validate_run=validation.validate_run,
        polygon_contains=validation.polygon_contains, sources_for=namespace['sources_for'],
        verify_trace=proof.verify_trace, exact_certificate=cover.exact_certificate_audit,
        PolarCover=polar.PolarCover)


def static_local_imports(entries):
    """Fingerprint direct local module imports, including the reception helper."""
    found, pending = set(), list(entries)
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        require(path.is_file(), 'local dependency missing '+str(path))
        found.add(path)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                     else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                for directory in (path.parent, HERE, Q4):
                    local = directory/(name+'.py')
                    if local.is_file():
                        pending.append(local)
                        break
    return found


def source_hashes():
    release = json.loads(RELEASE.read_text(encoding='utf-8'))
    production = {key:digest(resolve_logical('src/'+key)) for key in release['source_sha256']}
    require(production == release['source_sha256'], 'production differs from its frozen release')
    paths = {resolve_logical('src/'+key) for key in production}
    paths |= set(support_paths().values()) | {INDEX, RELEASE}
    paths |= static_local_imports([Path(__file__), HERE/'feedback_cost_experiment.py',
                                  HERE/'refined_cover_audit.py', Path(frozen.__file__)])
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}


def decision_diagnostics(events, trace):
    base = frozen.diagnostics(events, trace)
    base = {k:v for k, v in base.items() if not k.startswith('joint_') and k != 'actual_joint_measures'}
    predictions = [r for r in trace if r['phase'] == 'feedback_cost_prediction']
    base['feedback_cost_predictions'] = len(predictions)
    base['prediction_fields'] = sorted({key for row in predictions for key in row})
    base['decision_evidence_scope'] = 'Full original/new action scores, support and risk terms remain in the complete trace.'
    return base


def run_case(support, sources, seed, population, field, variant):
    arena = support['LocalArena'](copy.deepcopy(sources), seed, field)
    api = frozen.frozen_api(arena)
    for name in ('sources', '_sources', '_arena', 'events', 'time_s', 'evaluation', '__dict__'):
        require(not hasattr(api, name), 'nonpublic environment attribute exposed')
    trace, result, error = [], None, None
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        if variant == VARIANTS[0]:
            result = support['baseline'](api, variant='share25', trace=trace)
        else:
            result = support['candidate'](api, variant='share25', trace=trace, max_active_measures=6)
    except Exception as exc:
        error = frozen.exception_record(exc)
    cpu_runtime, runtime = time.process_time()-cpu_started, time.perf_counter()-started
    audit_start = time.perf_counter()
    try:
        require(error is None, 'solver failed; preserve partial evidence')
        audit = frozen.post_audit(support, sources, arena, trace, result)
    except Exception as exc:
        audit = dict(passed=False, error=frozen.exception_record(exc))
    audit_wall = time.perf_counter()-audit_start
    try:
        diagnostics = decision_diagnostics(arena.events, trace)
    except Exception as exc:
        diagnostics = dict(passed=False, partial_trace=True, error=frozen.exception_record(exc))
    summary = dict(seed=seed, population=population, field=field, variant=variant,
        runtime_s=runtime, process_cpu_s=cpu_runtime, independent_audit_wall_s=audit_wall,
        error=error, audit=audit, decision_diagnostics=diagnostics,
        trace_phase_counts=dict(Counter(r['phase'] for r in trace)), **arena.evaluation())
    return dict(summary=summary, sources=sources, result=result, events=arena.events, trace=trace)


def aggregate(rows, pairs):
    groups = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        values = [r['total_s'] for r in group]
        groups[variant] = dict(runs=len(group), all_cleared=sum(r['all_cleared'] for r in group),
            passes=sum(r['audit']['passed'] for r in group),
            source_condition_instances=sum(r['source_count'] for r in group),
            cleared=sum(r['cleared'] for r in group), mean_total_s=float(np.mean(values)),
            maximum_total_s=max(values), p90_total_s=float(np.quantile(values, .9)),
            pooled_per_source_s=sum(values)/sum(r['source_count'] for r in group),
            mean_solver_wall_s=float(np.mean([r['runtime_s'] for r in group])),
            maximum_solver_wall_s=max(r['runtime_s'] for r in group),
            mean_process_cpu_s=float(np.mean([r['process_cpu_s'] for r in group])))
    before, after = [groups[v]['mean_total_s'] for v in VARIANTS]
    return dict(groups=groups, mean_reduction_percent=100*(before-after)/before,
        faster=sum(p['saved_s'] > 1e-7 for p in pairs), slower=sum(p['saved_s'] < -1e-7 for p in pairs),
        ties=sum(abs(p['saved_s']) <= 1e-7 for p in pairs),
        worst_pair=min(pairs, key=lambda p:p['reduction_percent']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds-base', type=int, default=57000)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--fields', default='hashed', help='Comma-separated constant,hashed,extreme.')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'frozen independent validators require assertions enabled')
    require(args.seeds_base >= 57000 and 1 <= args.layouts <= 99, 'invalid independent seed allocation')
    fields = tuple(args.fields.split(','))
    require(fields and len(set(fields)) == len(fields) and set(fields) <= set(FIELDS), 'invalid field selection')
    out, root_results = args.out.resolve(), (HERE.parent/'results').resolve()
    require(out.is_relative_to(root_results) and out != root_results and out.name.startswith('feedback_cost_'),
            'use an independent feedback_cost_ results directory')
    require(not out.exists(), 'refuse to overwrite any existing result')
    seeds = [args.seeds_base+100*i+j for i in range(5) for j in range(args.layouts)]
    inventory = frozen.seed_inventory(set(seeds))
    require(not inventory['collisions'], 'prior seed collision: '+json.dumps(inventory['collisions']))
    support = load_support()
    snapshot = source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    for relative, expected in snapshot.items():
        raw = (PROJECT/relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'source changed before snapshot archive')
        archived = out/'source_snapshot'/relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        with archived.open('xb') as stream:
            stream.write(raw)
    write(out/'manifest.json', dict(code_hashes=snapshot, source_archive='source_snapshot',
        variants=VARIANTS, seeds=seeds, seeds_base=args.seeds_base, populations=POPULATIONS,
        layouts_per_population=args.layouts, fields=fields, independent_layouts=len(seeds),
        paired_conditions=len(seeds)*len(fields), max_active_measures=6,
        local_only=True, official_contacted=False, frozen_before_runs=True,
        parameter_tuning_during_run=False, prior_seed_inventory=inventory,
        baseline='Direct current production ordered_optical_prune.solve_multi(variant=share25).',
        change_scope='Only local service feedback-branch cost prediction; same original candidate points, fixed stations, route, sharing and ordered optical execution.',
        reused_audit_helpers='Unmodified benchmark_station_joint_view_v1 pure API and post-run checks; its candidate, load_support, main and scoring are not used.',
        current_physical_support={k:p.relative_to(PROJECT).as_posix() for k,p in support_paths().items()},
        public_interface=['position copy','channel','measure','clear'], hidden_truth_scope='Environment and post-run audit only.',
        error_fields=dict(constant='Fixed +0.99 degrees.', hashed='Fixed coordinate/seed/channel BLAKE2b in [-0.99,+0.99).',
            extreme='Fixed spatial sign field +/-0.99 degrees.'),
        field_distribution_is_official=False, population_distribution_is_official=False,
        solver_wall_timing_excludes='Post-run proof audits and serialization.', process_cpu_clock='time.process_time()',
        action_limit=frozen.ACTION_LIMIT, solver_wall_limit_s=frozen.WALL_LIMIT_S,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    rows, pairs, stable_runs = [], [], []
    for i, population in enumerate(POPULATIONS):
        for layout in range(args.layouts):
            seed = args.seeds_base+100*i+layout
            sources = support['sources_for'](seed, population)
            encoded = json.dumps(sources, sort_keys=True, allow_nan=False)
            for field in fields:
                cases = {}
                for variant in VARIANTS:
                    require(source_hashes() == snapshot, 'source changed during frozen batch')
                    case = run_case(support, sources, seed, population, field, variant)
                    require(json.dumps(sources, sort_keys=True, allow_nan=False) == encoded, 'scene mutated')
                    stable = source_hashes() == snapshot
                    stable_runs.append(stable)
                    row = case['summary']
                    row['source_snapshot_stable'] = stable
                    path = out/f'{seed}-{population}-{field}-{variant}.json'
                    row['artifact'] = path.relative_to(PROJECT).as_posix()
                    write(path, case)
                    rows.append(row)
                    cases[variant] = case
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.6f}s; '
                        f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                        f'wall={row["runtime_s"]:.3f}s; cpu={row["process_cpu_s"]:.3f}s', flush=True)
                    if row['error'] or not row['audit']['passed']:
                        print(json.dumps(dict(error=row['error'], audit=row['audit']), ensure_ascii=False), flush=True)
                baseline, candidate = (cases[v]['summary'] for v in VARIANTS)
                a, b = baseline['total_s'], candidate['total_s']
                pairs.append(dict(seed=seed, population=population, field=field,
                    scene_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                    baseline_s=a, candidate_s=b, saved_s=a-b, reduction_percent=100*(a-b)/a,
                    baseline_passed=baseline['audit']['passed'], candidate_passed=candidate['audit']['passed'],
                    all_cleared=baseline['all_cleared'] and candidate['all_cleared'],
                    source_count=baseline['source_count'], same_source_layout=cases[VARIANTS[0]]['sources'] == cases[VARIANTS[1]]['sources'],
                    shared_point_feedback=frozen.shared_point_feedback(cases[VARIANTS[0]], cases[VARIANTS[1]])))
    stable = all(stable_runs) and source_hashes() == snapshot
    passed = stable and all(r['audit']['passed'] and r['error'] is None and r['all_cleared'] for r in rows)
    summary = dict(passed=passed, source_snapshot_stable=stable, rows=rows, pairs=pairs,
        overall=aggregate(rows, pairs),
        by_field={f:aggregate([r for r in rows if r['field']==f], [p for p in pairs if p['field']==f]) for f in fields},
        independent_layouts=len(seeds), paired_conditions=len(pairs), official_contacted=False,
        performance_comparison_valid=passed,
        warning='Repeated fields share each layout. Failures remain recorded and invalidate performance claims. Saved source snapshots belong to this feedback-cost experiment only.')
    write(out/'rows.json', rows)
    write(out/'summary.json', summary)
    print(json.dumps(dict(passed=passed, source_snapshot_stable=stable, overall=summary['overall']), ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
