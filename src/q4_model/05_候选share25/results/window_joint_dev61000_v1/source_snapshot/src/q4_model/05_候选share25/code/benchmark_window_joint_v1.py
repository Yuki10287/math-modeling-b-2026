"""Frozen local triples for production share25_prune and window depths one/two.

Only LocalArena and independent post-run auditors receive hidden source data.
The candidate exposes solve_multi(api, variant, trace, max_active_measures,
depth). This driver does not run any official client or select model budgets.
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
POPULATIONS, FIELDS = frozen.POPULATIONS, frozen.FIELDS
VARIANTS = ('baseline_share25_prune', 'window_depth1', 'window_depth2')
DEPTHS = {'window_depth1': 1, 'window_depth2': 2}
COMPARISONS = ((VARIANTS[0], VARIANTS[1]), (VARIANTS[0], VARIANTS[2]),
               (VARIANTS[1], VARIANTS[2]))
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
    return {key: resolve_logical(logical) for key, logical in dict(
        environment='src/q3_model_v2/environment.py', validation='src/q4_model/validation.py',
        cases='src/q4_model/benchmark.py', populations='src/q4_model/benchmark_cells.py').items()}


def load_support():
    sys.path.insert(0, str(Q4))
    baseline = importlib.import_module('ordered_optical_prune')
    candidate = importlib.import_module('window_joint_planner_v1')
    require(Path(baseline.__file__).resolve() == Q4/'ordered_optical_prune.py', 'wrong baseline module')
    require(Path(candidate.__file__).resolve() == HERE/'window_joint_planner_v1.py', 'wrong candidate module')
    paths = support_paths()
    env = frozen.load_module('_window_joint_local_arena', paths['environment'])
    validation = frozen.load_module('_window_joint_post_validation', paths['validation'])
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
        candidate_module=candidate, LocalArena=env.LocalArena,
        validate_run=validation.validate_run, polygon_contains=validation.polygon_contains,
        sources_for=namespace['sources_for'], verify_trace=proof.verify_trace,
        exact_certificate=cover.exact_certificate_audit, PolarCover=polar.PolarCover)


def static_local_imports(entries):
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
    production = {key: digest(resolve_logical('src/'+key)) for key in release['source_sha256']}
    require(production == release['source_sha256'], 'production differs from its frozen release')
    paths = {resolve_logical('src/'+key) for key in production}
    paths |= set(support_paths().values()) | {INDEX, RELEASE}
    paths |= static_local_imports([Path(__file__), HERE/'window_joint_planner_v1.py',
                                  HERE/'refined_cover_audit.py', Path(frozen.__file__)])
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in sorted(paths)}


def model_configuration(module):
    """Record JSON-compatible public configuration without selecting parameters."""
    values = {}
    for key, value in vars(module).items():
        if key.isupper() and not key.startswith('_'):
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                continue
            values[key] = value
    return values


def audit_planner_budget(trace, variant, module):
    """Check declared work counts; this is not a proof of rollout predictions."""
    config = model_configuration(module)
    names = ('WORLD_COUNT', 'MAX_TRANSITIONS', 'MAX_PREDICTED_API_CALLS', 'MAX_WINDOW_DECISIONS')
    require(all(type(config.get(k)) is int and config[k] > 0 for k in names),
        'candidate must publish positive integer world/work/window limits')
    predictions = [r for r in trace if r.get('phase') == 'window_rollout_prediction']
    if variant == VARIANTS[0]:
        require(not predictions, 'baseline unexpectedly ran a window planner')
    used, predicted_calls, terminals = [], [], []
    for row in predictions:
        require(row['depth'] == DEPTHS[variant], 'wrong planner depth in trace')
        budget = row['budget']
        require(budget['max_transitions'] == config['MAX_TRANSITIONS'], 'trace changed transition budget')
        counts = {k: budget[k] for k in ('used_transitions', 'predicted_measure', 'predicted_clear', 'terminal_evaluations')}
        require(all(type(v) is int and v >= 0 for v in counts.values()), 'invalid declared work counts')
        require(counts['used_transitions'] <= config['MAX_TRANSITIONS'], 'transition budget exceeded')
        calls = counts['predicted_measure']+counts['predicted_clear']
        require(calls <= config['MAX_PREDICTED_API_CALLS'], 'predicted API budget exceeded')
        used.append(counts['used_transitions'])
        predicted_calls.append(calls)
        terminals.append(counts['terminal_evaluations'])
    evaluated_windows = sum(n > 0 for n in used)
    require(evaluated_windows <= config['MAX_WINDOW_DECISIONS'], 'too many evaluated windows')
    return dict(passed=True, prediction_rows=len(predictions), evaluated_windows=evaluated_windows,
        used_transitions_total=sum(used), used_transitions_maximum=max(used, default=0),
        predicted_api_calls_total=sum(predicted_calls), predicted_api_calls_maximum=max(predicted_calls, default=0),
        terminal_evaluations_total=sum(terminals), configuration={k: config[k] for k in names},
        scope='Independent count/limit checks of declared profiles; prediction semantics require separate model audit. Same cap does not imply equal actual work.')


def decision_diagnostics(events, trace):
    common = frozen.diagnostics(events, trace)
    common = {k: v for k, v in common.items() if not k.startswith('joint_') and k != 'actual_joint_measures'}
    planner = [r for r in trace if r.get('phase', '').startswith(('window_', 'planner_'))]
    common['window_trace_phase_counts'] = dict(Counter(r['phase'] for r in planner))
    common['window_trace_fields'] = sorted({key for row in planner for key in row})
    common['planner_profiles'] = [dict(phase=r['phase'], trace_index=i,
        **{k: v for k, v in r.items() if 'profile' in k or 'budget' in k or 'evaluation' in k})
        for i, r in enumerate(trace) if r.get('phase', '').startswith(('window_', 'planner_')) and
        any('profile' in k or 'budget' in k or 'evaluation' in k for k in r)]
    common['decision_evidence_scope'] = 'All planner alternatives, scores, profiles and budgets remain in the original complete trace.'
    return common


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
            result = support['candidate'](api, variant='share25', trace=trace,
                max_active_measures=6, depth=DEPTHS[variant])
        require(time.perf_counter()-started <= frozen.WALL_LIMIT_S, 'solver wall-clock limit exceeded')
        require(time.process_time()-cpu_started <= frozen.WALL_LIMIT_S, 'solver process-CPU limit exceeded')
        require(len(arena.events) <= frozen.ACTION_LIMIT, 'solver action limit exceeded')
    except (Exception, KeyboardInterrupt) as exc:
        error = frozen.exception_record(exc)
    cpu_runtime, runtime = time.process_time()-cpu_started, time.perf_counter()-started
    audit_start = time.perf_counter()
    try:
        require(error is None, 'solver failed; preserve partial evidence')
        audit = frozen.post_audit(support, sources, arena, trace, result)
        audit['planner_declared_budget'] = audit_planner_budget(trace, variant, support['candidate_module'])
    except (Exception, KeyboardInterrupt) as exc:
        audit = dict(passed=False, error=frozen.exception_record(exc))
    audit_wall = time.perf_counter()-audit_start
    try:
        diagnostics = decision_diagnostics(arena.events, trace)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostics = dict(passed=False, partial_trace=True, error=frozen.exception_record(exc))
    summary = dict(seed=seed, population=population, field=field, variant=variant,
        depth=DEPTHS.get(variant), runtime_s=runtime, process_cpu_s=cpu_runtime,
        independent_audit_wall_s=audit_wall, error=error, audit=audit,
        decision_diagnostics=diagnostics, trace_phase_counts=dict(Counter(
            str(r.get('phase', '<missing>')) if isinstance(r, dict) else '<nonobject>' for r in trace)),
        **arena.evaluation())
    return dict(summary=summary, sources=sources, result=result, events=arena.events, trace=trace)


def compare_cases(cases, seed, population, field, encoded_scene):
    comparisons = []
    for before, after in COMPARISONS:
        if before not in cases or after not in cases:
            continue
        left, right = cases[before], cases[after]
        a, b = left['summary'], right['summary']
        comparisons.append(dict(seed=seed, population=population, field=field,
            before_variant=before, after_variant=after,
            scene_sha256=hashlib.sha256(encoded_scene.encode()).hexdigest(),
            baseline_s=a['total_s'], candidate_s=b['total_s'], saved_s=a['total_s']-b['total_s'],
            reduction_percent=100*(a['total_s']-b['total_s'])/a['total_s'],
            baseline_passed=a['audit']['passed'], candidate_passed=b['audit']['passed'],
            all_cleared=a['all_cleared'] and b['all_cleared'], source_count=a['source_count'],
            same_source_layout=left['sources'] == right['sources'],
            shared_point_feedback=frozen.shared_point_feedback(left, right)))
    return comparisons


def aggregate(rows, comparisons):
    groups, differences = {}, {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        if not group:
            groups[variant] = dict(runs=0)
            continue
        values = [r['total_s'] for r in group]
        groups[variant] = dict(runs=len(group), all_cleared=sum(r['all_cleared'] for r in group),
            passes=sum(r['audit']['passed'] for r in group),
            source_condition_instances=sum(r['source_count'] for r in group),
            cleared=sum(r['cleared'] for r in group), mean_total_s=float(np.mean(values)),
            maximum_total_s=max(values), p90_total_s=float(np.quantile(values, .9)),
            pooled_per_source_s=sum(values)/sum(r['source_count'] for r in group),
            mean_solver_wall_s=float(np.mean([r['runtime_s'] for r in group])),
            maximum_solver_wall_s=max(r['runtime_s'] for r in group),
            mean_process_cpu_s=float(np.mean([r['process_cpu_s'] for r in group])),
            maximum_process_cpu_s=max(r['process_cpu_s'] for r in group))
    for before, after in COMPARISONS:
        pairs = [p for p in comparisons if p['before_variant'] == before and p['after_variant'] == after]
        if not pairs:
            continue
        mean_before = float(np.mean([p['baseline_s'] for p in pairs]))
        mean_after = float(np.mean([p['candidate_s'] for p in pairs]))
        differences[before+'__vs__'+after] = dict(pairs=len(pairs), mean_before_s=mean_before,
            mean_after_s=mean_after, mean_saved_s=mean_before-mean_after,
            mean_reduction_percent=100*(mean_before-mean_after)/mean_before,
            faster=sum(p['saved_s'] > 1e-7 for p in pairs),
            slower=sum(p['saved_s'] < -1e-7 for p in pairs),
            ties=sum(abs(p['saved_s']) <= 1e-7 for p in pairs),
            worst_pair=min(pairs, key=lambda p: p['reduction_percent']))
    return dict(groups=groups, comparisons=differences)


def make_summary(rows, comparisons, stable, planned_conditions, completed_conditions, fields, failure):
    passed = (failure is None and stable and completed_conditions == planned_conditions
        and len(rows) == planned_conditions*len(VARIANTS)
        and all(r['audit']['passed'] and r['error'] is None and r['all_cleared'] for r in rows))
    return dict(passed=passed, source_snapshot_stable=stable, rows=rows, comparisons=comparisons,
        overall=aggregate(rows, comparisons),
        by_field={f: aggregate([r for r in rows if r['field'] == f],
            [p for p in comparisons if p['field'] == f]) for f in fields},
        planned_conditions=planned_conditions, completed_conditions=completed_conditions,
        stopped_on_failure=failure, official_contacted=False, performance_comparison_valid=passed,
        warning='Repeated fields share each layout. All three variants share every condition. Failures stop this batch, preserve partial evidence and invalidate performance claims.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds-base', type=int, default=61000)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--fields', default='hashed', help='Comma-separated constant,hashed,extreme.')
    parser.add_argument('--inventory-only', action='store_true', help='Read seed history only; do not import or run candidate, or write results.')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'frozen independent validators require assertions enabled')
    require(args.seeds_base >= 61000 and 1 <= args.layouts <= 99, 'invalid independent seed allocation')
    fields = tuple(args.fields.split(','))
    require(fields and len(set(fields)) == len(fields) and set(fields) <= set(FIELDS), 'invalid field selection')
    out, results = args.out.resolve(), (HERE.parent/'results').resolve()
    require(out.is_relative_to(results) and out != results and out.name.startswith('window_joint_'),
        'use an independent window_joint_ results directory')
    require(not out.exists(), 'refuse to overwrite any existing result')
    seeds = [args.seeds_base+100*i+j for i in range(len(POPULATIONS)) for j in range(args.layouts)]
    inventory = frozen.seed_inventory(set(seeds))
    require(not inventory['collisions'], 'prior seed collision: '+json.dumps(inventory['collisions']))
    if args.inventory_only:
        print(json.dumps(dict(seeds=seeds, inventory=inventory, candidate_imported=False,
            solver_runs=0, files_written=0), ensure_ascii=False, indent=2))
        return 0
    support = load_support()
    snapshot = source_hashes()
    configuration = model_configuration(support['candidate_module'])
    out.mkdir(parents=True, exist_ok=False)
    for relative, expected in snapshot.items():
        raw = (PROJECT/relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'source changed before snapshot archive')
        archived = out/'source_snapshot'/relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        with archived.open('xb') as stream:
            stream.write(raw)
    planned_conditions = len(seeds)*len(fields)
    write(out/'manifest.json', dict(code_hashes=snapshot, source_archive='source_snapshot',
        variants=VARIANTS, depths=DEPTHS, seeds=seeds, seeds_base=args.seeds_base,
        populations=POPULATIONS, layouts_per_population=args.layouts, fields=fields,
        independent_layouts=len(seeds), paired_conditions=planned_conditions,
        runs_per_condition=len(VARIANTS), planned_runs=planned_conditions*len(VARIANTS),
        max_active_measures=6, candidate_module_configuration=configuration,
        transition_budget_scope='Candidate-owned frozen configuration; root and second-level transition evaluations must share the same total cap. See complete decision profiles.',
        local_only=True, official_contacted=False, frozen_before_runs=True,
        parameter_tuning_during_run=False, prior_seed_inventory=inventory,
        baseline='Direct current production ordered_optical_prune.solve_multi(variant=share25).',
        change_scope='Outer window choice at depth 1 or 2; actual optical/prune proof and execution remain independently audited.',
        reused_audit_helpers='Unmodified benchmark_station_joint_view_v1 pure API and post-run checks; its candidate, load_support, main and scoring are not used.',
        current_physical_support={k: p.relative_to(PROJECT).as_posix() for k, p in support_paths().items()},
        public_interface=['position copy', 'channel', 'measure', 'clear'],
        hidden_truth_scope='Environment and post-run audit only.',
        error_fields=dict(constant='Fixed +0.99 degrees.', hashed='Fixed coordinate/seed/channel BLAKE2b in [-0.99,+0.99).',
            extreme='Fixed spatial sign field +/-0.99 degrees.'),
        field_distribution_is_official=False, population_distribution_is_official=False,
        solver_wall_timing_excludes='Post-run proof audits and serialization.',
        process_cpu_clock='time.process_time()', action_limit=frozen.ACTION_LIMIT,
        solver_wall_limit_s=frozen.WALL_LIMIT_S, solver_cpu_limit_s=frozen.WALL_LIMIT_S,
        limit_enforcement='Inherited action/wall check before every public action and final CPU/wall check; a non-returning pure computation is not preempted by this in-process driver.',
        failure_policy='Save failed case and partial trace; stop batch without retrying, skipping layouts or relaxing checks.',
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    rows, comparisons, stable_runs = [], [], []
    completed_conditions, failure = 0, None
    try:
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
                        require(json.dumps(sources, sort_keys=True, allow_nan=False) == encoded, 'scene mutated')
                        require(stable, 'source changed during frozen run')
                        require(row['error'] is None and row['audit']['passed'] and row['all_cleared'],
                            'failed case saved; stopping fixed batch')
                    comparisons.extend(compare_cases(cases, seed, population, field, encoded))
                    completed_conditions += 1
    except (Exception, KeyboardInterrupt) as exc:
        failure = frozen.exception_record(exc)
        write(out/'failure.json', failure)
    try:
        stable = all(stable_runs) and source_hashes() == snapshot
    except Exception as exc:
        stable = False
        if failure is None:
            failure = frozen.exception_record(exc)
            write(out/'failure.json', failure)
    summary = make_summary(rows, comparisons, stable, planned_conditions, completed_conditions, fields, failure)
    summary['independent_layouts'] = len(seeds)
    write(out/'rows.json', rows)
    write(out/'summary.json', summary)
    print(json.dumps(dict(passed=summary['passed'], source_snapshot_stable=stable,
        stopped_on_failure=failure, overall=summary['overall']), ensure_ascii=False), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
