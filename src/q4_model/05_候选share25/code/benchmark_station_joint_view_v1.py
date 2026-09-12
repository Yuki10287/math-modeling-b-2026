"""New-layout local pairs: production share25_prune versus joint station views.

The baseline calls the production solver directly. Only the environment and
post-run auditors see hidden sources. Each candidate receives the same four
public attributes as the baseline, with fixed location-dependent error fields.
All artifacts are new files; failed runs retain their partial events and trace.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
POPULATIONS = ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')
FIELDS = ('constant', 'hashed', 'extreme')
VARIANTS = ('baseline_share25_prune', 'station_joint_view')
ENVIRONMENT = Q4.parent/'q3_model_v2/00_主方案_lean/code/environment.py'
VALIDATION = Q4/'00_公共几何与定位/code/validation.py'
CASE_GENERATOR = Q4/'01_初版31站/code/benchmark.py'
POPULATION_GENERATOR = Q4/'04_22站覆盖实验/code/benchmark_cells.py'
RELEASE = HERE.parent/'results/share25_prune_client_release.json'
ACTION_LIMIT = 10000
WALL_LIMIT_S = 1200


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_support():
    """Load only the historical environment, validation and case functions."""
    sys.path.insert(0, str(Q4))
    baseline = importlib.import_module('ordered_optical_prune')
    candidate = importlib.import_module('station_joint_view_experiment')
    require(Path(baseline.__file__).resolve() == Q4/'ordered_optical_prune.py',
            'baseline must be the current production module')
    require(Path(candidate.__file__).resolve() == HERE/'station_joint_view_experiment.py',
            'candidate must be the independent new experiment module')
    env = load_module('_joint_station_local_arena', ENVIRONMENT)
    validation = load_module('_joint_station_post_validation', VALIDATION)
    namespace = dict(np=np, math=math)
    for path, names in ((CASE_GENERATOR, {'make_case'}),
                        (POPULATION_GENERATOR, {'sources_for'})):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({node.name for node in nodes} == names, 'historical case generator unavailable')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    proof = importlib.import_module('prune_proof')
    exact_cover = importlib.import_module('refined_cover_audit')
    polar = importlib.import_module('polar_cover')
    return dict(baseline=baseline.solve_multi, candidate=candidate.solve_multi,
                LocalArena=env.LocalArena, validate_run=validation.validate_run,
                polygon_contains=validation.polygon_contains,
                sources_for=namespace['sources_for'], verify_trace=proof.verify_trace,
                exact_certificate=exact_cover.exact_certificate_audit, PolarCover=polar.PolarCover)


def source_hashes():
    release = json.loads(RELEASE.read_text(encoding='utf-8'))
    production = {key:digest(PROJECT/'src'/key) for key in release['source_sha256']}
    require(production == release['source_sha256'], 'production differs from the frozen client release')
    paths = [PROJECT/'src'/name for name in production]
    paths += [Path(__file__), HERE/'station_joint_view_experiment.py',
              ENVIRONMENT, VALIDATION, CASE_GENERATOR, POPULATION_GENERATOR,
              HERE/'refined_cover_audit.py', RELEASE]
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in paths}


def seed_inventory(wanted):
    """Check saved case filenames and recorded manifest seed declarations."""
    hits = defaultdict(set)
    checked_manifests = 0
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('seed', 'start_seed', 'seeds_base') and type(child) is int and child in wanted:
                    hits[child].add(path)
                if key == 'seeds' and isinstance(child, list):
                    for seed in child:
                        if type(seed) is int and seed in wanted:
                            hits[seed].add(path)
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
    for path in (PROJECT/'src').rglob('*.json'):
        if 'results' not in path.parts:
            continue
        relative = path.relative_to(PROJECT).as_posix()
        match = re.match(r'^(\d+)-', path.name)
        if match and int(match.group(1)) in wanted:
            hits[int(match.group(1))].add(relative)
        if path.name == 'manifest.json':
            checked_manifests += 1
            try:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError):
                continue
            visit(value, relative)
    return dict(checked_manifests=checked_manifests,
                collisions={str(seed):sorted(paths) for seed, paths in sorted(hits.items())},
                checked_sources='Existing result filenames and explicit seed fields in all readable result manifests.')


def frozen_api(arena):
    """No truth, time, action counter or evaluation accessor reaches the solver."""
    started = time.perf_counter()
    def budget():
        if len(arena.events) >= ACTION_LIMIT or time.perf_counter()-started > WALL_LIMIT_S:
            raise RuntimeError('local experiment action or wall-clock limit exceeded')
    class FrozenAPI:
        __slots__ = ()
        def __getattribute__(self, name):
            if name not in ('position', 'channel', 'measure', 'clear'):
                raise AttributeError('Only position/channel/measure/clear are public')
            return object.__getattribute__(self, name)
        @property
        def position(self):
            return arena.position.copy()
        @property
        def channel(self):
            return arena.channel
        def measure(self, q, channel):
            budget()
            return arena.measure(q, channel)
        def clear(self, q, channel):
            budget()
            return arena.clear(q, channel)
    return FrozenAPI()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__+' is not JSON serializable')


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
        stream.write('\n')


def exception_record(exc):
    return dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())


def independent_repeat_check(events):
    removed, observed, repeated = set(), {}, 0
    for event in events:
        if event['action'] == 'clear':
            if event['clear_result'] == 'success':
                removed.add(event['channel'])
            continue
        key = (event['channel'], tuple(event['position']), event['channel'] in removed)
        reading = (event['measure_result'], event.get('svd_deg'))
        if key in observed:
            require(observed[key] == reading, 'same source and point gave inconsistent fixed-field feedback')
            repeated += 1
        observed[key] = reading
    return dict(passed=True, exact_repeated_points=repeated)


def post_audit(support, sources, arena, trace, result):
    mesh = support['PolarCover']()
    audit = support['validate_run'](sources, arena.events, trace, result, mesh.stations, mesh.indices)
    audit['complete_optical_and_prune_proofs'] = support['verify_trace'](trace, arena.events)
    audit['exact_continuous_absence'] = support['exact_certificate'](result, arena.events)
    audit['fixed_feedback_consistency'] = independent_repeat_check(arena.events)
    truth = {source['channel']:source['position'] for source in sources}
    contractions = 0
    for row in trace:
        if row['phase'] == 'ordered_optical_prune':
            require(support['polygon_contains'](row['polygon'], truth[row['channel']]),
                    'contracted optical region excludes hidden source')
            contractions += int(row['info']['applied'])
    audit['postrun_contracted_true_source_retention'] = dict(passed=True, applied_contractions=contractions)
    require(arena.evaluation()['all_cleared'], 'not all environment sources removed')
    return audit


def diagnostics(events, trace):
    predictions = [r for r in trace if r['phase'] == 'station_joint_prediction']
    markers = [r for r in trace if r['phase'] == 'actual_action']
    require(len(events) == len(markers), 'diagnostics require all actual trace markers')
    seen, scan_positions = set(), set()
    counts = Counter()
    by_reason = defaultdict(Counter)
    for event, marker in zip(events, markers):
        require(marker['event'] == sum(counts[k] for k in ('measure', 'clear'))+1,
                'invalid diagnostic event order')
        counts[event['action']] += 1
        reason = marker['reason']
        by_reason[reason]['actions'] += 1
        if event['action'] == 'measure':
            positive = event['measure_result'] != 'no_signal'
            by_reason[reason]['positive' if positive else 'negative'] += 1
            if reason == 'scan':
                scan_positions.add(tuple(event['position']))
                counts['scan_measures'] += 1
                counts['unknown_scan_measures'] += int(event['channel'] not in seen)
            if reason == 'joint_source_measure':
                counts['joint_measures'] += 1
                counts['joint_positive' if positive else 'joint_negative'] += 1
            if positive:
                seen.add(event['channel'])
    return dict(joint_predictions=len(predictions),
        joint_selected=sum(p.get('selected') is not None for p in predictions),
        actual_joint_measures=counts['joint_measures'], joint_positive=counts['joint_positive'],
        joint_negative=counts['joint_negative'], actual_scan_stations=len(scan_positions),
        actual_scan_measures=counts['scan_measures'], unknown_scan_measures=counts['unknown_scan_measures'],
        by_reason={k:dict(v) for k, v in sorted(by_reason.items())})


def shared_point_feedback(case_a, case_b):
    def observations(events):
        removed, values = set(), {}
        for event in events:
            if event['action'] == 'clear':
                if event['clear_result'] == 'success':
                    removed.add(event['channel'])
            else:
                key = (event['channel'], tuple(event['position']), event['channel'] in removed)
                values[key] = (event['measure_result'], event.get('svd_deg'))
        return values
    a, b = observations(case_a['events']), observations(case_b['events'])
    common = set(a) & set(b)
    require(all(a[k] == b[k] for k in common), 'paired runs have different feedback at same point and state')
    return dict(passed=True, shared_measurement_keys=len(common))


def run_case(support, sources, seed, population, field, variant):
    arena = support['LocalArena'](copy.deepcopy(sources), seed, field)
    trace, result, error = [], None, None
    api = frozen_api(arena)
    for name in ('_sources', 'sources', '_arena', 'time_s', 'events', 'evaluation', '__dict__'):
        require(not hasattr(api, name), 'hidden environment accessor is visible to the solver')
    before = time.perf_counter()
    try:
        if variant == VARIANTS[0]:
            result = support['baseline'](api, variant='share25', trace=trace)
        else:
            result = support['candidate'](api, variant='share25', trace=trace, max_active_measures=6)
    except Exception as exc:
        error = exception_record(exc)
    runtime = time.perf_counter()-before
    audit_start = time.perf_counter()
    try:
        require(error is None, 'solver failed; partial trajectory retained')
        audit = post_audit(support, sources, arena, trace, result)
    except Exception as exc:
        audit = dict(passed=False, error=exception_record(exc))
    audit_runtime = time.perf_counter()-audit_start
    try:
        diagnostic = diagnostics(arena.events, trace)
    except Exception as exc:
        diagnostic = dict(passed=False, partial_trace=True, error=exception_record(exc))
    row = dict(seed=seed, population=population, field=field, variant=variant,
               runtime_s=runtime, independent_audit_wall_s=audit_runtime,
               error=error, audit=audit, trace_phase_counts=dict(Counter(r['phase'] for r in trace)),
               joint_diagnostics=diagnostic,
               **arena.evaluation())
    return dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace)


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
            maximum_solver_wall_s=max(r['runtime_s'] for r in group))
    a, b = [groups[v]['mean_total_s'] for v in VARIANTS]
    return dict(groups=groups, mean_reduction_percent=100*(a-b)/a,
                faster=sum(p['saved_s'] > 1e-7 for p in pairs),
                slower=sum(p['saved_s'] < -1e-7 for p in pairs),
                ties=sum(abs(p['saved_s']) <= 1e-7 for p in pairs),
                worst_pair=min(pairs, key=lambda p:p['reduction_percent']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds-base', type=int, default=54000)
    parser.add_argument('--layouts', type=int, default=1, help='Independent layouts per population (1 to 99).')
    parser.add_argument('--fields', default='hashed', help='Comma-separated constant,hashed,extreme; each field reuses each layout.')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'frozen independent validators require assertions enabled')
    require(args.seeds_base >= 54000 and 1 <= args.layouts <= 99, 'invalid new seed allocation')
    fields = tuple(args.fields.split(','))
    require(fields and len(fields) == len(set(fields)) and set(fields) <= set(FIELDS), 'invalid or duplicate field')
    out = args.out.resolve()
    results_root = (HERE.parent/'results').resolve()
    require(out.is_relative_to(results_root) and out != results_root, 'output must be a new directory inside this study results')
    require(not out.exists(), 'refuse to overwrite an existing experiment directory')
    seeds = [args.seeds_base+100*i+j for i in range(len(POPULATIONS)) for j in range(args.layouts)]
    inventory = seed_inventory(set(seeds))
    require(not inventory['collisions'], 'seed already used by earlier saved experiment: '+json.dumps(inventory['collisions']))
    support = load_support()
    snapshot = source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_hashes=snapshot, populations=POPULATIONS, seeds=seeds,
        seeds_base=args.seeds_base, layouts_per_population=args.layouts, fields=fields,
        variants=VARIANTS, model_variant='share25', candidate_max_active_measures=6,
        source_seed_rule='seeds_base + 100 * population_index + layout_index',
        independent_layouts=len(seeds), paired_conditions=len(seeds)*len(fields),
        local_only=True, official_contacted=False, frozen_before_runs=True, parameter_tuning_during_run=False,
        baseline='Direct call of production ordered_optical_prune.solve_multi, no AST extraction or research baseline.',
        error_fields=dict(constant='Fixed +0.99 degrees before two-decimal bearing rounding.',
            hashed='Fixed BLAKE2b(seed,channel,x and y to 8 decimals), scaled to [-0.99,+0.99).',
            extreme='Fixed spatial sign field +/-0.99 degrees; phase uses seed and channel.'),
        population_distribution_is_official=False, field_distribution_is_official=False,
        public_interface=['position copy', 'channel', 'measure', 'clear'],
        hidden_truth_scope='Local environment and post-run audit only; never passed to solver or used in choices.',
        solver_wall_timing_excludes='Post-run audits and serialization.', action_limit=ACTION_LIMIT,
        solver_wall_limit_s=WALL_LIMIT_S, prior_seed_inventory=inventory,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    write(out/'manifest.json', manifest)
    rows, pairs, source_stability = [], [], []
    for i, population in enumerate(POPULATIONS):
        for layout in range(args.layouts):
            seed = args.seeds_base+100*i+layout
            sources = support['sources_for'](seed, population)
            serialized_scene = json.dumps(sources, sort_keys=True, allow_nan=False)
            for field in fields:
                cases = {}
                for variant in VARIANTS:
                    require(source_hashes() == snapshot, 'source changed before completing frozen batch')
                    case = run_case(support, sources, seed, population, field, variant)
                    require(json.dumps(sources, sort_keys=True, allow_nan=False) == serialized_scene, 'scene was modified by a run')
                    stable = source_hashes() == snapshot
                    source_stability.append(stable)
                    case['summary']['source_snapshot_stable'] = stable
                    path = out/f'{seed}-{population}-{field}-{variant}.json'
                    write(path, case)
                    row = case['summary']
                    row['artifact'] = path.relative_to(PROJECT).as_posix()
                    rows.append(row)
                    cases[variant] = case
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.6f}s; '
                          f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                          f'wall={row["runtime_s"]:.3f}s', flush=True)
                    if row['error'] or not row['audit']['passed']:
                        print(json.dumps(dict(error=row['error'], audit=row['audit']), ensure_ascii=False), flush=True)
                baseline, candidate = (cases[v]['summary'] for v in VARIANTS)
                a, b = baseline['total_s'], candidate['total_s']
                pairs.append(dict(seed=seed, population=population, field=field,
                    scene_sha256=hashlib.sha256(serialized_scene.encode()).hexdigest(),
                    baseline_s=a, candidate_s=b, saved_s=a-b, reduction_percent=100*(a-b)/a,
                    baseline_passed=baseline['audit']['passed'], candidate_passed=candidate['audit']['passed'],
                    all_cleared=baseline['all_cleared'] and candidate['all_cleared'],
                    baseline_source_count=baseline['source_count'], candidate_source_count=candidate['source_count'],
                    shared_point_feedback=shared_point_feedback(cases[VARIANTS[0]], cases[VARIANTS[1]]),
                    same_source_layout=cases[VARIANTS[0]]['sources'] == cases[VARIANTS[1]]['sources']))
    stable = all(source_stability) and source_hashes() == snapshot
    passed = stable and all(r['audit']['passed'] and r['error'] is None and r['all_cleared'] for r in rows)
    summary = dict(passed=passed, source_snapshot_stable=stable, rows=rows, pairs=pairs,
        overall=aggregate(rows, pairs),
        by_field={field:aggregate([r for r in rows if r['field'] == field],
                                  [p for p in pairs if p['field'] == field]) for field in fields},
        independent_layouts=len(seeds), paired_conditions=len(pairs), official_contacted=False,
        performance_comparison_valid=passed,
        warning='Multiple fields reuse a layout. Count layouts and conditions separately. If any correctness audit fails, timing differences are diagnostic only.')
    write(out/'rows.json', rows)
    write(out/'summary.json', summary)
    print(json.dumps(dict(passed=passed, source_snapshot_stable=stable, overall=summary['overall']), ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
