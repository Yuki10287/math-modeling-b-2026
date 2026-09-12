"""Read-only experiment analysis; no solver, simulator, or network is imported.

Usage: python -X utf8 analyze_cell_experiment.py --directory results/my-cells
Only analysis.json in the supplied completed experiment directory is written.
All gains are reference minus candidate, so positive values mean improvement.
Error fields from one source layout are aggregated before layout comparisons.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def key(row):
    return row['seed'], row['population'], row['field']


def identity(row):
    return dict(seed=row['seed'], population=row['population'], field=row['field'])


def close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f'{label}: {a!r} != {b!r}')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def trace_statistics(case):
    """Cross-check event accounting and labels, then collect observed statistics."""
    row, events, trace = case['summary'], case['events'], case['trace']
    actual = [t for t in trace if t['phase'] == 'actual_action']
    if len(actual) != len(events):
        raise ValueError('Actual-action trace does not label every event exactly once')
    reason_counts, move_by_reason = Counter(), Counter()
    positions, batches = defaultdict(set), Counter()
    counts = Counter(measure=0, switch=0, clear_success=0, clear_fail=0)
    part = Counter(move=0., measure=0., switch=0., clear_success=0., clear_fail=0.)
    source_channels = {int(s['channel']) for s in case['sources']}
    cleared, found = set(), set()
    previous_position, previous_time, current_channel = (0., 0.), 0., 1
    previous_batch = None
    absent_measures = 0
    last_clear_time = 0.
    for index, (event, label) in enumerate(zip(events, actual), 1):
        if label['event'] != index or label['action'] != event['action']:
            raise ValueError('Action labels have wrong indices or actions')
        if label['channel'] != event['channel'] or label['position'] != event['position']:
            raise ValueError('Action labels disagree with event positions/channels')
        position = tuple(event['position'])
        travel = math.dist(previous_position, position)/5
        close(travel, label['move_s'], 'Action movement')
        reason, channel = label['reason'], int(event['channel'])
        reason_counts[reason] += 1
        move_by_reason[reason] += travel
        positions[reason].add(position)
        part['move'] += travel
        fee = travel
        if event['action'] == 'measure':
            kind = event['measure_result']
            if kind not in ('no_signal', 'near', 'direction'):
                raise ValueError('Unknown measurement result')
            counts['measure'] += 1
            part['measure'] += 5
            fee += 5
            if channel != current_channel:
                counts['switch'] += 1
                part['switch'] += 1
                fee += 1
            current_channel = channel
            absent_measures += int(channel not in source_channels)
            if kind in ('near', 'direction'):
                found.add(channel)
        elif event['action'] == 'clear':
            kind = event['clear_result']
            if kind == 'success':
                if channel in cleared:
                    raise ValueError('Duplicate successful clearance')
                cleared.add(channel)
                counts['clear_success'] += 1
                part['clear_success'] += 5
                fee += 5
                last_clear_time = event['time_s']
            elif kind == 'no_target_in_range':
                counts['clear_fail'] += 1
                part['clear_fail'] += 3
                fee += 3
            else:
                raise ValueError('Unknown clear result')
        else:
            raise ValueError('Unknown action')
        if kind != label['result']:
            raise ValueError('Trace feedback labels disagree with actual events')
        close(event['time_s']-previous_time, fee, 'Event fee')
        batch = (reason, position) if reason in ('scan', 'opportunity_scan') else None
        if batch is not None and batch != previous_batch:
            batches[reason] += 1
        previous_batch = batch
        previous_position, previous_time = position, event['time_s']
    close(previous_time, row['total_s'], 'Final time')
    for component, value in part.items():
        close(value, row['time_parts_s'][component], f'Time component {component}')
    if dict(counts) != row['counts']:
        raise ValueError('Action counts disagree with summary')
    if row['cleared'] != len(cleared) or row['source_count'] != len(source_channels):
        raise ValueError('Source/clear counts disagree with raw case')
    raw_full_clear = cleared == source_channels
    if bool(row['all_cleared']) != raw_full_clear:
        raise ValueError('Full-clear flag disagrees with successful source channels')
    close(row['average_s'], row['total_s']/row['source_count'], 'Per-source time')
    phases = Counter(t['phase'] for t in trace)
    tasks = Counter(t['task'] for t in trace if t['phase'] == 'task_choice')
    opportunities = [t for t in trace if t['phase'] == 'opportunity_prediction']
    planned_points = sum(len(t['path']) for t in trace if t['phase'] == 'optical_plan')
    return dict(
        event_count=len(events), raw_all_cleared=raw_full_clear,
        result_complete=bool(case['result'] and case['result'].get('complete')),
        action_reason_counts=dict(reason_counts), movement_s_by_action_reason=dict(move_by_reason),
        distinct_positions_by_reason={k:len(v) for k,v in positions.items()},
        consecutive_scan_batches=dict(batches), task_choice_counts=dict(tasks),
        optical_plans=phases['optical_plan'], planned_optical_path_points=planned_points,
        opportunity_predictions=len(opportunities),
        predicted_station_removals_sum=sum(len(t['removed_scan_stations']) for t in opportunities),
        distinct_predicted_station_indices=len({i for t in opportunities for i in t['removed_scan_stations']}),
        measured_absent_channels=absent_measures,
        final_tail_after_last_success_s=row['total_s']-last_clear_time,
        discovered_channels_from_positive_measurements=len(found))


def means_for_dict(records, name):
    names = sorted({k for r in records for k in r[name]})
    return {k:mean(r[name].get(k, 0) for r in records) for k in names}


def summarize(rows, diagnostics):
    records = [diagnostics[(r['variant'], *key(r))] for r in rows]
    source_total = sum(r['source_count'] for r in rows)
    result = dict(
        conditions=len(rows), independent_layouts=len({key(r)[:2] for r in rows}),
        source_instances=source_total,
        mean_total_s=mean(r['total_s'] for r in rows),
        mean_case_per_source_s=mean(r['average_s'] for r in rows),
        pooled_per_source_s=sum(r['total_s'] for r in rows)/source_total,
        mean_runtime_s=mean(r['runtime_s'] for r in rows),
        max_runtime_s=max(r['runtime_s'] for r in rows),
        audit_passes=sum(bool(r['audit']['passed']) for r in rows),
        all_cleared_runs=sum(bool(r['all_cleared']) for r in rows),
        raw_event_full_clear_runs=sum(r['raw_all_cleared'] for r in records),
        solver_complete_runs=sum(r['result_complete'] for r in records),
        mean_time_parts_s=means_for_dict(rows, 'time_parts_s'),
        mean_action_counts=means_for_dict(rows, 'counts'))
    scalar = ('event_count', 'optical_plans', 'planned_optical_path_points',
              'opportunity_predictions', 'predicted_station_removals_sum',
              'distinct_predicted_station_indices', 'measured_absent_channels',
              'final_tail_after_last_success_s', 'discovered_channels_from_positive_measurements')
    result['mean_trace_statistics'] = {k:mean(r[k] for r in records) for k in scalar}
    for name in ('action_reason_counts', 'movement_s_by_action_reason',
                 'distinct_positions_by_reason', 'consecutive_scan_batches', 'task_choice_counts'):
        result['mean_trace_statistics'][name] = means_for_dict(records, name)
    return result


def signs(values):
    return dict(win=sum(x>1e-6 for x in values), tie=sum(abs(x)<=1e-6 for x in values),
                loss=sum(x < -1e-6 for x in values))


def paired(reference, candidate):
    a, b = {key(r):r for r in reference}, {key(r):r for r in candidate}
    if a.keys() != b.keys():
        raise ValueError('Candidate/reference condition sets are not identical')
    records = []
    layout = defaultdict(list)
    for k in sorted(a):
        if a[k]['source_count'] != b[k]['source_count']:
            raise ValueError('Paired source counts differ')
        old, new = a[k]['total_s'], b[k]['total_s']
        r = dict(**identity(a[k]), source_count=a[k]['source_count'],
                 reference_s=old, candidate_s=new, saving_s=old-new,
                 relative_reduction_percent=100*(old-new)/old)
        records.append(r)
        layout[k[:2]].append(r)
    gains = [r['saving_s'] for r in records]
    layout_records = []
    for (seed, population), rr in sorted(layout.items()):
        old, new = mean(r['reference_s'] for r in rr), mean(r['candidate_s'] for r in rr)
        layout_records.append(dict(seed=seed, population=population, fields=[r['field'] for r in rr],
            reference_mean_s=old, candidate_mean_s=new, saving_s=old-new,
            relative_reduction_percent=100*(old-new)/old))
    old, new = mean(r['reference_s'] for r in records), mean(r['candidate_s'] for r in records)
    old_layout = mean(r['reference_mean_s'] for r in layout_records)
    new_layout = mean(r['candidate_mean_s'] for r in layout_records)
    parts = sorted({p for r in reference+candidate for p in r['time_parts_s']})
    return dict(reference=reference[0]['variant'], candidate=candidate[0]['variant'],
        paired_conditions=len(records), independent_layouts=len(layout_records),
        reference_mean_total_s=old, candidate_mean_total_s=new, mean_saving_s=old-new,
        relative_reduction_percent=100*(old-new)/old,
        condition_outcomes=signs(gains), worst_regression_s=max(0., -min(gains)),
        largest_improvement_s=max(0., max(gains)),
        worst_case=min(records, key=lambda r:r['saving_s']),
        best_case=max(records, key=lambda r:r['saving_s']),
        mean_time_part_savings_s={p:mean(a[k]['time_parts_s'].get(p,0)-b[k]['time_parts_s'].get(p,0) for k in a) for p in parts},
        layout_aggregated=dict(reference_mean_total_s=old_layout, candidate_mean_total_s=new_layout,
            mean_saving_s=old_layout-new_layout,
            relative_reduction_percent=100*(old_layout-new_layout)/old_layout,
            outcomes=signs([r['saving_s'] for r in layout_records]), records=layout_records),
        condition_records=records)


def analyze(directory):
    paths = [directory/name for name in ('manifest.json','rows.json','summary.json')]
    manifest, rows, summary = [read(p) for p in paths]
    args = manifest['arguments']
    populations, fields, variants = [args[k].split(',') for k in ('populations','fields','variants')]
    expected = {(variant, args['start_seed']+100*i+j, population, field)
                for i,population in enumerate(populations) for j in range(args['layouts'])
                for field in fields for variant in variants}
    actual = [(r['variant'], *key(r)) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Incomplete, duplicate, or unexpected experiment rows')
    if 'baseline25' not in variants:
        raise ValueError('A baseline25 paired reference is required')
    if manifest.get('local_only') is not True or manifest.get('official_simulator_used') is not False:
        raise ValueError('Expected an explicitly local-only experiment')
    diagnostics, sources, checksums = {}, {}, {}
    for row in rows:
        filename = f'{row["seed"]}-{row["population"]}-{row["field"]}-{row["variant"]}.json'
        path = directory/filename
        case = read(path)
        if case['summary'] != row:
            raise ValueError(f'Row differs from raw summary: {filename}')
        signature = json.dumps(case['sources'], sort_keys=True, separators=(',',':'))
        lk = key(row)[:2]
        if lk in sources and sources[lk] != signature:
            raise ValueError('Paired variants/error fields have different source layouts')
        sources[lk] = signature
        diagnostics[(row['variant'], *key(row))] = trace_statistics(case)
        checksums[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = {v:summarize([r for r in rows if r['variant']==v], diagnostics) for v in variants}
    for v,g in groups.items():
        saved = summary['groups'][v]
        for source,target in [('mean_total_s','mean_total_s'),('pooled_per_source_s','pooled_per_source_s'),
                              ('conditions','runs'),('audit_passes','passed'),('all_cleared_runs','all_cleared')]:
            close(g[source], saved[target], f'Saved group {v} {source}')
    by_variant = {v:[r for r in rows if r['variant']==v] for v in variants}
    comparisons = {v:paired(by_variant['baseline25'], by_variant[v]) for v in variants if v!='baseline25'}
    incremental = {}
    for old,new in [('baseline25','cells'),('cells','context'),('context','interleave'),('interleave','opscan')]:
        if old in by_variant and new in by_variant:
            incremental[f'{old}_to_{new}'] = paired(by_variant[old], by_variant[new])
    by_population = {}
    for population in populations:
        pop_rows = {v:[r for r in by_variant[v] if r['population']==population] for v in variants}
        by_population[population] = dict(
            groups={v:summarize(rr, diagnostics) for v,rr in pop_rows.items()},
            comparisons={v:paired(pop_rows['baseline25'], rr) for v,rr in pop_rows.items() if v!='baseline25'})
    safety = dict(all_audits_passed=all(r['audit']['passed'] for r in rows),
        all_full_clears=all(r['all_cleared'] for r in rows),
        all_raw_events_clear_every_source=all(r['raw_all_cleared'] for r in diagnostics.values()),
        all_solver_results_complete=all(r['result_complete'] for r in diagnostics.values()),
        source_snapshot_stable=summary.get('source_snapshot_stable') is True,
        raw_rows_and_saved_summary_match=True, event_accounting_rechecked=True,
        source_layouts_identical_across_variants_and_fields=True)
    result = dict(description='Paired descriptive analysis of completed local cell-cover experiments; no candidate selection.',
        experiment_directory=str(directory.resolve()), conditions_per_variant=len(expected)//len(variants),
        independent_layouts=len(sources), fields=fields, safety_checks=safety,
        groups=groups, comparisons_to_baseline25=comparisons, incremental_comparisons=incremental,
        by_population=by_population,
        interpretation=dict(gain_sign='Positive savings mean reference minus candidate.',
            pooled_per_source='Sum of whole-task times divided by sum of source counts; not the mean of case averages.',
            independence='Error fields on one (seed, population) source layout are paired repetitions. Layout comparisons average fields first; no independent-sample uncertainty claim is made.',
            incremental='Adjacent-variant differences are empirical marginal effects in this fixed order; interactions prevent treating them as universal independent benefits.',
            movement='Movement is attributed to the following actual action reason; scan movement can include travel departing a cleared source.',
            stations='Distinct positions and contiguous scan batches are observed; they are not a proof that a fixed station was eliminated.',
            opportunity='Predicted station removals describe the all-negative design scenario, not proven actual or counterfactual savings.',
            optical='Planned optical points include unused suffixes after success; actual optical actions are counted separately.',
            audit='Stored independent audit flags are checked, not recomputed; event counts and time accounting are rechecked without importing a solver.'),
        provenance=dict(input_metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            raw_case_sha256=checksums, solver_code_hashes=manifest['code_hashes'],
            analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            local_only=True, official_simulator_used=False))
    (directory/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(sources)} independent layouts; {len(expected)//len(variants)} paired conditions; all checks={all(safety.values())}')
    for v,g in groups.items():
        extra = '' if v=='baseline25' else f'; saving={comparisons[v]["relative_reduction_percent"]:.2f}%; W/T/L={comparisons[v]["condition_outcomes"]}'
        print(f'{v}: mean={g["mean_total_s"]:.2f}s; pooled={g["pooled_per_source_s"]:.2f}s/source{extra}')
    print(f'Analysis saved: {directory / "analysis.json"}')
    return 0 if all(safety.values()) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    options = parser.parse_args()
    raise SystemExit(analyze(options.directory))
