"""Descriptive paired statistics; repeated fields are not independent layouts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from bootstrap import ROOT
from benchmark_joint import summary


def describe(rows):
    result = summary(rows)
    result['conditions'] = len({r['name'] for r in rows})
    result['layout_seeds'] = len({r['seed'] for r in rows})
    result['all_passed'] = all(r['passed'] for r in rows)
    for variant, group in result['groups'].items():
        selected = [r for r in rows if r['schedule'] == variant]
        group['mean_time_parts_s'] = {
            key: float(np.mean([r['time_parts_s'].get(key, 0) for r in selected]))
            for key in sorted({key for r in selected for key in r['time_parts_s']})}
        group['probe_outcomes'] = dict(sum((Counter(r.get('probe_outcomes', {})) for r in selected), Counter()))
        group['probe_selected'] = sum(r.get('probe_selected', 0) for r in selected)
        group['scan_previews'] = sum(r.get('scan_previews', 0) for r in selected)
        group['pooled_seconds_per_source'] = sum(r['total_s'] for r in selected) / group['sources']
    base = {r['name']: r for r in rows if r['schedule'] == 'lean'}
    for variant, comparison in result['comparisons_to_lean'].items():
        paired = [(base[r['name']], r) for r in rows if r['schedule'] == variant]
        comparison['mean_saved_by_cost_s'] = {
            key: float(np.mean([a['time_parts_s'].get(key, 0)-b['time_parts_s'].get(key, 0)
                                for a, b in paired]))
            for key in result['groups']['lean']['mean_time_parts_s']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    all_rows, reports = [], {}
    for directory in args.batches:
        rows = json.loads((directory / 'rows.json').read_text(encoding='utf-8'))
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        completed = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        assert completed['source_snapshot_stable'], directory
        report = describe(rows)
        report['split'] = manifest.get('split', manifest.get('arguments', {}).get('split'))
        report['source_snapshot_stable'] = True
        reports[directory.name] = report
        all_rows.extend(rows)
    development = [r for d in args.batches if reports[d.name]['split'] == 'development'
                   for r in json.loads((d / 'rows.json').read_text(encoding='utf-8'))
                   if r['schedule'] != 'reference']
    combined = describe(development) if development else None
    strata = {}
    if development:
        for population in ('uniform', 'sparse10', 'boundary10', 'one_side10'):
            subset = [r for r in development if f'-{population}-' in r['name']]
            if subset:
                strata[population] = describe(subset)
    payload = dict(local_only=True, official_simulator_used=False, main_solver_replaced=False,
                   batches=reports, development_combined=combined, development_by_population=strata,
                   note='Descriptive development/stress results, not independent official performance estimates; repeated error fields share source layouts.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(dict(batches={k: v['comparisons_to_lean'] for k, v in reports.items()},
                          development_combined=combined['comparisons_to_lean'] if combined else None)), flush=True)


if __name__ == '__main__':
    main()
