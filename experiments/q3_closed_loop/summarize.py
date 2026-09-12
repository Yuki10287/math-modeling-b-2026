"""Recheck recorded hashes and separate descriptive results from selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from bootstrap import ROOT
from benchmark_joint import summary


def batch(directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    recorded = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert recorded['source_snapshot_stable']
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
               for path, digest in manifest['source_sha256'].items())
    rows = json.loads((directory/'rows.json').read_text(encoding='utf-8'))
    details = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((directory/'cases').glob('*.json'))]
    bases = {d['result']['name']: d for d in details if d['result']['schedule'] == 'lean'}
    result = summary(rows)
    result.update(conditions=len(bases), independent_layouts=len({r['seed'] for r in rows}),
                  all_passed=all(r['passed'] for r in rows), source_hashes_current=True)
    for variant, group in result['groups'].items():
        ds = [d for d in details if d['result']['schedule'] == variant]
        forecasts = [r for d in ds for r in d['trace']
                     if r.get('phase') in ('closed_scan_forecast', 'conditional_forecast')]
        group.update(event_equal_to_lean=sum(d['events'] == bases[d['result']['name']]['events'] for d in ds),
            forecasts=len(forecasts), changed=sum(r['changed'] for r in forecasts),
            cached_forecasts=sum(r['cached'] for r in forecasts),
            unavailable_forecasts=sum(r['prediction'] is None for r in forecasts),
            failed_scenario_continuations=sum(cost is None for r in forecasts if r['prediction'] is not None
                for candidate in r['prediction']['candidates'] for cost in candidate['costs_s']))
        group['plan_states_with_pending_known_sources'] = sum(
            r.get('phase') == 'plan' and any(t['kind'] == 'source' for t in r['route'])
            for d in ds for r in d['trace'])
        group['scan_first_states_with_pending_known_sources'] = sum(
            r.get('phase') == 'plan' and r['route'][0]['kind'] == 'scan'
            and any(t['kind'] == 'source' for t in r['route'])
            for d in ds for r in d['trace'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batches', required=True, nargs='+', type=Path)
    args = parser.parse_args()
    results = {path.name: batch(path) for path in args.batches}
    expanded = results['development-expanded-v1']
    eligible = [name for name in ('closed', 'closed_mean')
                if expanded['all_passed']
                and expanded['comparisons_to_lean'][name]['reduction_pct'] >= 2.
                and expanded['comparisons_to_lean'][name]['faster'] > expanded['comparisons_to_lean'][name]['slower']]
    conditional_rows = [r for directory in args.batches if directory.name.startswith('conditional-')
                        for r in json.loads((directory/'rows.json').read_text(encoding='utf-8'))
                        if r['schedule'] != 'ledger_reference']
    conditional_summary = summary(conditional_rows) if conditional_rows else None
    conditional_eligible = [name for name in ('conditional', 'conditional_mean')
        if conditional_summary is not None and all(r['passed'] for r in conditional_rows)
        and conditional_summary['comparisons_to_lean'][name]['reduction_pct'] >= 2.
        and conditional_summary['comparisons_to_lean'][name]['faster'] > conditional_summary['comparisons_to_lean'][name]['slower']]
    payload = dict(local_only=True, official_simulator_used=False, batches=results,
                   holdout_eligible=eligible,
                   conditional_combined=conditional_summary, conditional_holdout_eligible=conditional_eligible,
                   source_hashes_verified=True,
                   note='Repeated fields on a source layout are dependent. Cached runtimes cannot be used as independent runtime estimates.')
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(dict(batches={k:v['comparisons_to_lean'] for k,v in results.items()},
                          holdout_eligible=eligible, conditional_holdout_eligible=conditional_eligible)), flush=True)


if __name__ == '__main__':
    main()
