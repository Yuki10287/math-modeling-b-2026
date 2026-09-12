"""Re-evaluate every logged pre-contraction input, including unchanged regions.

The original benchmark's fallback counter matched an obsolete reason string.
This supplementary audit reconstructs the exact inputs from belief/action and
contraction trace rows and checks every wrapper outcome without rerunning or
changing actions. It does not substitute for independent geometric proof audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import negative_region_boxes_v1 as model


def inspect_case(path):
    case = json.loads(path.read_text(encoding='utf-8'))
    prefixes, current, reasons = {}, None, Counter()
    attempts = applied = errors = 0
    for row in case['trace']:
        if row['phase'] == 'actual_action':
            current = row
        elif row['phase'] == 'negative_contraction':
            prefixes[(row['event'], row['channel'])] = row['original_polygon']
        elif row['phase'] == 'belief':
            assert current is not None and current['action'] == 'measure'
            assert current['channel'] == row['channel']
            key = (current['event'], row['channel'])
            before = prefixes.get(key, row['polygon'])
            after, info = model.contract_position(before, row['positives'], row['negatives'])
            assert np.array_equal(after, np.asarray(row['polygon'])), (path.name, key, 'region differs')
            assert info['applied'] == (key in prefixes), (path.name, key, 'application differs')
            attempts += 1
            applied += int(info['applied'])
            errors += int('error' in info)
            reasons[info['reason']] += 1
    return dict(case=path.name, passed=True, attempts=attempts, applied=applied,
                error_fallbacks=errors, reasons=dict(reasons),
                input_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.results/'manifest.json').read_text())
    path = Path(model.__file__).resolve()
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = [v for k, v in manifest['source_sha256'].items() if k.endswith('/negative_region_boxes_v1.py')]
    assert expected == [fingerprint]
    rows = [inspect_case(p) for p in sorted(args.results.glob('*-negative_contraction25.json'))]
    assert rows
    report = dict(passed=True, local_only=True, official_simulator_contacted=False,
        candidate_model_sha256=fingerprint, cases=rows,
        attempts=sum(r['attempts'] for r in rows), contractions=sum(r['applied'] for r in rows),
        error_fallbacks=sum(r['error_fallbacks'] for r in rows),
        scope='Supplementary deterministic wrapper outcome audit; no new strategy performance score.',
        limitation='Original counter omitted this exception reason; this file supplies the corrected check.')
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
