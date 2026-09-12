"""Adversarial checks that independent station-geometry audits reject bad proof data."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from audit_station_certificates_v1 import audit
from validate_station22_v1 import validate
from station22_geometry_v1 import CERTIFICATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(CERTIFICATE.read_text())
    mutations = {}
    x = deepcopy(data); x['cells'].pop(); mutations['missing_leaf'] = x
    x = deepcopy(data); x['cells'].append(deepcopy(x['cells'][0])); mutations['duplicate_leaf'] = x
    x = deepcopy(data); x['cells'][0]['witnesses'] = x['cells'][0]['witnesses'][:2]; mutations['insufficient_witness'] = x
    x = deepcopy(data); x['cells'][0]['branch'] += '0'; mutations['incomplete_midpoint_tree'] = x
    x = deepcopy(data); x['stations'][8] = [0., 0.]; mutations['outer_vertex_removed'] = x
    rows = []
    for name, value in mutations.items():
        path = args.out/(name+'.json')
        path.write_text(json.dumps(value), encoding='utf-8')
        try:
            audit(path)
        except (AssertionError, ValueError, IndexError) as exc:
            rows.append(dict(name=name, rejected=True, error=type(exc).__name__))
        else:
            raise AssertionError('bad static certificate accepted: '+name)
    case = json.loads(args.case.read_text())
    original = case['result']
    validate(original, case['events'], CERTIFICATE)
    mutations = {}
    absent = original['certificate']['absent'][0]
    x = deepcopy(original); x['certificate']['channels'][str(absent)]['negative_points'][0][0] += 2.**-30
    mutations['unobserved_nearby_negative'] = x
    x = deepcopy(original); x['certificate']['absent'].append(absent); mutations['duplicate_absent_channel'] = x
    x = deepcopy(original); x['certificate']['basis'] = 'count_upper_bound'; mutations['unjustified_16_source_stop'] = x
    x = deepcopy(original); x['certificate']['channels'][str(absent)]['triangle_witnesses']['0'] = [-1]
    mutations['negative_witness_index'] = x
    for name, value in mutations.items():
        try:
            validate(value, case['events'], CERTIFICATE)
        except (AssertionError, ValueError, IndexError) as exc:
            rows.append(dict(name=name, rejected=True, error=type(exc).__name__))
        else:
            raise AssertionError('bad actual certificate accepted: '+name)
    report = dict(passed=True, valid_control_passed=True, malformed_cases=rows,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('audit_station_certificates_v1.py'),
             Path(__file__).with_name('validate_station22_v1.py'))},
        input_case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest())
    (args.out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASSED', len(rows), 'malformed evidence cases rejected; genuine case accepted')


if __name__ == '__main__':
    main()
