"""Record local proof budgets on the two preidentified official long regions.

This is a geometry/proof workload diagnostic, not a changed-policy replay or
official score estimate. The complete online comparison uses separate seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/studies.json').is_file())
MODEL = ROOT/'src/q4_model/05_候选share25/code/negative_region_boxes_v1.py'
sys.path.insert(0, str(MODEL.parent))
from negative_region_boxes_v1 import contract_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert source_hash == '020a8f9088c9710412753d3d027d5d44d3a996f1b6bb7cb8e67b023c1a3e2da5'
    rows = []
    with zipfile.ZipFile(args.zip) as archive:
        for depth in (4, 6):
            for case, channel in ((1,20), (3,14)):
                name = f'测试结果/q4-share25-practice-{case:02}.jsonl.beliefs.json'
                traces = json.loads(archive.read(name))
                belief = [r for r in traces if r['phase'] == 'belief' and r['channel'] == channel][-1]
                started = time.perf_counter()
                result = contract_region(belief['polygon'], belief['positives'], belief['negatives'], depth, 255)
                elapsed = time.perf_counter()-started
                row = dict(case=case, channel=channel, max_depth=depth, max_nodes=255,
                    runtime_s=elapsed, result=result)
                rows.append(row)
                print(json.dumps(dict(case=case, channel=channel, max_depth=depth, runtime_s=elapsed,
                    **result['statistics']), ensure_ascii=False), flush=True)
    report = dict(model_source_sha256=source_hash,
        script_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archive_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        rows=rows, selected_online_budget=dict(max_depth=6, max_nodes=255),
        official_simulator_contacted=False, counterfactual_score_estimated=False,
        preliminary_unfrozen_observations=dict(
            provenance='First exploratory local invocation before adding the bounded online wrapper; source hash was not logged then.',
            depth4_runtime_s=[.028004099993268028, .019781599985435605],
            depth6_runtime_s=[.07919560000300407, .030953200010117143],
            depth4_convex_hull_area_reduction=[.43749999988750005, .7499999999499987],
            depth6_convex_hull_area_reduction=[.7499999999500001, .7499999999499987]),
        limitations=['These cases were selected after inspecting official failures; they are diagnostic, not holdout evidence.',
            'Runtime measurements vary between invocations and machines.',
            'Region reduction is not an estimate of saved virtual time.',
            'Each saved exclusion certificate still requires independent checker review.'])
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == source_hash
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
