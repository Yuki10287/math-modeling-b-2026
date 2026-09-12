"""Read de-identified official task blocks and compute a hindsight optimum."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from block_route_diagnostic import solve_blocks

BASE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE/'results/official_time_audit_v1.json')
    parser.add_argument('--out', type=Path, default=BASE/'results/block_route_diagnostic_v1.json')
    parser.add_argument('--precedence', choices=['channel_order', 'removal_state'], default='channel_order')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.input.read_text(encoding='utf-8'))
    results = []
    for case in audit['cases']:
        start = time.perf_counter()
        result = solve_blocks(case['task_blocks'], precedence=args.precedence)
        result.update(case=case['case'], blocks=len(case['task_blocks']),
                      runtime_s=time.perf_counter()-start)
        results.append(result)
        print(json.dumps({k:result[k] for k in ('case','blocks','original_s','total_s','saved_s','states','runtime_s')}), flush=True)
    report = dict(official_simulator_contacted=False, online_solver_changed=False,
        hidden_truth_used=False, method='Exact subset DP of recorded blocks', precedence=args.precedence,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), Path(__file__).with_name('block_route_diagnostic.py')]},
        cases=results, aggregate=dict(cases=len(results),
            mean_original_s=sum(r['original_s'] for r in results)/len(results),
            mean_reordered_s=sum(r['total_s'] for r in results)/len(results),
            mean_saved_s=sum(r['saved_s'] for r in results)/len(results)),
        limitations=[
            'Hindsight uses recorded future action points, unavailable to an online planner.',
            'Optimum is only for fixed blocks, fixed positions and the stated precedence constraints.',
            'This does not bound gains from changing points, splitting blocks or changing within-block action order.',
            'Feedback preservation follows Q3 static source and fixed-position feedback rules; no official replay performed.'
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['aggregate']), flush=True)


if __name__ == '__main__':
    main()
