"""Development-only ablation for range-independent short lateral baselines."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import geometry as core
import local_checks

original_candidates = core.candidates


def enriched_candidates(P, s, witness=None):
    witness = s if witness is None else witness
    options = original_candidates(P, s, witness)
    c, r = core.mec(P)
    _, (i, j) = core.diameter(P)
    axis = P[j] - P[i]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    perp = np.array([-axis[1], axis[0]])
    # Fixed meters encode the actual optical radius; they avoid scaling all
    # useful cross-range travel with a 1500 m radial uncertainty interval.
    for shift in (20., 40., 80.):
        if shift >= .3 * r:
            continue
        for sign in (-1., 1.):
            for along in (0., -.25 * r, .25 * r):
                q = c + along * axis + sign * shift * perp
                if np.linalg.norm(q - s) > .1 and core.reception_certified(P, q, witness):
                    if all(np.linalg.norm(q - p) > .1 for p in options):
                        options.append(q)
    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1000')
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--output', default='local_short_baseline_experiment.json')
    args = parser.parse_args()
    core.candidates = enriched_candidates
    results = local_checks.development([int(x) for x in args.seeds.split(',')],
                                       args.fields.split(','), ['hybrid'])
    results['model_change'] = 'additional fixed 20,40,80m lateral candidates, original expected-risk objective'
    (Path(__file__).parent / args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results['summary'], indent=2))
