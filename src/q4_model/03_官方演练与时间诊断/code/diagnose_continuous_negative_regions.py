"""Dense position-sample diagnosis with continuous heading feasibility.

For each position g the unknown radius is eliminated as
L(g)=max(1000,max distance to positive receivers). Increasing R beyond L can
only activate more negative receiver constraints. Active negative receivers
therefore have distance <= L; their direction normals must be strictly opposite
to the transmitter's heading. Unit headings are not sampled: feasibility is
the circular half-plane intersection (maximum angular gap >= pi).

This is a floating-point, dense position-sample screen, NOT a certified set
contraction and NOT a simulation of a changed policy. Boundary-ambiguous
samples are retained. It reads saved positive/negative points and P only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def grid(P, nx, ny):
    P = np.asarray(P, float)
    center = P.mean(axis=0)
    distances = np.linalg.norm(P[:, None]-P, axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    direction = P[j]-P[i]
    direction = direction/np.linalg.norm(direction) if np.linalg.norm(direction) else np.array([1., 0.])
    rotation = np.column_stack((direction, [-direction[1], direction[0]]))
    local = (P-center)@rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    area = abs(float(np.sum(P[:, 0]*np.roll(P[:, 1], -1)-P[:, 1]*np.roll(P[:, 0], -1))))/2
    if area <= 1e-12 or hi[1]-lo[1] < 1e-10:
        values = (np.arange(nx)+.5)/nx
        points = P[i]+values[:, None]*(P[j]-P[i])
        return points, center, rotation, area, 'line_sample'
    x = lo[0]+(np.arange(nx)+.5)*(hi[0]-lo[0])/nx
    y = lo[1]+(np.arange(ny)+.5)*(hi[1]-lo[1])/ny
    xx, yy = np.meshgrid(x, y)
    candidates = center+np.column_stack((xx.ravel(), yy.ravel()))@rotation.T
    edges = np.roll(P, -1, axis=0)-P
    v = candidates[:, None]-P
    cross = edges[None, :, 0]*v[:, :, 1]-edges[None, :, 1]*v[:, :, 0]
    orientation = np.sum(P[:, 0]*np.roll(P[:, 1], -1)-P[:, 1]*np.roll(P[:, 0], -1))
    inside = (cross >= -1e-9).all(axis=1) if orientation >= 0 else (cross <= 1e-9).all(axis=1)
    return candidates[inside], center, rotation, area, 'oriented_rectangle_uniform_grid'


def classify(points, positives, negatives):
    p = np.asarray(positives, float).reshape(-1, 2)
    q = np.asarray(negatives, float).reshape(-1, 2)
    delta = p[None]-points[:, None]
    radius = np.maximum(1000., np.linalg.norm(delta, axis=2).max(axis=1))
    # P is a conservative polygon around physical disk intersections. Keep its
    # approximation excess distinct from additional negative-data exclusions.
    positive_valid = (radius <= 1500.+1e-8) & (np.linalg.norm(points, axis=1) <= 1800.+1e-8)
    if not len(q):
        return positive_valid, positive_valid.copy(), np.ones(len(points), bool), np.zeros(len(points), bool)
    neg_delta = points[:, None]-q[None]
    distances = np.linalg.norm(neg_delta, axis=2)
    active = distances < radius[:, None]-1e-8
    radial_boundary = (np.abs(distances-radius[:, None]) <= 1e-8).any(axis=1)
    omni = ~active.any(axis=1)
    positive_angles = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    negative_angles = np.arctan2(neg_delta[:, :, 1], neg_delta[:, :, 0])
    negative_angles[~active] = np.inf
    angles = np.sort(np.concatenate((positive_angles, negative_angles), axis=1), axis=1)
    counts = len(p)+active.sum(axis=1)
    with np.errstate(invalid='ignore'):
        gaps = np.diff(angles, axis=1)
    gaps[np.arange(gaps.shape[1])[None] >= counts[:, None]-1] = 0.
    wrap = angles[:, 0]+2*math.pi-angles[np.arange(len(points)), counts-1]
    maximum_gap = np.maximum(gaps.max(axis=1), wrap)
    angular_boundary = np.abs(maximum_gap-math.pi) <= 1e-10
    directional = maximum_gap >= math.pi-1e-10
    # At a negative receiver itself both models are impossible. This is a
    # degenerate measure-zero event for the diagnostic grid.
    at_negative = (distances <= 1e-12).any(axis=1)
    directional &= ~at_negative
    omni &= ~at_negative
    return positive_valid, positive_valid & (omni | directional), omni, radial_boundary | angular_boundary


def analyze(row, nx, ny):
    points, center, rotation, area, method = grid(row['polygon'], nx, ny)
    valid, compatible, omni, ambiguous = classify(points, row['positives'], row['negatives'])
    axis = ((points-center)@rotation)[:, 0]
    def width(mask):
        return float(np.ptp(axis[mask])) if np.count_nonzero(mask) >= 2 else 0.
    excluded = valid & ~compatible
    return dict(grid=[nx, ny], method=method, samples=len(points), polygon_area_m2=area,
        samples_inside_exact_positive_radius_and_domain=int(valid.sum()),
        outer_polygon_approximation_excess=int((~valid).sum()),
        negative_incompatible_samples=int(excluded.sum()), compatible_samples=int(compatible.sum()),
        negative_excluded_sample_fraction=float(excluded.sum()/valid.sum()) if valid.any() else None,
        approximate_negative_excluded_area_m2=float(area*excluded.sum()/len(points)),
        sampled_initial_long_axis_span_m=width(valid), sampled_compatible_long_axis_span_m=width(compatible),
        sampled_long_axis_span_reduction_m=width(valid)-width(compatible),
        omni_compatible_samples=int((valid & omni).sum()),
        boundary_ambiguous_samples_retained=int((compatible & ambiguous).sum()),
        positive_measurements=len(row['positives']), negative_measurements=len(row['negatives']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--nx', type=int, default=320)
    parser.add_argument('--ny', type=int, default=48)
    args = parser.parse_args()
    assert args.nx > 1 and args.ny > 1
    if args.out.exists():
        raise FileExistsError(args.out)
    final, detailed = [], []
    with zipfile.ZipFile(args.zip) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith('.jsonl.beliefs.json')):
            case = Path(name.removesuffix('.jsonl.beliefs.json')).name
            traces = json.loads(archive.read(name))
            last = {}
            event = 0
            for index, row in enumerate(traces):
                if row['phase'] == 'actual_action':
                    event = row['event']
                elif row['phase'] == 'belief':
                    last[row['channel']] = (index, event, row)
                    if (case.endswith('-01') and row['channel'] == 20 or
                            case.endswith('-03') and row['channel'] == 14):
                        detailed.append(dict(case=case, channel=row['channel'], trace_index=index,
                            after_event=event, **analyze(row, args.nx, args.ny)))
            for channel, (index, event, row) in sorted(last.items()):
                final.append(dict(case=case, channel=channel, trace_index=index,
                    after_event=event, **analyze(row, args.nx, args.ny)))
    fractions = [r['negative_excluded_sample_fraction'] for r in final if r['negative_excluded_sample_fraction'] is not None]
    result = dict(archive_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_simulator_contacted=False, runtime_solver_changed=False,
        heading_grid_used=False, radius_dimension_eliminated=True,
        certified_continuous_region_contraction=False, counterfactual_time_estimated=False,
        final_beliefs=final, long_region_stages=detailed,
        aggregate=dict(sources=len(final), mean_fraction=float(np.mean(fractions)),
            median_fraction=float(np.median(fractions)),
            sources_above_10pct=sum(x > .1 for x in fractions),
            sources_above_50pct=sum(x > .5 for x in fractions)),
        limitations=['Position samples cannot prove a whole-region exclusion or a safe clearance.',
            'Heading feasibility is continuous mathematically, but evaluated with floating point here.',
            'Near-boundary samples are retained rather than declared impossible.',
            'Only the positive-measurement polygon and its positive/negative receiver ledger are used.',
            'No ground-truth position, orientation, radius, or official count is supplied to the analysis.',
            'No alternative action receives a saved official response.'])
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result['aggregate'], ensure_ascii=False))
    print(json.dumps(detailed, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
