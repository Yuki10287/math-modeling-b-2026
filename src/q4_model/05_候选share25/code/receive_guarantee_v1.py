"""Exact sufficient receipt certificate for the local cost-model experiment.

For a still-active source, each actual positive measurement lies in its fixed
receipt set. That set is a closed disk (omnidirectional) or a closed disk
intersected with a closed half-plane (directional), hence convex. Therefore
q in conv(actual_positive_points) guarantees a non-negative receipt response,
including the near response. No sampled position, heading, radius, or assumed
feedback is used in this theorem. The 180-degree boundary is included.

Only this sufficient condition is implemented. A false result means unproved,
not that a negative response is feasible. P is an interface parameter and is
deliberately not used: convexity proves both range and direction directly.
The certificate concerns exact binary coordinates supplied by the caller; it
does not repair an inaccurate positive ledger or communication rounding.

The caller must supply prior accepted positive points for this same uncleared
source. A hypothetical positive can be used only inside its own hypothetical
branch and must never become real evidence or eliminate other branches.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

VERSION = 'positive_convex_hull_receipt_v1'


def _point(row):
    if len(row) != 2:
        raise ValueError('Expected two coordinates')
    values = tuple(float(x) for x in row)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Coordinates must be finite')
    return tuple(Fraction(x) for x in values)


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _hull(points):
    ordered = sorted(set(points))
    if len(ordered) < 2:
        return ordered
    chains = []
    for sequence in (ordered, ordered[::-1]):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return chains[0][:-1]+chains[1][:-1]


def _combination(hull, q):
    if len(hull) == 1:
        return [(hull[0], Fraction(1))] if q == hull[0] else None
    if len(hull) == 2:
        a, b = hull
        k = 0 if b[0] != a[0] else 1
        t = (q[k]-a[k])/(b[k]-a[k])
        if 0 <= t <= 1 and all(q[j] == (1-t)*a[j]+t*b[j] for j in (0, 1)):
            return [(a, 1-t), (b, t)]
        return None
    for i in range(1, len(hull)-1):
        a, b, c = hull[0], hull[i], hull[i+1]
        area = _cross(a, b, c)
        weights = (_cross(b, c, q)/area, _cross(c, a, q)/area, _cross(a, b, q)/area)
        if min(weights) >= 0:
            return list(zip((a, b, c), weights))
    return None


def receipt_certificate(P, positives, q):
    """Return a JSON-safe certificate; only certified=True removes no_signal.

    Fail closed on empty/nonfinite/malformed input. Exact Fraction arithmetic
    certifies the supplied binary doubles, without geometric tolerances.
    """
    del P
    try:
        positive = tuple(_point(p) for p in positives)
        point = _point(q)
    except (TypeError, ValueError, OverflowError):
        return dict(certified=False, reason='invalid_input', proof=None, model=VERSION)
    if not positive:
        return dict(certified=False, reason='no_actual_positive_points', proof=None, model=VERSION)
    hull = _hull(positive)
    witness = _combination(hull, point)
    if witness is None:
        return dict(certified=False, reason='outside_positive_convex_hull_unproved', proof=None,
                    model=VERSION)
    proof = dict(
        type='exact_positive_convex_combination',
        q=[str(x) for x in point],
        witnesses=[dict(positive_index=positive.index(p), point=[str(x) for x in p],
                        weight=str(weight)) for p, weight in witness if weight],
        position_region_used=False,
        radius_and_direction_both_proved=True,
        boundary_included=True,
        premise='Same active source; fixed radius and type/direction; witnesses are prior accepted positive measurements.')
    return dict(certified=True, reason='positive_convex_hull', proof=proof, model=VERSION)


def _check_witness(certificate, positives, q):
    """Independent algebraic reconstruction used only by the local self-test."""
    proof = certificate['proof']
    weights = [Fraction(w['weight']) for w in proof['witnesses']]
    points = [tuple(Fraction(x) for x in w['point']) for w in proof['witnesses']]
    if sum(weights) != 1 or min(weights) < 0:
        return False
    if any(p != _point(positives[w['positive_index']])
           for p, w in zip(points, proof['witnesses'])):
        return False
    return all(sum(weight*p[k] for weight, p in zip(weights, points)) == _point(q)[k]
               for k in (0, 1))


def self_test():
    tests = []
    cases = [
        ('same_actual_positive', [(500, 0)], (500, 0), True),
        ('one_point_offline', [(500, 0)], (500, 1), False),
        ('segment_interior', [(0, 0), (2, 0)], (1, 0), True),
        ('segment_endpoint_duplicates', [(0, 0), (2, 0), (2, 0)], (2, 0), True),
        ('segment_extension', [(0, 0), (2, 0)], (3, 0), False),
        ('triangle_interior', [(0, 0), (4, 0), (0, 4)], (1, 1), True),
        ('triangle_exact_boundary', [(0, 0), (2, 0), (0, 2)], (1, 1), True),
        ('one_ulp_outside', [(0, 0), (2, 0), (0, 2)], (1, math.nextafter(1., math.inf)), False),
        ('distance_1000_alone_is_insufficient', [(500, 0)], (-10, 0), False),
        ('no_positive', [], (0, 0), False),
        ('nonfinite', [(0, 0)], (math.nan, 0), False),
    ]
    for name, positive, q, expected in cases:
        result = receipt_certificate([(0, 0)], positive, q)
        if result['certified'] != expected or expected and not _check_witness(result, positive, q):
            raise RuntimeError('Failed case: '+name)
        tests.append(dict(name=name, expected_certified=expected, passed=True))
    # A legal negative response is missed by all 24 sampled accepted headings.
    # The true unit direction is proportional to (1,-300), and R=1000.
    positive, q = [(500, 1), (500, -1)], (500, 2)
    accepted = []
    for k in range(24):
        n = (math.cos(k*math.pi/12), math.sin(k*math.pi/12))
        if all(sum(a*b for a, b in zip(p, n)) >= 0 for p in positive):
            accepted.append(sum(a*b for a, b in zip(q, n)) >= 0)
    actual_n = (1, -300)
    positive_dots = [sum(a*b for a, b in zip(p, actual_n)) for p in positive]
    negative_dot = sum(a*b for a, b in zip(q, actual_n))
    if not accepted or not all(accepted) or min(positive_dots) < 0 or negative_dot >= 0:
        raise RuntimeError('Invalid finite-heading counterexample')
    if receipt_certificate([(0, 0)], positive, q)['certified']:
        raise RuntimeError('False receipt guarantee on the finite-heading counterexample')
    counterexample = dict(source=[0, 0], radius=1000, positives=positive, candidate=q,
                          actual_heading_proportional_to=actual_n, positive_dot_products=positive_dots,
                          candidate_dot_product=negative_dot, sampled_accepted_headings=len(accepted),
                          sampled_receive_fraction=1., actual_response='no_signal', certified=False)
    rng = random.Random(57000)
    physical_checks, certified_checks = 0, 0
    for case in range(40):
        g = (rng.randint(-1000, 1000), rng.randint(-1000, 1000))
        radius = rng.randint(1000, 1500)
        n = (rng.randint(-7, 7), rng.randint(-7, 7))
        if n == (0, 0):
            n = (1, 0)
        directional = bool(case % 2)
        positive = []
        while len(positive) < 4:
            p = (g[0]+rng.randint(-radius, radius), g[1]+rng.randint(-radius, radius))
            d = tuple(a-b for a, b in zip(p, g))
            if sum(x*x for x in d) <= radius**2 and (not directional or sum(a*b for a,b in zip(d,n)) >= 0):
                positive.append(p)
        queries = list(positive)
        queries.extend(tuple((positive[i][k]+positive[(i+1)%4][k])/2 for k in (0,1)) for i in range(4))
        queries.extend((g[0]+rng.randint(-radius, radius), g[1]+rng.randint(-radius, radius)) for _ in range(8))
        for q in queries:
            result = receipt_certificate([g], positive, q)
            physical_checks += 1
            if result['certified']:
                certified_checks += 1
                d = tuple(a-b for a, b in zip(_point(q), _point(g)))
                if not _check_witness(result, positive, q):
                    raise RuntimeError('Invalid exact witness')
                if sum(x*x for x in d) > radius**2 or directional and sum(a*b for a,b in zip(d,n)) < 0:
                    raise RuntimeError('False guarantee for a legal test source')
    return dict(model=VERSION, passed=True, deterministic_cases=tests,
                finite_heading_counterexample=counterexample,
                physical_property_checks=physical_checks, certified_property_checks=certified_checks,
                seed=57000, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                test_boundary='Local algebra/physical consistency tests; no simulator or network calls.')


if __name__ == '__main__':
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
