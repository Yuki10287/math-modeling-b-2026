"""Read saved feedback-cost cases; audit real-prefix prediction premises.

No solver, scoring helper, simulator, network client or source-truth input is
used. Real beliefs are rebuilt with the frozen observation geometry only.
Hypothetical branch values are audited for support/aggregation, not recomputed
as new observations or claimed as physical completion-time guarantees.
"""
import argparse
import ast
from collections import Counter
import copy
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
from localization import Belief
from shared import core


def demand(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, label):
    demand(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b) <= 1e-7,
           f'{label}: {a!r} != {b!r}')


def point(q):
    demand(len(q) == 2 and all(math.isfinite(float(x)) for x in q), 'invalid coordinate')
    return tuple(F(float(x)) for x in q)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def in_positive_hull(positives, q):
    """Independent Caratheodory enumeration, without the producer's hull code."""
    pts, q = list(dict.fromkeys(map(point, positives))), point(q)
    if q in pts:
        return True
    for a, b in itertools.combinations(pts, 2):
        if cross(a, b, q) == 0 and all(min(a[k], b[k]) <= q[k] <= max(a[k], b[k]) for k in (0, 1)):
            return True
    for a, b, c in itertools.combinations(pts, 3):
        area = cross(a, b, c)
        if area:
            weights = (cross(b, c, q)/area, cross(c, a, q)/area, cross(a, b, q)/area)
            if min(weights) >= 0:
                return True
    return False


def verify_receipt(certificate, positives, q):
    """Bind each witness to THIS source's preceding accepted positive ledger."""
    certified = certificate['certified']
    demand(type(certified) is bool, 'receipt flag must be boolean')
    demand(certified == in_positive_hull(positives, q), 'receipt flag differs from exact positive hull')
    if not certified:
        demand(certificate['proof'] is None, 'uncertified receipt unexpectedly has proof')
        return False
    proof = certificate['proof']
    demand(proof['type'] == 'exact_positive_convex_combination', 'unknown receipt proof')
    actual_q = point(q)
    demand(tuple(F(x) for x in proof['q']) == actual_q, 'receipt q differs from candidate q')
    witnesses = proof['witnesses']
    demand(bool(witnesses), 'empty receipt witness')
    weights, locations = [], []
    for witness in witnesses:
        index = witness['positive_index']
        demand(type(index) is int and 0 <= index < len(positives), 'witness index not in actual positive prefix')
        p = tuple(F(x) for x in witness['point'])
        demand(p == point(positives[index]), 'witness location is not its indexed prior positive')
        weight = F(witness['weight'])
        demand(weight > 0, 'nonpositive witness mass')
        weights.append(weight)
        locations.append(p)
    demand(sum(weights) == 1, 'receipt weights do not sum to one')
    demand(all(sum(w*p[k] for w, p in zip(weights, locations)) == actual_q[k] for k in (0, 1)),
           'receipt barycenter differs from actual candidate')
    demand(proof['position_region_used'] is False and proof['radius_and_direction_both_proved'] is True
           and proof['boundary_included'] is True, 'receipt proof semantics changed')
    return True


def forecast_checks(value):
    for name in ('mean_s', 'worst_s', 'score'):
        demand(math.isfinite(value[name]) and value[name] >= 0, 'invalid forecast cost')
    demand(value['mean_s'] <= value['worst_s']+1e-7, 'mean forecast exceeds full tail')
    close(value['score'], .8*value['mean_s']+.2*value['worst_s'], 'clear forecast mixture')
    demand(value['kept_points'] > 0 and value['kept_points']+value['removed_points'] == value['original_points'],
           'forecast point accounting')
    if value['mean_fallback_to_worst']:
        close(value['mean_s'], value['worst_s'], 'empty/uncovered quadrature fallback')


def original_points(belief, start):
    # Reconstruct the frozen ten-point recipe; never call candidate scoring.
    center, r = core.mec(belief.P)
    _, (i, j) = core.diameter(belief.P)
    direction = belief.P[j]-belief.P[i]
    length = np.linalg.norm(direction)
    direction = direction/length if length > 1e-9 else np.array([1., 0.])
    transverse = np.array([-direction[1], direction[0]])
    return [center, (center+start)/2]+[
        center+along*direction+offset*transverse
        for along in (0., -.35*r) for offset in (-100., -40., 40., 100.)]


def work_support(belief, q, prepared, certified):
    gs, scenarios = prepared
    all_weights = [[] for _ in gs]
    hit_weights = [[] for _ in gs]
    total = sum(s[4] for s in scenarios)
    for index, g, heading, radius, weight in scenarios:
        mass = weight/total
        all_weights[index].append(mass)
        if math.dist(q, g) <= radius and (heading is None or sum((q[k]-g[k])*heading[k] for k in (0, 1)) >= 0):
            hit_weights[index].append(mass)
    raw = math.fsum(math.fsum(v) for v in hit_weights)
    weights = [math.fsum(v) for v in (all_weights if certified else hit_weights)]
    return raw, weights


def candidate_checks(value, belief, start, channel, current, prepared):
    q = value['q']
    guarantee = verify_receipt(value['receipt_certificate'], belief.positives, q)
    raw, masses = work_support(belief, q, prepared, guarantee)
    probability = math.fsum(masses)
    close(value['sampled_signal_mass'], raw, 'sampled receipt mass')
    close(value['signal_mass'], probability, 'effective receipt mass')
    demand(value['scenarios'] == len(prepared[1]), 'scenario count differs from real-prefix belief')
    close(value['immediate_s'], math.dist(q, start)/5+5+int(channel != current), 'immediate move/measure/switch')
    negative = value['negative_forecast']
    demand(value['negative_in_risk'] is (not guarantee) and (negative is not None) is (not guarantee),
           'unproved negative branch removed from risk')
    negative_mass = 0. if guarantee else max(0., 1-probability)
    close(value['negative_work_mass'], negative_mass, 'negative work mass')
    branches = value['positive_branches']
    expected_readings = []
    for index, mass in enumerate(masses):
        if mass <= 0:
            continue
        g = prepared[0][index]
        replies = [{'measure_result': 'near'}] if math.dist(g, q) <= 5 else [
            {'measure_result': 'direction', 'svd_deg': round((math.degrees(math.atan2(g[1]-q[1], g[0]-q[0]))+err) % 360, 2) % 360}
            for err in (-1., 0., 1.)]
        expected_readings += [(index, reply, mass/len(replies)) for reply in replies]
    demand(len(branches) == len(expected_readings) and bool(branches), 'positive branch inventory differs')
    for branch, (index, reply, mass) in zip(branches, expected_readings):
        demand(branch['position_index'] == index and branch['feedback'] == reply, 'predictive reading mismatch')
        close(branch['work_mass'], mass, 'positive reading mass')
        forecast_checks(branch)
    mean = math.fsum(b['work_mass']*b['mean_s'] for b in branches)
    tails = [b['worst_s'] for b in branches]
    if negative is not None:
        forecast_checks(negative)
        mean += negative_mass*negative['mean_s']
        tails.append(negative['worst_s'])
    risk = max(tails)
    close(value['expected_tail_s'], mean, 'expected branch mean')
    close(value['risk_tail_s'], risk, 'maximum branch full tail')
    close(value['score'], value['immediate_s']+.8*mean+.2*risk, 'total feedback score')
    return dict(certified=guarantee, zero_mass_negative_in_risk=not guarantee and negative_mass == 0.,
                positive_branches=len(branches))


def prediction_checks(row, belief, start, current):
    demand(row['model'] == 'feedback_cost_v1' and row['mean_weight'] == .8 and row['risk_weight'] == .2,
           'prediction model or coefficients changed')
    forecast_checks(row['current_optical'])
    prepared = belief.scenarios()
    eligible = []
    for q in original_points(belief, start):
        if not prepared[1] or any(math.dist(q, p) <= .1 for p in belief.measured):
            continue
        certified = in_positive_hull(belief.positives, q)
        _, masses = work_support(belief, q, prepared, certified)
        if math.fsum(masses) >= .05:
            eligible.append(q.tolist())
    demand([v['q'] for v in row['candidates']] == eligible, 'candidate order/inventory differs from frozen ten points')
    reports = [candidate_checks(value, belief, start, row['channel'], current, prepared) for value in row['candidates']]
    best = None
    for value in row['candidates']:
        if best is None or value['score'] < best['score']:
            best = value
    selected = best is not None and best['score'] < row['current_optical']['score']
    demand(row['selected'] == ('measure' if selected else 'optical'), 'wrong candidate/optical decision')
    demand(row['selected_q'] == (best['q'] if selected else None), 'wrong minimum candidate or tie policy')
    return dict(candidates=len(reports), certified_receipt=sum(r['certified'] for r in reports),
                zero_mass_negative_in_risk=sum(r['zero_mass_negative_in_risk'] for r in reports),
                positive_branches=sum(r['positive_branches'] for r in reports))


def audit_case(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    events, trace = payload['events'], payload['trace']
    del payload  # Do not inspect sources, hidden totals or simulation settings.
    beliefs, negatives, active = {}, {c: [] for c in range(1, 21)}, Counter()
    count, belief_rows, pending = 0, 0, None
    position, current = np.zeros(2), 1
    optical_channel = None
    reports = []
    for trace_index, row in enumerate(trace):
        phase = row.get('phase')
        if phase == 'feedback_cost_prediction':
            c = row['channel']
            demand(pending is None and c in beliefs and active[c] < 6, 'unexpected/orphan prediction')
            report = prediction_checks(row, beliefs[c], position, current)
            reports.append(dict(trace_index=trace_index, event_prefix=count, channel=c,
                                actual_positive_points=[p.tolist() for p in beliefs[c].positives],
                                selected=row['selected'], **report))
            pending = row
        elif phase == 'actual_action':
            demand(row['event'] == count+1 and count < len(events), 'nonconsecutive or extra action trace')
            event = events[count]
            count += 1
            c, q = event['channel'], np.asarray(event['position'])
            key = 'measure_result' if event['action'] == 'measure' else 'clear_result'
            demand((row['action'], row['channel'], row['position'], row['result']) ==
                   (event['action'], c, event['position'], event[key]), 'actual action differs from independent ledger')
            close(row['move_s'], math.dist(position, q)/5, 'actual movement prefix')
            if pending is not None:
                demand(c == pending['channel'], 'prediction consumed by other channel')
                if pending['selected'] == 'measure':
                    demand(row['reason'] == 'source_measure' and row['position'] == pending['selected_q'],
                           'selected measure not executed')
                else:
                    demand(row['reason'] == 'optical_cover', 'selected optical fallback not executed')
                pending = None
            elif row['reason'] == 'source_measure':
                raise ValueError('source measure lacks its complete prediction')
            elif row['reason'] == 'optical_cover' and optical_channel != c:
                demand(active[c] >= 6, 'new optical service below budget lacks prediction')
            if row['reason'] == 'optical_cover':
                optical_channel = c
            elif optical_channel is not None:
                raise ValueError('optical execution interrupted before success')
            if event['action'] == 'measure':
                current = c
                feedback = {k: event[k] for k in ('measure_result', 'svd_deg') if k in event}
                if event[key] == 'no_signal':
                    negatives[c].append(q.copy())
                    if c in beliefs:
                        beliefs[c].update(q, feedback)
                elif c not in beliefs:
                    beliefs[c] = Belief(q, feedback)
                    beliefs[c].negatives = [p.copy() for p in negatives[c]]
                else:
                    beliefs[c].update(q, feedback)
                if row['reason'] == 'source_measure':
                    active[c] += 1
            elif event[key] == 'success':
                beliefs.pop(c, None)
                optical_channel = None
            elif c in beliefs:
                beliefs[c].failed_clears.append(q.copy())
            position = q.copy()
        elif phase == 'belief':
            b = beliefs[row['channel']]
            demand(b.P.tolist() == row['polygon'] and [p.tolist() for p in b.positives] == row['positives']
                   and [p.tolist() for p in b.negatives] == row['negatives'], 'real belief prefix mismatch')
            belief_rows += 1
    demand(count == len(events) and pending is None and optical_channel is None, 'incomplete audited trace')
    return dict(input=binding(path), passed=True, actions=count, belief_rows=belief_rows,
                prediction_rows=len(reports), candidates=sum(r['candidates'] for r in reports),
                certified_receipt=sum(r['certified_receipt'] for r in reports),
                zero_mass_negative_in_risk=sum(r['zero_mass_negative_in_risk'] for r in reports),
                positive_branches=sum(r['positive_branches'] for r in reports), predictions=reports)


def binding(path):
    return dict(path=path.relative_to(PROJECT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_review():
    model = HERE/'feedback_cost_experiment.py'
    tree = ast.parse(model.read_text(encoding='utf-8-sig'))
    names = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    demand([a.arg for a in names['score_measure'].args.args] ==
           ['belief', 'start', 'q', 'channel', 'current_channel', 'prepared', 'diagnostics'], 'score interface changed')
    demand([a.arg for a in names['forecast_clear'].args.args] == ['belief', 'start', 'plan'], 'forecast interface changed')
    blocked = {'sources', 'true_sources', 'truth', 'arena', 'environment', 'time_s', 'seed', 'request_id'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            demand(n.attr not in blocked, 'hidden/time attribute in model')
        if isinstance(n, ast.Name):
            demand(n.id not in blocked, 'hidden/time name in model')
    return dict(passed=True, model=binding(model), inspected_score_parameters='public belief and coordinates/channel only',
                hidden_truth_names_or_attributes_found=False,
                limitation='Static interface/name inspection and public-prefix binding, not a general formal noninterference proof.')


def witness_challenges():
    positive = [[0., 0.], [2., 0.], [0., 2.]]
    q = [1., 0.]
    valid = dict(certified=True, proof=dict(type='exact_positive_convex_combination', q=['1', '0'],
        witnesses=[dict(positive_index=0, point=['0', '0'], weight='1/2'),
                   dict(positive_index=1, point=['2', '0'], weight='1/2')],
        position_region_used=False, radius_and_direction_both_proved=True, boundary_included=True))
    demand(verify_receipt(valid, positive, q), 'valid boundary witness rejected')
    attacks = []
    v = copy.deepcopy(valid); v['proof']['q'] = ['0', '1']; attacks.append(('different_q', v, positive, q))
    v = copy.deepcopy(valid); v['proof']['witnesses'][1]['positive_index'] = 2; attacks.append(('wrong_actual_positive_index', v, positive, q))
    v = copy.deepcopy(valid); v['proof']['witnesses'][0]['weight'] = '1'; attacks.append(('wrong_weight', v, positive, q))
    attacks.append(('different_source_ledger', copy.deepcopy(valid), [[0., 0.]], q))
    attacks.append(('future_positive_not_yet_accepted', copy.deepcopy(valid), [[0., 0.], [0., 2.]], q))
    for name, cert, premise, candidate in attacks:
        try:
            verify_receipt(cert, premise, candidate)
        except (ValueError, KeyError, IndexError):
            continue
        raise ValueError('forged certificate accepted: '+name)
    return dict(passed=True, valid_exact_boundary=1, forged_rejected=[a[0] for a in attacks])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='completed saved batch directory')
    parser.add_argument('--out', type=Path, help='NEW audit output directory')
    parser.add_argument('--self-test', action='store_true', help='only five forged receipt challenges; no cases')
    args = parser.parse_args()
    challenges = witness_challenges()
    if args.self_test:
        print(json.dumps(challenges))
        return
    demand(args.input is not None and args.out is not None, '--input and --out are required')
    demand(not args.out.exists(), 'refuse to overwrite existing audit')
    manifest_path = args.input/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths = sorted(args.input.glob('*-feedback_cost.json'))
    demand(len(paths) == manifest['paired_conditions'], 'batch candidate files incomplete')
    expected = manifest['code_hashes']
    dependencies = [Q4/'localization.py', Q4/'shared.py', PROJECT/'src/q3_model_v2/geometry.py',
                    HERE/'feedback_cost_experiment.py', HERE/'receive_guarantee_v1.py']
    for dependency in dependencies:
        record = binding(dependency)
        demand(record['sha256'] == expected[record['path']], 'active audited dependency differs from batch snapshot')
    review = source_review()
    rows = [audit_case(path) for path in paths]
    report = dict(passed=True, scope='Read-only audit of saved local predictions; no new execution or official contact.',
        manifest=binding(manifest_path), audit_source=binding(Path(__file__)), dependencies=[binding(p) for p in dependencies],
        static_interface_review=review, receipt_challenges=challenges, rows=rows,
        total={k:sum(r[k] for r in rows) for k in ('actions', 'belief_rows', 'prediction_rows', 'candidates',
                                                'certified_receipt', 'zero_mass_negative_in_risk', 'positive_branches')},
        limitations=['Finite position/type/direction scenarios are the frozen working distribution, not official probabilities.',
                     'Hypothetical optical mean/full-tail values are checked for finite accounting and branch aggregation; this script does not rerun every hypothetical geometric contraction/cover.',
                     'Actual optical deletion/completion proofs and physical source retention belong to the separate complete trajectory audits.',
                     'An unproved receipt retains no_signal in the risk maximum even with zero sampled negative mass; .2 risk weighting is not a worst-case time guarantee.'])
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.out/'summary.json'), passed=True, **report['total']), ensure_ascii=False))


if __name__ == '__main__':
    main()
