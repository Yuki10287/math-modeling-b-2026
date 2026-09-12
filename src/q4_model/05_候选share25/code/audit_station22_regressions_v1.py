"""Read-only decomposition of two recorded station22 regression pairs.

Contraction diagnostics use only P and observations already present at that
trace position. Newly planned optical points are never executed or assigned
counterfactual feedback/time. Truth is used only for a posteriori containment.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = next(p for p in HERE.parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from shared import core
from localization import optical_plan
from negative_region_boxes_v1 import contract_position
from test_negative_region_truth_v1 import exact, independent_contains

RESULTS = HERE.parent/'results'
PAIRS = (('station22_share_holdout39000_v1', '39002-uniform-smooth'),
         ('station22_share_pressure42000_v1', '42000-uniform-extreme'))


def shape(P):
    P = np.asarray(P)
    length, (i, j) = core.diameter(P)
    direction = (P[j]-P[i])/length if length > 1e-12 else np.array([1., 0.])
    transverse = np.array([-direction[1], direction[0]])
    width = float(np.ptp(P@transverse))
    area = float(abs(np.sum(P[:, 0]*np.roll(P[:, 1], -1)-P[:, 1]*np.roll(P[:, 0], -1)))/2)
    return dict(diameter_m=length, transverse_width_m=width,
        aspect_ratio=length/max(width, 1e-12), polygon_area_m2=area)


def summarize(path):
    data = json.loads(path.read_text())
    reasons, services = defaultdict(lambda:defaultdict(float)), defaultdict(lambda:defaultdict(float))
    discovery, scan_batches, actions = [], [], []
    previous, tuned, previous_reason = np.zeros(2), 1, None
    actual = [r for r in data['trace'] if r['phase'] == 'actual_action']
    assert len(actual) == len(data['events'])
    for event, row in zip(data['events'], actual):
        c, q, reason = event['channel'], np.array(event['position']), row['reason']
        move = float(np.linalg.norm(q-previous))/5
        if event['action'] == 'measure':
            components = dict(move=move, measure=5., switch=float(tuned != c), clear_success=0., clear_fail=0.)
            tuned = c
        else:
            success = event['clear_result'] == 'success'
            components = dict(move=move, measure=0., switch=0., clear_success=5. if success else 0., clear_fail=0. if success else 3.)
        for key, value in components.items():
            reasons[reason][key] += value
            if reason != 'scan':
                services[c][key] += value
        if reason == 'scan':
            if previous_reason != 'scan' or not np.array_equal(q, previous):
                scan_batches.append(dict(position=q.tolist(), first_event=row['event'],
                    first_time_s=event['time_s'], arrival_move_s=move, measurements=0,
                    discovered_channels=[]))
            scan_batches[-1]['measurements'] += 1
        if event['action'] == 'measure' and event['measure_result'] != 'no_signal' and c not in discovery:
            discovery.append(c)
            assert reason == 'scan'
            scan_batches[-1]['discovered_channels'].append(c)
        actions.append(dict(channel=c, reason=reason, event=row['event'],
            total_s=sum(components.values()), **components))
        previous, previous_reason = q, reason
    accounted = sum(sum(v.values()) for v in reasons.values())
    assert abs(accounted-data['summary']['total_s']) < 1e-6
    # Actual optical episodes and preceding belief observations are recoverable
    # directly from trace order without resynthesizing any measurements.
    latest, position, episodes, decisions = {}, np.zeros(2), [], []
    truth = {r['channel']:r for r in data['sources']}
    for trace_index, row in enumerate(data['trace']):
        if row['phase'] == 'belief':
            latest[row['channel']] = row
        elif row['phase'] == 'actual_action':
            position = np.array(row['position'])
        elif row['phase'] in ('optical_plan', 'decision'):
            c = row['channel']
            belief = latest[c]
            P = np.asarray(row.get('polygon', belief['polygon']))
            assert np.array_equal(P, np.array(belief['polygon']))
            record = dict(channel=c, trace_index=trace_index, position=position.tolist(),
                phase=row['phase'], shape=shape(P), positives=len(belief['positives']), negatives=len(belief['negatives']))
            if row['phase'] == 'optical_plan':
                following = []
                for later in data['trace'][trace_index+1:]:
                    if later['phase'] != 'actual_action':
                        break
                    if later['reason'] != 'optical_cover' or later['channel'] != c:
                        break
                    following.append(actions[later['event']-1])
                record.update(planned_points=len(row['path']), predicted_score_s=row['score'],
                    planned_worst_s=row['worst_s'], actual_episode_s=sum(a['total_s'] for a in following),
                    actual_attempts=len(following), actual_failures=sum(a['clear_fail'] > 0 for a in following))
                episodes.append(record)
            else:
                record.update(predicted_s=row['predicted_s'], old_optical_score_s=row['optical_s'], signal_mass=row['signal_mass'])
                decisions.append(record)
            # Diagnose both pre-optical and actual active-view-decision states.
            new_P, info = contract_position(P, belief['positives'], belief['negatives'])
            true_kept = independent_contains(exact(new_P), exact([truth[c]['position']])[0])
            assert true_kept
            old_plan, new_plan = optical_plan(P, position), optical_plan(new_P, position)
            record['contraction'] = dict(applied=info['applied'], reason=info['reason'],
                true_source_preserved_exactly=true_kept, before=shape(P), after=shape(new_P),
                statistics=info['statistics'], certificate=info['certificate'],
                old_optical_proxy_s=old_plan['score'], new_optical_proxy_s=new_plan['score'],
                old_optical_points=len(old_plan['path']), new_optical_points=len(new_plan['path']),
                role='existing-observation geometric plan diagnostic only; no new feedback or measured runtime')
    return dict(input_file=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total_s=data['summary']['total_s'], counts=data['summary']['counts'],
        source_count=data['summary']['source_count'], reasons={k:dict(v,total_s=sum(v.values())) for k,v in reasons.items()},
        service_by_channel={str(k):dict(v,total_s=sum(v.values())) for k,v in services.items()},
        scan_stops=len(scan_batches), scan_batches=scan_batches, discovery_order=discovery,
        optical_episodes=episodes, active_measure_decisions=decisions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__), HERE/'negative_region_boxes_v1.py', HERE/'test_negative_region_truth_v1.py',
               Q4/'localization.py', Q4.parent/'q3_model_v2/geometry.py']
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    pairs = []
    for folder, prefix in PAIRS:
        a, b = (summarize(RESULTS/folder/f'{prefix}-{variant}.json') for variant in ('baseline_share25', 'station22_share'))
        reasons = {key:{part:b['reasons'].get(key, {}).get(part,0)-a['reasons'].get(key,{}).get(part,0)
            for part in ('move','measure','switch','clear_success','clear_fail','total_s')}
            for key in set(a['reasons']) | set(b['reasons'])}
        channels = sorted({int(k) for k in a['service_by_channel']} | {int(k) for k in b['service_by_channel']})
        service_delta = sorted([dict(channel=c, delta_s=b['service_by_channel'].get(str(c),{}).get('total_s',0)-
            a['service_by_channel'].get(str(c),{}).get('total_s',0)) for c in channels], key=lambda r:-r['delta_s'])
        pair = dict(case=prefix, baseline=a, candidate=b, total_delta_s=b['total_s']-a['total_s'],
            reason_differences=reasons, service_channel_differences=service_delta)
        pairs.append(pair)
        print(prefix, 'delta', pair['total_delta_s'], 'scan_delta', reasons['scan']['total_s'],
            'candidate_optical_states', len(b['optical_episodes']),
            'shrunk', sum(x['contraction']['applied'] for x in b['optical_episodes']), flush=True)
    stable = before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert stable
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(dict(source_hashes=before, source_snapshot_stable=stable, pairs=pairs,
            local_read_only_analysis=True, official_contacted=False, new_action_feedback_generated=False,
            attribution_note='Travel is charged to arrival action; source service excludes all scan actions because first tuned scan channel is not its cause.'), stream, indent=2)


if __name__ == '__main__':
    main()
