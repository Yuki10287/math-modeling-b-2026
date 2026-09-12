"""Decompose saved Q3 official actions without contacting a simulator.

Only public accepted actions and saved plan markers are used.  All movement
and action-fee buckets are disjoint; geometry diagnostics are hindsight only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from study_runtime import resolve


def close(a, b, eps=1e-7):
    return math.dist(a, b) <= eps


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker(trace):
    phase = trace['phase']
    if phase not in ('local_action', 'shared_bearing', 'opportunity_search', 'shared_clear'):
        return None
    action = trace['action'] if phase == 'local_action' else ('clear' if phase == 'shared_clear' else 'measure')
    return action, trace['channel'], trace['position']


def matches(event, item):
    return event['action'] == item[0] and event['channel'] == item[1] and close(event['position'], item[2])


def verify_saved_actions(case, raw):
    assert sha(raw) == case['input_sha256']
    records = [json.loads(line) for line in raw.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    records = [r for r in records if r.get('path') in ('/measure', '/clear')]
    assert len(records) == len(case['actions'])
    for event, record in zip(case['actions'], records):
        req, reply = record['request'], record['response']
        assert reply['accepted'] is True
        assert matches(event, (record['path'][1:], req['channel'], [req['position']['x'], req['position']['y']]))
        result_key = 'measure_result' if event['action'] == 'measure' else 'clear_result'
        assert event[result_key] == reply[result_key]
        assert abs(event['time_s'] - reply['virtual_time_s']) < 1e-8
        if event.get('measure_result') == 'direction':
            assert event['svd_deg'] == reply['svd_deg']


def task_blocks(case, traces):
    actions = case['actions']
    starts = [i for i, t in enumerate(traces) if t['phase'] == 'plan']
    blocks, cursor, discovered = [], 0, set()
    for block_id, begin in enumerate(starts):
        end = starts[block_id + 1] if block_id + 1 < len(starts) else len(traces)
        plan = traces[begin]
        task = plan['route'][0]
        lo = cursor
        expected_position = actions[cursor]['from_position'] if cursor < len(actions) else actions[-1]['position']
        assert close(plan['position'], expected_position), (case['case'], block_id, 'plan position')
        if task['kind'] == 'scan':
            while cursor < len(actions) and actions[cursor]['tag'] == 'scan' and close(actions[cursor]['position'], task['position']):
                cursor += 1
            assert cursor > lo, (case['case'], block_id, 'empty scan')
        for trace in traces[begin + 1:end]:
            item = marker(trace)
            if item is None:
                continue
            assert cursor < len(actions) and matches(actions[cursor], item), (case['case'], block_id, cursor, item)
            cursor += 1
        assert cursor > lo
        events = actions[lo:cursor]
        first_discoveries = []
        for e in events:
            if e.get('measure_result') in ('direction', 'near') and e['channel'] not in discovered:
                discovered.add(e['channel'])
                first_discoveries.append(dict(channel=e['channel'], action_index=e['index'], tag=e['tag']))
        blocks.append(dict(
            block_id=block_id, iteration=plan['iteration'], kind=task['kind'], key=task['key'],
            planned_position=task['position'], from_position=plan['position'],
            entry=events[0]['position'], exit=events[-1]['position'],
            first_action_index=lo, last_action_index=cursor-1, entry_movement_m=events[0]['movement_m'],
            internal_distance_m=sum(e['movement_m'] for e in events[1:]),
            distance_m=sum(e['movement_m'] for e in events),
            fee_s=sum(e['fee_s'] for e in events), duration_s=sum(e['duration_s'] for e in events),
            first_discoveries=first_discoveries,
            cleared_channels=[e['channel'] for e in events if e.get('clear_result') == 'success'],
            actions=events))
    assert cursor == len(actions), (case['case'], cursor, len(actions))
    assert set(discovered) == {e['channel'] for e in actions if e.get('clear_result') == 'success'}
    return blocks


def audit(case, blocks):
    movement = Counter()
    fees = Counter()
    action_counts = Counter()
    serviced = set()
    source_detours, scans = [], []
    for block in blocks:
        kind = block['kind']
        first_service = kind == 'source' and block['key'] not in serviced
        if kind == 'source':
            serviced.add(block['key'])
        for j, e in enumerate(block['actions']):
            if kind == 'scan':
                movement_bucket = 'scan_entry' if j == 0 else 'scan_inside'
            else:
                movement_bucket = ('source_first_service_entry' if first_service else 'source_revisit_entry') if j == 0 else 'source_inside_service'
            movement[movement_bucket] += e['movement_m'] / 5
            if e['action'] == 'measure':
                location = 'stationary' if e['movement_m'] < 1e-7 else 'moving'
                fee_bucket = f"{e['tag']}_{location}_measure"
                fees[fee_bucket] += 5
                fees[f"{e['tag']}_{location}_switch"] += e['switch']
                action_counts[fee_bucket] += 1
            else:
                outcome = 'success' if e['clear_result'] == 'success' else 'failure'
                fees[f'clear_{outcome}'] += e['fee_s']
                action_counts[f'clear_{outcome}'] += 1
        if kind == 'source':
            direct = math.dist(block['from_position'], block['exit'])
            assert block['distance_m'] + 1e-6 >= direct
            source_detours.append(dict(block_id=block['block_id'], channel=block['key'],
                distance_m=block['distance_m'], direct_start_to_exit_m=direct,
                excess_over_direct_s=(block['distance_m']-direct)/5,
                actions=len(block['actions']), local_measures=sum(e['tag']=='local_measure' for e in block['actions']),
                zero_movement_local_measures=sum(e['tag']=='local_measure' and e['movement_m']<1e-7 for e in block['actions'])))
        else:
            scans.append(dict(block_id=block['block_id'], first_discoveries=len(block['first_discoveries']),
                entry_movement_s=block['entry_movement_m']/5, fee_s=block['fee_s'],
                total_s=block['duration_s'], actions=len(block['actions']),
                after_last_clear=block['actions'][0]['start_s'] >= case['last_clear_s']-1e-5))
    reconstructed = sum(movement.values()) + sum(fees.values())
    assert abs(reconstructed-case['total_s']) < 1e-4
    assert abs(sum(movement.values())-case['costs_s']['movement']) < 1e-6
    assert sum(action_counts.values()) == len(case['actions'])
    tag_groups = defaultdict(list)
    for e in case['actions']:
        tag_groups[e['tag']].append(e)
    by_tag = {tag:dict(actions=len(es), movement_s=sum(e['movement_m']/5 for e in es),
                         fee_s=sum(e['fee_s'] for e in es)) for tag, es in tag_groups.items()}
    return dict(case=case['case'], total_s=case['total_s'], cleared=case['cleared'],
        task_count=len(blocks), source_task_count=sum(b['kind']=='source' for b in blocks),
        scan_task_count=len(scans), movement_s=dict(movement), fee_s=dict(fees),
        action_counts=dict(action_counts), reconstructed_total_s=reconstructed,
        by_tag=by_tag, source_detours=source_detours, scans=scans, task_blocks=blocks)


def aggregate(cases):
    n = len(cases)
    total = sum(c['total_s'] for c in cases)
    movements, fees, counts = Counter(), Counter(), Counter()
    for case in cases:
        movements.update(case['movement_s']); fees.update(case['fee_s']); counts.update(case['action_counts'])
    def entries(values):
        return {k:dict(total_s=v, mean_per_run_s=v/n, share_total_pct=100*v/total) for k,v in values.items()}
    source_detours = [s for c in cases for s in c['source_detours']]
    scans = [s for c in cases for s in c['scans']]
    stationary = sum(v for k,v in counts.items() if 'stationary' in k)
    stationary_cost = sum(v for k,v in fees.items() if 'stationary' in k)
    return dict(cases=n, cleared=sum(c['cleared'] for c in cases), mean_total_s=total/n,
        movement=entries(movements), fees=entries(fees), action_counts=dict(counts),
        task_counts=dict(source=len(source_detours),scan=len(scans)),
        zero_movement_measurements=dict(count=stationary, mean_cost_s=stationary_cost/n,
            share_total_pct=100*stationary_cost/total),
        source_path_excess_over_direct=dict(mean_s=sum(s['excess_over_direct_s'] for s in source_detours)/n,
            max_single_task_s=max(s['excess_over_direct_s'] for s in source_detours),
            interpretation='Hindsight geometric slack only; a direct chord need not obtain required bearings or clear a source.'),
        scans_without_discovery=dict(count=sum(s['first_discoveries']==0 for s in scans),
            mean_time_s=sum(s['total_s'] for s in scans if s['first_discoveries']==0)/n,
            mean_movement_s=sum(s['entry_movement_s'] for s in scans if s['first_discoveries']==0)/n,
            interpretation='No discovery is not no value: negative measurements certify absence.'),
        audit_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'local_data/official_runs/q3/received-before-reorganization/received-20260911')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = resolve('src/q3_model_v2/results/official_practice_20260911_analysis.json')
    saved = json.loads(source.read_text(encoding='utf-8'))
    cases, inputs = [], []
    for case in saved['cases']:
        raw = args.input_dir/(case['case']+'.jsonl')
        trace = Path(str(raw)+'.beliefs.json')
        verify_saved_actions(case, raw)
        traces = json.loads(trace.read_text(encoding='utf-8-sig'))
        cases.append(audit(case, task_blocks(case,traces)))
        inputs.append(dict(case=case['case'], accepted_log_sha256=sha(raw), trace_sha256=sha(trace)))
    result = dict(analysis_kind='read_only_disjoint_time_and_task_audit',
        official_simulator_contacted=False, hidden_source_truth_used=False,
        source_analysis_sha256=sha(source), analyzer_sha256=sha(Path(__file__)), inputs=inputs,
        definitions={
            'movement':'Each leg charged exactly once, by destination task and first versus later action.',
            'entry':'Movement from previous task exit to first actual executed action; not source-center distance.',
            'source_inside_service':'All later legs in one chosen-source task, including optical moves and any sharing.',
            'fees':'Each 5 second measure, 1 second switch, and 5/3 second clear fee assigned exactly once.',
            'blocks':'Exact saved execution groups between successive plan traces; all actions consumed and matched.'},
        limitations=[
            'First-service entry often doubles as a bearing-design move; it is not pure travel to a known target.',
            'Zero-movement and negative measurements can be necessary for localization or completion evidence.',
            'Task endpoints and internal observation paths are hindsight information, not an online decision rule.',
            'Only five user-supplied completed practices; no official source positions or alternative-point feedback.'],
        aggregate=aggregate(cases), cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['aggregate'],ensure_ascii=False))


if __name__ == '__main__':
    main()
