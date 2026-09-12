"""Read-only fee and event-timing attribution of the saved 53300 extreme pair."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def describe(case):
    actions = [r for r in case['trace'] if r['phase']=='actual_action']
    assert len(actions)==len(case['events'])
    state_channel,position = 1,[0.,0.]
    discovered,discoveries,by_reason = set(),[],{}
    source_views,optical_starts,successes = [],[],[]
    per_channel = {}
    for event,row in zip(case['events'],actions):
        reason,c,q = row['reason'],event['channel'],event['position']
        assert row['action']==event['action'] and row['position']==q and row['channel']==c
        record = by_reason.setdefault(reason,dict(actions=0,move_s=0.,measure_s=0.,switch_s=0.,clear_success_s=0.,clear_fail_s=0.))
        record['actions'] += 1
        record['move_s'] += math.dist(q,position)/5
        if event['action']=='measure':
            record['measure_s'] += 5
            record['switch_s'] += int(c!=state_channel)
            state_channel = c
            if event['measure_result']!='no_signal' and c not in discovered:
                discovered.add(c)
                discoveries.append(dict(count=len(discovered),channel=c,event=row['event'],time_s=event['time_s'],reason=reason,position=q))
                per_channel.setdefault(c,{})['discovery_s'] = event['time_s']
            if reason=='source_measure':
                source_views.append(dict(channel=c,event=row['event'],time_s=event['time_s'],result=event['measure_result']))
                per_channel.setdefault(c,{}).setdefault('dedicated_measure_s',[]).append(event['time_s'])
        elif event['clear_result']=='success':
            record['clear_success_s'] += 5
            successes.append(dict(channel=c,event=row['event'],time_s=event['time_s'],reason=reason,position=q))
            per_channel.setdefault(c,{})['cleared_s'] = event['time_s']
        else:
            record['clear_fail_s'] += 3
        position = q
    latest = 0
    for row in case['trace']:
        if row['phase']=='actual_action':
            latest = row['event']
        if row['phase']=='optical_plan':
            start = case['events'][latest-1]['time_s'] if latest else 0.
            optical_starts.append(dict(channel=row['channel'],before_event=latest+1,start_s=start,path_points=len(row['path'])))
            per_channel.setdefault(row['channel'],{})['optical_start_s'] = start
    for r in by_reason.values():
        r['total_s'] = sum(v for k,v in r.items() if k!='actions')
    assert abs(sum(r['total_s'] for r in by_reason.values())-case['summary']['total_s'])<1e-6
    all_found = discoveries[-1]
    scan_actions = [r for r in actions if r['reason']=='scan']
    scan_positions = {tuple(r['position']) for r in scan_actions}
    return dict(total_s=case['summary']['total_s'],source_count=case['summary']['source_count'],
        certificate_basis=case['result']['certificate']['basis'],time_parts_s=case['summary']['time_parts_s'],
        by_reason=by_reason,discovery_timeline=discoveries,all_sources_discovered=all_found,
        time_after_all_discovered_s=case['summary']['total_s']-all_found['time_s'],
        distinct_actual_scan_stations_including_origin=len(scan_positions),actual_scan_measurements=len(scan_actions),
        scan_task_choices=sum(r['phase']=='task_choice' and r['task']=='scan' for r in case['trace']),
        scan_after_all_discovered=sum(r['event']>all_found['event'] for r in scan_actions),
        source_measure_timeline=source_views,optical_start_timeline=optical_starts,success_timeline=successes,
        per_channel=per_channel)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    loaded,rows,hashes={},{},{}
    for variant in ('share25','station22','station22_prune'):
        path=args.batch/f'53300-omni_heavy-extreme-{variant}.json'
        raw=path.read_bytes();loaded[variant]=json.loads(raw)
        hashes[path.name]=hashlib.sha256(raw).hexdigest()
        rows[variant]=describe(loaded[variant])
    assert loaded['station22']['events']==loaded['station22_prune']['events']
    before,after=rows['share25'],rows['station22_prune']
    result=dict(passed=True,input_sha256=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,fee_difference_s={k:after['time_parts_s'][k]-before['time_parts_s'][k] for k in before['time_parts_s']},
        station22_and_prune_actual_events_identical=True,no_new_solver_runs=True,
        interpretation='Observed decomposition and event chronology; not a causal ablation or hidden-count rule.')
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(fees=result['fee_difference_s'],rows={k:{n:v for n,v in r.items() if n in (
        'total_s','all_sources_discovered','time_after_all_discovered_s','distinct_actual_scan_stations_including_origin',
        'actual_scan_measurements','scan_task_choices','scan_after_all_discovered','by_reason','optical_start_timeline')}
        for k,r in rows.items()}),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
