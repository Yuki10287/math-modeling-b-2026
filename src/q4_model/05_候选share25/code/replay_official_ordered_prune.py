"""Restricted offline deletion replay of supplied official logs; no HTTP calls.

Deletion decisions use only the original belief and already observed positive/
negative positions. Logged failures are read only afterwards to verify every
removed attempt was a failure and the original success is retained. No new
measurement positions, substituted feedback, or hidden source truth is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from ordered_optical_prune_experiment import (
    HERE, PROJECT, constrained_replay, retime, hashes, write, demand,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    snapshot[Path(__file__).relative_to(PROJECT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(out/'manifest.json', dict(code_hashes=snapshot, zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        local_only=True, official_simulator_contacted=False, no_new_measurement_feedback=True,
        rule='Only remove original ordered optical clearances whose radius-20 disk is proved disjoint from a valid source region.',
        premise='Static scene and position/channel feedback; clear failures do not change radar channel or environment.',
        rounding='Preserve reported server timestamps separately; compute physical charges without response rounding.'))
    rows = []
    with zipfile.ZipFile(args.zip) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.jsonl'))
        demand(len(names) == 4, 'expected four supplied official batch-6 logs')
        for index, name in enumerate(names, 1):
            raw = archive.read(name)
            records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
            trace_raw = archive.read(name+'.beliefs.json')
            trace = json.loads(trace_raw.decode('utf-8-sig'))
            starts = [r for r in records if r.get('type') == 'session_start']
            demand(len(starts) == 1 and starts[0]['variant'] == 'share25', 'wrong baseline variant')
            demand(all(snapshot['src/'+key] == value for key,value in starts[0]['source_sha256'].items()), 'frozen source mismatch')
            events = []
            for record in records:
                if record.get('path') not in ('/clear','/measure'):
                    continue
                request, response = record['request'], record['response']
                demand(response['accepted'] is True, 'rejected original request')
                q = request['position']
                events.append(dict(action=record['path'][1:],position=[q['x'],q['y']],
                    channel=request['channel'],time_s=response['virtual_time_s'],
                    **{key:value for key,value in response.items() if key in ('measure_result','clear_result','svd_deg')}))
            actual = [r for r in trace if r['phase'] == 'actual_action']
            demand(len(actual) == len(events), 'trace and server event counts differ')
            for i, (t,e) in enumerate(zip(actual,events), 1):
                demand(t['event'] == i and t['action'] == e['action'] and t['position'] == e['position']
                       and t['channel'] == e['channel'] and t['result'] == e[e['action']+'_result'], 'trace not bound to server response')
            # Independent per-action time reconstruction. Server responses
            # round timestamps; do not silently claim byte equality.
            exact_events = retime(events)
            previous_server, previous_model, max_step_error = 0., 0., 0.
            for server, model in zip(events, exact_events):
                error = abs((server['time_s']-previous_server)-(model['time_s']-previous_model))
                max_step_error = max(max_step_error,error)
                demand(error < 1.1e-6, 'official per-action costs differ from model')
                previous_server, previous_model = server['time_s'], model['time_s']
            max_total_error = max(abs(e['time_s']-m['time_s']) for e,m in zip(events,exact_events))
            case = dict(events=exact_events,trace=trace)
            replay = constrained_replay(case)
            replay.update(input_log_sha256=hashlib.sha256(raw).hexdigest(),
                          input_trace_sha256=hashlib.sha256(trace_raw).hexdigest(),
                          original_server_total_s=events[-1]['time_s'],
                          original_server_time_s=[e['time_s'] for e in events],
                          max_single_action_server_rounding_error_s=max_step_error,
                          max_total_server_rounding_error_s=max_total_error,
                          no_new_measurement_feedback=True)
            # Sanitized input permits external proof binding without IDs,
            # request tokens, wall timestamps, or any hidden source data.
            write(out/f'case-{index:02d}-sanitized-input.json', case)
            write(out/f'case-{index:02d}-restricted-replay.json',replay)
            row = dict(case=index, **{k:v for k,v in replay.items() if k not in ('events','proofs','original_server_time_s')})
            row['reduction_percent'] = 100*replay['saved_s']/replay['original_s']
            rows.append(row)
            print(f'case {index}: {replay["original_s"]:.6f} -> {replay["candidate_s"]:.6f}; '
                  f'saved {replay["saved_s"]:.6f}s; removed {replay["removed_actual"]}', flush=True)
    a,b = float(np.mean([r['original_s'] for r in rows])),float(np.mean([r['candidate_s'] for r in rows]))
    summary = dict(passed=True, rows=rows, original_mean_s=a, candidate_mean_s=b,
        reduction_percent=100*(a-b)/a, removed_actual=sum(r['removed_actual'] for r in rows),
        no_new_measurement_feedback=True, no_official_run=True,
        interpretation='Exact physical-cost subtraction for an ordered subsequence of supplied trajectories, conditional on stated static feedback invariants; not a newly observed official score.')
    write(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
