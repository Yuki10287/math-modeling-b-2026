"""Bind restricted prune proofs to raw uploaded logs without contacting HTTP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

from audit_ordered_optical_prune_v1 import verify_case, self_tests


def charges(events):
    previous, channel, total, times = (0., 0.), 1, 0., []
    for row in events:
        total += math.dist(previous, row['position'])/5
        if row['action'] == 'measure':
            total += 5+(row['channel'] != channel)
            channel = row['channel']
        else:
            total += 5 if row['clear_result'] == 'success' else 3
        previous = row['position']
        times.append(total)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--replays', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    manifest = json.loads((args.replays/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(args.zip.read_bytes()).hexdigest() == manifest['zip_sha256']
    rows = []
    with zipfile.ZipFile(args.zip) as archive:
        logs = sorted(name for name in archive.namelist() if name.endswith('.jsonl'))
        assert len(logs) == 4
        for index, name in enumerate(logs, 1):
            raw, trace_raw = archive.read(name), archive.read(name+'.beliefs.json')
            records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
            case = json.loads((args.replays/f'case-{index:02}-sanitized-input.json').read_text(encoding='utf-8'))
            replay = json.loads((args.replays/f'case-{index:02}-restricted-replay.json').read_text(encoding='utf-8'))
            assert replay['input_log_sha256'] == hashlib.sha256(raw).hexdigest()
            assert replay['input_trace_sha256'] == hashlib.sha256(trace_raw).hexdigest()
            assert case['trace'] == json.loads(trace_raw.decode('utf-8-sig')), 'sanitized trace differs from raw belief log'
            starts = [r for r in records if r.get('type') == 'session_start']
            assert len(starts) == 1 and starts[0]['variant'] == 'share25'
            assert all(manifest['code_hashes']['src/'+key] == value for key, value in starts[0]['source_sha256'].items())
            events, reported_times = [], []
            for record in records:
                if record.get('path') not in ('/measure', '/clear'):
                    continue
                request, response = record['request'], record['response']
                assert response['accepted'] is True
                event = dict(action=record['path'][1:], channel=request['channel'],
                    position=[request['position']['x'], request['position']['y']])
                event.update({key: response[key] for key in ('measure_result', 'clear_result', 'svd_deg') if key in response})
                events.append(event)
                reported_times.append(response['virtual_time_s'])
            assert [{k: v for k, v in e.items() if k != 'time_s'} for e in case['events']] == events
            assert replay['original_server_time_s'] == reported_times
            assert replay['original_server_total_s'] == reported_times[-1]
            audit = verify_case(case, replay)
            old_times, new_times = charges(events), charges(replay['events'])
            assert len(old_times) == len(case['events']) and len(new_times) == len(replay['events'])
            assert max(abs(a-e['time_s']) for a, e in zip(old_times, case['events'])) < 1e-7
            assert max(abs(a-e['time_s']) for a, e in zip(new_times, replay['events'])) < 1e-7
            assert abs(replay['original_s']-old_times[-1]) < 1e-7
            assert abs(replay['candidate_s']-new_times[-1]) < 1e-7
            assert abs(replay['saved_s']-(old_times[-1]-new_times[-1])) < 1e-7
            assert new_times[-1] <= old_times[-1]+1e-7
            rounding = max(abs(a-b) for a, b in zip(old_times, reported_times))
            rows.append(dict(case=index, audit=audit, original_s=old_times[-1], candidate_s=new_times[-1],
                saved_s=old_times[-1]-new_times[-1], max_server_rounding_error_s=rounding,
                raw_log_bound=True, all_geometric_premises_observed_previously=True))
    result = dict(passed=True, rows=rows, self_tests=self_tests(),
        plans=sum(r['audit']['plans'] for r in rows), archive_sha256=manifest['zip_sha256'],
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_auditor_sha256=hashlib.sha256(Path(__file__).with_name('audit_ordered_optical_prune_v1.py').read_bytes()).hexdigest(),
        no_official_simulator_contacted=True,
        interpretation='Restricted physical-cost replay under retained original feedback; not a newly observed official score.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, cases=len(rows), plans=result['plans'],
                         removed_events=sum(r['audit']['removed_events'] for r in rows)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
