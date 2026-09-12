"""Replay one saved Q4 shared session offline, including documented route ties.

The archive, member hashes and actions must match the supplied analysis. No
simulator connection is opened, and existing output files are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

from analyze_official_archive import prepare_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay_case(archive_path, analysis_path, case_name):
    # The existing replay validators use assertions; never silently disable them.
    require(__debug__, 'Replay validation cannot run with Python optimization enabled.')
    report = json.loads(analysis_path.read_text(encoding='utf-8-sig'))
    archive_data = archive_path.read_bytes()
    require(digest(archive_data) == report['archive_sha256'],
            'ZIP hash does not match the supplied analysis.')
    matches = [case for case in report['cases'] if case['case'] == case_name]
    require(len(matches) == 1, 'Analysis must contain exactly one matching case.')
    case = matches[0]
    require(case['variant'] == 'shared', 'This replay supports the shared policy only.')

    # Reading the hashed bytes also avoids reopening a changed archive by path.
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(archive_data)) as archive:
        names = archive.namelist()
        members = [name for name in names
                   if name.endswith('.jsonl') and PurePosixPath(name).stem == case_name]
        require(len(members) == 1, 'ZIP must contain exactly one matching session log.')
        member = members[0]
        trace_member = member + '.beliefs.json'
        require(names.count(trace_member) == 1, 'ZIP must contain one matching belief trace.')
        raw = archive.read(member)
        trace_raw = archive.read(trace_member)
    require(digest(raw) == case['input_sha256'], 'Session member hash differs from analysis.')
    require(digest(trace_raw) == case['trace_sha256'], 'Belief trace hash differs from analysis.')
    rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    saved_trace = json.loads(trace_raw.decode('utf-8-sig'))
    starts = [row for row in rows if row.get('type') == 'session_start']
    summaries = [row for row in rows if row.get('type') == 'session_summary']
    require(len(starts) == len(summaries) == 1, 'Expected one session start and summary.')
    start, summary = starts[0], summaries[0]
    require(start['problem'] == summary['problem'] == 4, 'Expected a Q4 session.')
    require(start['variant'] == summary['variant'] == 'shared', 'Expected the shared policy.')
    require(summary['complete'] and summary['exit_accepted'], 'Session was not completed.')
    require(not any(row.get('type') == 'client_error' for row in rows), 'Session has a client error.')
    calls = [row for row in rows if row.get('path')]
    require(len(calls) >= 2 and calls[0]['path'] == '/enter' and calls[-1]['path'] == '/exit',
            'Session enter/exit boundaries are missing.')
    require(all(row['response']['accepted'] is True for row in calls), 'Session has a rejected request.')
    require(len({row['request']['request_id'] for row in calls}) == len(calls),
            'Session contains duplicate request identifiers.')
    action_calls = calls[1:-1]
    saved_actions = case['actions']
    require(len(action_calls) == len(saved_actions), 'Action count differs from analysis.')
    markers = [row for row in saved_trace if row['phase'] == 'actual_action']
    require(len(markers) == len(saved_actions), 'Trace action count differs from analysis.')

    _, paths, audit = prepare_audit()
    hashes = summary['source_sha256']
    require(hashes == start['source_sha256'], 'Session source fingerprints disagree.')
    require(len(hashes) == case['source_files_checked'], 'Source fingerprint count differs from analysis.')
    for name, expected in hashes.items():
        require(digest(paths['src/' + name].read_bytes()) == expected,
                'Current source differs from the recorded policy: ' + name)
    audit.assert_tree_close(case['total_s'], summary['total_virtual_s'])
    require(case['cleared'] == summary['successful_clear_count'], 'Clear count differs from analysis.')
    actions = []
    for number, (action, call, marker) in enumerate(zip(saved_actions, action_calls, markers), 1):
        request, response = call['request'], call['response']
        kind = call['path'][1:]
        require(kind in ('measure', 'clear'), 'Unexpected action type.')
        require(action['index'] == marker['event'] == number, 'Action numbering differs.')
        require(action['action'] == marker['action'] == kind, 'Action type differs from analysis or trace.')
        require(action['channel'] == marker['channel'] == request['channel'],
                'Action channel differs from analysis or trace.')
        position = [request['position']['x'], request['position']['y']]
        audit.assert_tree_close(action['position'], position)
        audit.assert_tree_close(marker['position'], position)
        audit.assert_tree_close(action['time_s'], response['virtual_time_s'])
        key = 'measure_result' if kind == 'measure' else 'clear_result'
        require(action[key] == marker['result'] == response[key], 'Feedback differs from analysis or trace.')
        require(action['reason'] == marker['reason'], 'Action purpose differs from the trace.')
        actions.append(dict(action, response=response))

    result, trace, details = audit.replay_logged_actions(actions)
    audit.assert_tree_close(trace, saved_trace)
    audit.assert_tree_close(json.loads(json.dumps(result)), {key: summary[key] for key in result})
    details.update(trace_and_certificate_match=True, official_simulator_contacted=False)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path, help='Saved official session archive; read locally only.')
    parser.add_argument('--analysis', required=True, type=Path, help='Analysis containing archive/member hashes and actions.')
    parser.add_argument('--output', required=True, type=Path, help='New replay report; must not already exist.')
    parser.add_argument('--case', default='q4-practice-04', help='Session filename stem in the analysis and ZIP.')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError('Output already exists: ' + str(args.output))
        details = replay_case(args.zip, args.analysis, args.case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(details, stream, ensure_ascii=False, indent=2)
    except (AssertionError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print('Replay verification failed: ' + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
