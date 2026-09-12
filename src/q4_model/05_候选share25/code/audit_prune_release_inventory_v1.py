"""Read-only Git payload inventory; writes only a new audit summary."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def git_paths(*args):
    data = subprocess.check_output(['git', '-C', str(ROOT), *args, '-z'])
    return {part.decode('utf-8') for part in data.split(b'\0') if part}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--authorized-team-id', default=None,
                        help='Explicitly authorized noncredential team identifier; never emitted in output.')
    args = parser.parse_args()
    untracked = git_paths('ls-files', '--others', '--exclude-standard')
    tracked = git_paths('diff', '--name-only', 'HEAD')
    paths = sorted(p for p in untracked | tracked if (ROOT/p).is_file())
    patterns = {
        'potential_team_identifier': re.compile(r'(?<![\w.])2026\d{8}(?!\w)'),
        'github_credential_shape': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
        'private_key_header': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'aws_access_key_shape': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
        'openai_credential_shape': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b'),
    }
    sensitive_keys = {'robot_id', 'robotid', 'team_id', 'api_key', 'apikey',
        'password', 'authorization', 'access_token', 'refresh_token', 'secret', 'cookie'}
    hits = []
    intentional_team_identifiers = []
    machine_paths = []
    raw_logs = []
    sizes = []
    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in sensitive_keys and child not in (None, '', False):
                    placeholder = isinstance(child, str) and re.search(
                        r'local|fixture|test|example|placeholder|your|redacted|demo|参赛|队号|填写', child, re.I)
                    if not placeholder:
                        hits.append({'type': 'nonplaceholder_sensitive_json_field', 'path': path})
                walk(child, path)
        elif isinstance(value, list):
            for child in value:
                walk(child, path)
    for name in paths:
        p = ROOT/name
        sizes.append((p.stat().st_size, name))
        if '.jsonl' in p.name.lower():
            raw_logs.append(name)
        try:
            text = p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            continue
        for kind, pattern in patterns.items():
            matches = list(pattern.finditer(text))
            if kind == 'potential_team_identifier' and matches and args.authorized_team_id:
                authorized = [m for m in matches if m.group() == args.authorized_team_id]
                if authorized:
                    intentional_team_identifiers.append(dict(
                        type='intentional_user_provided_team_id', path=name, occurrences=len(authorized)))
                matches = [m for m in matches if m.group() != args.authorized_team_id]
            if matches:
                hits.append({'type': kind, 'path': name})
        if re.search(r'[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s]+', text):
            machine_paths.append(name)
        if p.suffix.lower() == '.json':
            try:
                walk(json.loads(text), name)
            except json.JSONDecodeError:
                pass
    hits = [dict(type=a, path=b) for a, b in sorted({(h['type'], h['path']) for h in hits})]
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Git changed and untracked nonignored files at this snapshot; excludes this new output',
        pending_files=len(paths), untracked_files=len(untracked), tracked_changed_files=len(tracked),
        total_bytes=sum(n for n, _ in sizes),
        untracked_bytes=sum((ROOT/p).stat().st_size for p in untracked if (ROOT/p).is_file()),
        largest_files=[dict(path=p, bytes=n) for n, p in sorted(sizes, reverse=True)[:10]],
        files_at_least_50_mib=[p for n, p in sizes if n >= 50*1024**2],
        files_at_least_100_mib=[p for n, p in sizes if n >= 100*1024**2],
        sensitive_findings=hits, raw_jsonl_paths=raw_logs,
        intentional_user_provided_team_id= intentional_team_identifiers,
        machine_provenance_path_files=machine_paths,
        interpretation='Machine provenance paths are local metadata, not authentication. Preserve frozen evidence bytes; use separate sanitized derivatives if a public release is later required.',
        scanner_limit='Pattern and structured-field screening, not a guarantee that all possible secrets are detectable.',
        initial_scan_resolution='An initial loose identifier regex also matched fractional digits; word and decimal boundaries now exclude them. An explicit noncredential team identifier in authorized running examples is recorded separately, with the identifier itself omitted.',
        extension_counts=dict(Counter(Path(p).suffix for p in paths)),
        runtime_boundary='15 fingerprinted runtime Python files plus root launcher and installation instructions; no research directory required.',
        evidence_boundary='Preserve all research outcomes and referenced input logs/certificates; fair-control replay reports require the saved share25 inputs from the 50000 and 53000 folders.',
        recommended_commits=['Research code, reports and full linked evidence, including negative outcomes.',
            'Independent 25-station ordered-pruning runtime, release documentation and local validation.'],
        passed=not hits and not raw_logs and not any(n >= 100*1024**2 for n, _ in sizes))
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: summary[k] for k in ('passed', 'pending_files', 'untracked_files',
        'total_bytes', 'sensitive_findings', 'raw_jsonl_paths', 'files_at_least_100_mib')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
