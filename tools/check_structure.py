"""Read-only study navigation and frozen-evidence checks. Never starts HTTP."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from study_runtime import ROOT, index, safe_path, frozen_checks


def check():
    errors = []
    manifest = index()
    locations = {}
    for record in manifest['files']:
        if record['logical'] in locations:
            errors.append('重复逻辑路径：' + record['logical'])
        source = safe_path(ROOT, record['path'])
        locations[record['logical']] = source
        if not source.is_file():
            errors.append('缺少文件：' + record['path'])
    for group in manifest['groups'].values():
        if not (ROOT / group['path'] / 'README.md').is_file():
            errors.append('缺少阶段说明：' + group['path'])
    for name, problem in [('运行第三问.cmd', 'q3'), ('运行第四问.cmd', 'q4'),
                          ('运行第四问候选.cmd', 'share25')]:
        path = ROOT / name
        if not path.is_file() or ('tools\\launch_model.ps1" -Problem ' + problem) not in path.read_text(encoding='utf-8-sig'):
            errors.append('运行入口未指向对应方案：' + name)
    counts, failures = frozen_checks()
    errors.extend('冻结文件不匹配：' + name for name in failures)
    additional = 0
    for name, key in [('joint_selection.json', 'code_hashes'),
                      ('stress-endpoints/manifest.json', 'source_sha256')]:
        evidence = json.loads(locations['src/q4_model/results/' + name].read_text(encoding='utf-8'))
        for relative, expected in evidence[key].items():
            path = locations['src/' + relative]
            additional += 1
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append('第四问实验指纹不匹配：' + relative)
    links = 0
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git', 'tmp', 'local_data', '__pycache__',
                    '.venv', '.codex', '.agents', 'official_runs')]
        for filename in files:
            if not filename.endswith('.md'):
                continue
            path = Path(parent) / filename
            content = path.read_text(encoding='utf-8-sig')
            if content.count('```') % 2:
                errors.append('代码块未闭合：' + str(path.relative_to(ROOT)))
            for target in re.findall(r'\]\(([^\n)]+)\)', content):
                if ':' in target or target.startswith('#'):
                    continue
                dest = target.split('#')[0].strip('<>')
                if dest.endswith(('.pdf', '.docx', '.xlsx', '.zip')) or 'local_data/' in dest:
                    continue
                links += 1
                if not (path.parent / dest).exists():
                    errors.append(f'失效链接：{path.relative_to(ROOT)} -> {target}')
    return dict(passed=not errors, local_only=True, official_simulator_contacted=False,
                studies=len(manifest['groups']), mapped_files=len(manifest['files']),
                document_links=links, frozen_files=counts, additional_evidence_checks=additional,
                errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result['passed']:
        print(f"{result['studies']} 个实验组、三个入口、冻结文件及 {result['document_links']} 个文档链接检查通过。")
        print('全部为本地只读检查，未连接官方模拟器。')
    else:
        for error in result['errors']:
            print(error)
    raise SystemExit(0 if result['passed'] else 1)
