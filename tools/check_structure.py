"""Read-only project layout, documentation and frozen-release checks. No HTTP."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check():
    errors = []
    for name in ('src/q1_q2_geometry', 'src/q3_model_v2', 'src/q4_model',
                 'archive/q3_prototype', 'archive/q3_improved', 'docs/论文整理'):
        if not (ROOT/name).is_dir():
            errors.append(f'缺少目录：{name}')
    for name, target in [('运行第三问.cmd', 'src\\q3_model_v2\\run_official.ps1'),
                         ('运行第四问.cmd', 'src\\q4_model\\run_official.ps1'),
                         ('运行第四问候选.cmd', 'src\\q4_model\\run_share25.ps1')]:
        path = ROOT/name
        if not path.is_file() or target not in path.read_text(encoding='utf-8-sig'):
            errors.append(f'根目录入口未指向已验证版本：{name}')
    checks = []
    for script in ('src/q3_model_v2/verify_main_solution.py', 'src/q4_model/verify_release.py',
                   'src/q4_model/verify_share25_release.py'):
        result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/script)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        checks.append(dict(script=script, passed=result.returncode==0))
        if result.returncode:
            errors.append(result.stdout+result.stderr)
    q4 = ROOT/'src/q4_model/results'
    for name, key in [('joint_selection.json', 'code_hashes'),
                      ('stress-endpoints/manifest.json', 'source_sha256')]:
        manifest = json.loads((q4/name).read_text(encoding='utf-8'))
        for relative, expected in manifest[key].items():
            path = ROOT/'src'/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'第四问实验指纹不匹配：{relative}')
    links = 0
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git','tmp','local_data','__pycache__',
                    '.venv','.codex','.agents','official_runs')]
        for filename in files:
            if not filename.endswith('.md'):
                continue
            path = Path(parent)/filename
            content = path.read_text(encoding='utf-8-sig')
            if content.count('```')%2:
                errors.append(f'代码块未闭合：{path.relative_to(ROOT)}')
            for target in re.findall(r'\]\(([^\n)]+)\)', content):
                if ':' in target or target.startswith('#'):
                    continue
                dest = target.split('#')[0].strip('<>')
                # Original problem attachments are intentionally not in a Git checkout.
                if dest.endswith(('.pdf','.docx','.xlsx','.zip')) or 'local_data/' in dest:
                    continue
                links += 1
                if not (path.parent/dest).exists():
                    errors.append(f'失效链接：{path.relative_to(ROOT)} -> {target}')
    return dict(passed=not errors, local_only=True, official_simulator_contacted=False,
                document_links=links, release_checks=checks, errors=errors)


if __name__ == '__main__':
    result = check()
    if result['passed']:
        print(f'目录、三个运行入口、第三/四问原版与候选版本及 {result["document_links"]} 个文档链接检查通过。')
        print('全部为本地只读检查，未连接官方模拟器。')
    else:
        for error in result['errors']:
            print(error)
    raise SystemExit(0 if result['passed'] else 1)
