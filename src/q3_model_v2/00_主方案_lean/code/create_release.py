"""Package local source, evidence and reports without transient caches."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import zipfile
from benchmark import code_hashes

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent/'B题第三问模型改进v2与本地验证.zip'


def main():
    summary = json.loads((ROOT/'results/holdout_summary.json').read_text(encoding='utf8'))
    assert summary['code_hashes'] == code_hashes()
    for name in ('validation.json', 'holdout_replay_validation.json'):
        assert json.loads((ROOT/'results'/name).read_text(encoding='utf8'))['passed']
    for name in ('本轮改进与实验结果.md', '模型推导与适用边界.md',
                 'figures/holdout_comparison.png', 'figures/median_trajectory_comparison.png'):
        assert (ROOT/name).is_file()
    assert (ROOT/'baseline_solver.py').read_bytes() == (ROOT.parents[1]/'archive/q3_improved/baseline_solver.py').read_bytes()
    assert (ROOT/'v1_solver.py').read_bytes() == (ROOT.parents[1]/'archive/q3_improved/solver.py').read_bytes()
    files = []
    for parent, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', 'tmp', '.git', '.pytest_cache'))
        for name in sorted(names):
            p = Path(parent)/name
            if p.suffix == '.pyc' or name == 'release_manifest.json':
                continue
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            files.append(p)
    manifest = dict(archive=ARCHIVE.name, code_hashes=code_hashes(), files={
        p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in files})
    manifest_path = ROOT/'release_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files+[manifest_path]:
            archive.write(p, 'q3_model_v2/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)+1
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(files)+1, bytes=ARCHIVE.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    main()
