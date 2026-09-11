"""Package the manual official-interface entry without contacting any service."""
import ast
import hashlib
import json
import os
from pathlib import Path
import zipfile
from benchmark import code_hashes

root = Path(__file__).resolve().parent
test = json.loads((root/'http-validation.json').read_text(encoding='utf8'))
selection = json.loads((root/'results/selection.json').read_text(encoding='utf8'))
assert test['passed'] and test['tests_run'] == 12
assert selection['frozen_code_hashes'] == code_hashes()
paths = []
for parent, dirs, names in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in ('tmp', '__pycache__', '.git', 'official_runs'))
    for name in sorted(names):
        if name in ('release_manifest.json', 'official_release_manifest.json') or name.endswith('.pyc'):
            continue
        p = Path(parent)/name
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        paths.append(p)
manifest_path = root/'official_release_manifest.json'
manifest = dict(official_simulator_contacted=False, local_http_checks=12, frozen_solver_unchanged=True,
                code_hashes=code_hashes(), files={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
archive_path = root.parent/'B题第三问v2官方接口版.zip'
with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in paths+[manifest_path]:
        archive.write(p, 'q3_model_v2/'+p.relative_to(root).as_posix())
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
print(json.dumps(dict(path=str(archive_path), bytes=archive_path.stat().st_size,
                     files=len(paths)+1, solver_unchanged=True), ensure_ascii=False))
