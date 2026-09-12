"""Assemble unchanged experiment sources from the study-oriented repository.

The generated directory is a disposable execution view, never another editable
source tree. Imports, historical relative paths and SHA256 records stay valid.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'src/studies.json'


def index():
    manifest=json.loads(INDEX.read_text(encoding='utf-8'))
    paths={r['path'] for r in manifest['files']}
    logicals={r['logical'] for r in manifest['files']}
    for study,group in manifest['groups'].items():
        for folder,prefix in [('code',''),('reports',''),('records',''),('results','results/'),('figures','figures/')]:
            base=ROOT/group['path']/folder
            if not base.is_dir():continue
            for parent,dirs,names in os.walk(base):
                dirs[:]=[d for d in dirs if d not in ('tmp','__pycache__','official_runs')]
                for name in names:
                    path=Path(parent)/name;relative=path.relative_to(ROOT).as_posix()
                    if relative in paths or path.suffix=='.pyc' or '.jsonl' in name:continue
                    # The generated batch index belongs to the browsing view only.
                    if path == base/'README.md' and folder == 'results':continue
                    logical=group['logical_root']+'/'+prefix+path.relative_to(base).as_posix()
                    if logical in logicals:raise ValueError(f'Duplicate source filename across studies: {logical}')
                    manifest['files'].append(dict(logical=logical,path=relative,study=study))
                    paths.add(relative);logicals.add(logical)
    return manifest


def safe_path(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside workspace: {relative}')
    return path


def assemble():
    manifest=index()
    base=ROOT/'tmp/study_runs'
    base.mkdir(parents=True,exist_ok=True)
    target=Path(tempfile.mkdtemp(prefix='run-',dir=base))
    for record in manifest['files']:
        source=safe_path(ROOT,record['path'])
        dest=safe_path(target,record['logical'])
        if not source.is_file():
            raise FileNotFoundError(f'Missing organized source: {source}')
        content=source.read_bytes()
        if not dest.is_file() or dest.read_bytes()!=content:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(content)
    # A small number of historical checks compare with the old complete release.
    for parent,dirs,files in os.walk(ROOT/'archive'):
        dirs[:]=[d for d in dirs if d not in ('tmp','__pycache__','official_runs')]
        for name in files:
            source=Path(parent)/name
            if source.suffix not in ('.py','.json','.txt'):
                continue
            dest=safe_path(target,source.relative_to(ROOT))
            content=source.read_bytes()
            if not dest.is_file() or dest.read_bytes()!=content:
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(content)
    return target


def resolve(logical):
    """Resolve a historical project-relative name in the editable study tree."""
    for record in index()['files']:
        if record['logical']==logical:
            return safe_path(ROOT,record['path'])
    return safe_path(ROOT,logical)


def frozen_checks():
    failures=[];counts={}
    locations={r['logical']:safe_path(ROOT,r['path']) for r in index()['files']}
    specifications=[('q3','src/q3_model_v2/results/main_solution_freeze.json','file_sha256','src/q3_model_v2/'),
        ('q4','src/q4_model/results/client_release.json','source_sha256','src/')]
    specifications += [('share25','src/q4_model/results/share25_client_release.json','source_sha256',''),
                       ('share25_evidence','src/q4_model/results/share25_client_release.json','evidence_sha256','')]
    for label,logical,key,prefix in specifications:
        manifest=json.loads(locations[logical].read_text(encoding='utf-8'))
        counts[label]=len(manifest[key])
        for name,expected in manifest[key].items():
            path=locations.get(prefix+name,safe_path(ROOT,prefix+name))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                failures.append(prefix+name)
    return counts,failures


def artifact_snapshot(runtime):
    result={}
    for record in index()['files']:
        path=safe_path(runtime,record['logical'])
        if path.suffix in ('.json','.png','.svg','.csv','.md'):
            result[record['logical']]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def collect_outputs(runtime,study,before):
    """Save changed runtime artifacts in a new batch; preserve historical evidence."""
    manifest=index();records={r['logical']:r for r in manifest['files']}
    copied=[]
    for parent,dirs,files in os.walk(runtime):
        dirs[:]=[d for d in dirs if d not in ('tmp','__pycache__','official_runs','archive')]
        for name in files:
            source=Path(parent)/name
            if source.suffix not in ('.json','.png','.svg','.csv','.md'):
                continue
            logical=source.relative_to(runtime).as_posix()
            data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
            if before.get(logical)==sha:
                continue
            if logical in records:
                original=safe_path(ROOT,records[logical]['path'])
                if original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest()!=before.get(logical):
                    raise RuntimeError(f'Source artifact changed during execution: {original}')
            else:
                if '/results/' not in logical and '/figures/' not in logical:
                    continue
            dest=safe_path(ROOT,manifest['groups'][study]['path']+'/results/local_runs/'+runtime.name+'/'+logical)
            if dest.exists():
                raise FileExistsError(f'Runtime output would overwrite an existing batch: {dest}')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            copied.append(dest.relative_to(ROOT).as_posix())
    return copied
