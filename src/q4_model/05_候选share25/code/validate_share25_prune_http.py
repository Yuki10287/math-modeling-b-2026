"""Two complete production prune CLI sessions on owned loopback fixtures only.

Reuse saved local source layouts. One includes actually skipped doomed optical
attempts; the other exercises the 16-source stopping certificate. No official
port, server, source count, or source truth is exposed to the client process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0,str(Q4))
from shared import load_file
from polar_cover import PolarCover
from audit_ordered_optical_prune_v2 import verify_case
from refined_cover_audit import exact_certificate_audit
from ordered_optical_prune_experiment import constrained_replay

FIXTURE = Q4/'02_主方案25站_shared/code/local_http_fixture.py'
ENV = Q4.parent/'q3_model_v2/00_主方案_lean/code/environment.py'
VALIDATION = Q4/'00_公共几何与定位/code/validation.py'
fixture = load_file('_prune_cli_owned_fixture',FIXTURE)
LocalArena = load_file('_prune_cli_local_arena',ENV).LocalArena
validate_run = load_file('_prune_cli_independent_trajectory',VALIDATION).validate_run
ENTRY = Q4/'q4_share25_prune_client.py'
CASE_INPUTS = (
    (HERE.parent/'results/ordered_optical_prune_dev48000/48000-uniform-share25.json','smooth'),
    (HERE.parent/'results/station22_ordered_prune_pressure53000/53300-omni_heavy-extreme-share25.json','extreme'),
)


def demand(condition,message):
    if not condition:
        raise AssertionError(message)


def write(path,data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def json_form(data):
    return json.loads(json.dumps(data,ensure_ascii=False))


def public_api(arena):
    class Public:
        __slots__=()
        @property
        def position(self):
            return arena.position.copy()
        @property
        def channel(self):
            return arena.channel
        def measure(self,q,c):
            return arena.measure(q,c)
        def clear(self,q,c):
            return arena.clear(q,c)
    return Public()


def trace_without_proof_runtime(trace):
    normalized=json_form(trace)
    for row in normalized:
        if row.get('phase')=='ordered_optical_prune' and isinstance(row.get('info'),dict):
            row['info'].pop('runtime_s',None)
    return normalized


def clean_environment():
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    return env


def import_help_checks(out):
    unrelated=out/'unrelated-working-directory'
    unrelated.mkdir()
    # runpy gets the same script-directory import root as a normal Python
    # entry. A socket guard makes import/help incapable of contacting even
    # the official default port if future import side effects regress.
    bootstrap='''import pathlib,runpy,socket,sys
entry=pathlib.Path(sys.argv[1]).resolve()
def forbidden(*args,**kwargs):
    raise RuntimeError('NETWORK_FORBIDDEN_DURING_IMPORT_OR_HELP')
socket.socket.connect=forbidden
socket.create_connection=forbidden
sys.path.insert(0,str(entry.parent))
mode=sys.argv[2]
sys.argv=[str(entry),'--help'] if mode=='help' else [str(entry)]
runpy.run_path(str(entry),run_name='__main__' if mode=='help' else '_offline_import_check')
'''
    rows=[]
    for location,cwd,entry in (('project_root',PROJECT,str(ENTRY.relative_to(PROJECT))),
                                ('unrelated_cwd',unrelated,str(ENTRY))):
        for mode in ('import','help'):
            result=subprocess.run([sys.executable,'-X','utf8','-c',bootstrap,entry,mode],
                cwd=cwd,env=clean_environment(),capture_output=True,text=True,encoding='utf-8',timeout=30)
            demand(result.returncode==0,f'{location} {mode} failed: {result.stderr}')
            if mode=='help':
                demand(all(flag in result.stdout for flag in ('--robot-id','--url','--log')),'missing CLI flags')
            rows.append(dict(location=location,mode=mode,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr,network_disabled=True,pythonpath_removed=True))
    write(out/'import-help.json',dict(passed=True,rows=rows))
    return unrelated


def test_hashes():
    paths=[Path(__file__),FIXTURE,ENV,VALIDATION,
           HERE/'audit_ordered_optical_prune_v1.py',HERE/'audit_ordered_optical_prune_v2.py',
           HERE/'audit_negative_region_boxes_v1.py',HERE/'ordered_optical_prune_experiment.py',
           HERE/'negative_region_boxes_v1.py',HERE/'continuous_hypothesis_experiment.py',
           HERE/'late_negative_optical_experiment.py',HERE/'refined_cover_audit.py']
    return {p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_case(input_path,field,directory,cwd,relative_entry,snapshot):
    import ordered_optical_prune
    directory.mkdir()
    log=directory/'client.jsonl'
    old=json.loads(input_path.read_text(encoding='utf-8'))
    sources,seed=old['sources'],old['summary']['seed']
    arena=LocalArena(sources,seed,field)
    scenario=fixture.ArenaScenario(arena)
    scenario.drop_first_measure=True
    process=None
    with fixture.LocalServer(scenario) as server:
        parsed=urlsplit(server.url)
        demand(parsed.hostname=='127.0.0.1' and parsed.port==server.server.server_address[1]
               and parsed.port not in (None,0,2026),'fixture endpoint is not owned loopback')
        entry=str(ENTRY.relative_to(PROJECT)) if relative_entry else str(ENTRY)
        command=[sys.executable,'-X','utf8',entry,'--robot-id','local-test-team',
                 '--url',server.url,'--log',str(log)]
        started=time.perf_counter()
        try:
            process=subprocess.run(command,cwd=cwd,env=clean_environment(),capture_output=True,
                text=True,encoding='utf-8',timeout=180,check=False)
        finally:
            write(directory/'server.json',dict(local_only=True,official_contacted=False,url=server.url,
                sources=sources,seed=seed,field=field,events=arena.events,evaluation=arena.evaluation(),
                requests=[dict(path=p,body_utf8=b.decode('utf-8'),request=r) for p,b,r in scenario.requests],
                executions=dict(scenario.executions),active=scenario.active,
                input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()))
            if process is not None:
                (directory/'stdout.txt').write_text(process.stdout,encoding='utf-8')
                (directory/'stderr.txt').write_text(process.stderr,encoding='utf-8')
                write(directory/'subprocess.json',dict(command=command,cwd=str(cwd),returncode=process.returncode,
                    elapsed_wall_s=time.perf_counter()-started,pythonpath_removed=True))
    demand(process is not None and process.returncode==0,f'CLI failed: {None if process is None else process.stderr}')
    records=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    starts=[r for r in records if r.get('type')=='session_start']
    summaries=[r for r in records if r.get('type')=='session_summary']
    demand(len(starts)==len(summaries)==1,'expected one start and one summary')
    summary=summaries[0]
    for record in (starts[0],summary):
        demand(record['variant']=='share25_prune' and record['problem']==4,'wrong candidate identity')
        demand(record['source_sha256']==snapshot,'production source hash mismatch')
    demand(summary['complete'] is True and summary['exit_accepted'] is True and not scenario.active,'incomplete or active session')
    demand(summary['successful_clear_count']==len(sources) and arena.evaluation()['all_cleared'],'hidden fixture sources not all cleared')
    demand(abs(summary['total_virtual_s']-arena.time_s)<=5.1e-7,'HTTP time mismatch')
    demand(summary['official_program_runtime_s'] is None,'local test claimed official runtime')
    demand(isinstance(summary.get('prune_audit'),dict) and summary['prune_audit'].get('passed') is True,'client omitted successful proof audit')
    trace=json.loads(Path(str(log)+'.beliefs.json').read_text(encoding='utf-8'))
    direct=LocalArena(sources,seed,field)
    direct_trace=[]
    result=ordered_optical_prune.solve_multi(public_api(direct),variant='share25',trace=direct_trace)
    direct_trace=json_form(direct_trace)
    demand(arena.events==direct.events,'CLI versus direct production actions differ')
    demand(trace_without_proof_runtime(trace)==trace_without_proof_runtime(direct_trace),
           'CLI versus direct trace differs beyond contraction wall time')
    demand(all(summary.get(k)==json_form(v) for k,v in result.items()),'CLI solver result fields differ')
    case=dict(sources=sources,result=summary,events=arena.events,trace=trace)
    mesh=PolarCover()
    audit=validate_run(sources,arena.events,trace,summary,mesh.stations,mesh.indices)
    audit['exact_completion']=exact_certificate_audit(summary,arena.events)
    audit['independent_prune']=verify_case(case)
    demand(audit['independent_prune']['plans']==sum(r['phase']=='optical_plan' for r in trace),'optical proof missing')
    replay=constrained_replay(old)
    demand(arena.events==replay['events'],'production HTTP path differs from frozen baseline deletion replay')
    audit['independent_baseline_replay']=verify_case(old,replay)
    skipped=len(old['events'])-len(arena.events)
    demand(skipped==replay['removed_actual'],'skipped actual count mismatch')
    demand(summary['optical_prune_skipped_before_success']==skipped,'client summary confuses planned and actual skipped points')
    measurement_requests=[r for r in scenario.requests if r[0]=='/measure']
    demand(len(measurement_requests)==arena.counts['measure']+1,'retry executed extra measurement')
    demand(measurement_requests[0][1]==measurement_requests[1][1],'retry changed body/id')
    demand(scenario.executions['/enter']==scenario.executions['/exit']==1,'unexpected enter/exit count')
    demand(scenario.executions['/measure']==arena.counts['measure'],'duplicate measurement executed')
    demand(scenario.executions['/clear']==arena.counts['clear_success']+arena.counts['clear_fail'],'clear accounting mismatch')
    write(directory/'http-case.json',case)
    write(directory/'direct-case.json',dict(sources=sources,result=result,events=direct.events,trace=direct_trace))
    write(directory/'baseline-replay.json',replay)
    write(directory/'audits.json',audit)
    return dict(passed=True,seed=seed,field=field,sources=len(sources),cleared=len(sources),
        total_virtual_s=arena.time_s,actual_skipped=skipped,
        certificate_basis=summary['certificate']['basis'],retry_same_body_executed_once=True,
        direct_events_exactly_equal=True,direct_trace_equal_except_proof_runtime=True,
        independently_audited_optical_plans=audit['independent_prune']['plans'],
        relative_entry=relative_entry,cwd=str(cwd),fixture_url=server.url)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import q4_share25_prune_client as client
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    snapshot,tests=client.source_hashes(),test_hashes()
    write(out/'manifest.json',dict(local_only=True,official_contacted=False,source_sha256=snapshot,
        validation_sha256=tests,python=platform.python_version(),
        cases=[dict(file=str(p.relative_to(PROJECT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),field=f)
               for p,f in CASE_INPUTS],
        policy='Only owned 127.0.0.1:0 fixture URLs, explicitly reject port 2026; no PYTHONPATH.',
        scope='Production transport/entry/proof equivalence; not model selection or performance testing.'))
    unrelated=import_help_checks(out)
    rows=[]
    for i,(path,field) in enumerate(CASE_INPUTS):
        try:
            row=run_case(path,field,out/f'case-{i+1}',PROJECT if i==0 else unrelated,i==0,snapshot)
        except Exception as exc:
            row=dict(passed=False,case=str(path),error=repr(exc),traceback=traceback.format_exc())
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    stable=client.source_hashes()==snapshot and test_hashes()==tests
    good=[r for r in rows if r['passed']]
    report=dict(passed=stable and all(r['passed'] for r in rows),source_snapshot_stable=stable,rows=rows,
        actual_deleted_case_verified=any(r['actual_skipped']>0 for r in good),
        both_stopping_branches={r['certificate_basis'] for r in good}=={'count_upper_bound','directional_triangle_cover'},
        local_only=True,official_contacted=False)
    report['passed'] &= report['actual_deleted_case_verified'] and report['both_stopping_branches']
    write(out/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
