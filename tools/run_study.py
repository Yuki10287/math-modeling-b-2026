"""Run a named study without changing its frozen import or path semantics."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from study_runtime import ROOT,index,assemble,artifact_snapshot,collect_outputs,frozen_checks


ENTRIES={
    'q12_geometry':'geometry.py','q12_current':'q2_solver.py',
    'q3_main':'benchmark.py','q3_local':'local_checks.py','q3_joint':'benchmark_joint.py',
    'q3_cost':'benchmark_forecast.py','q3_value':'benchmark_value.py',
    'q3_official':'analyze_official_results.py','q3_probe':'benchmark_exploration.py',
    'q3_closed':'benchmark_closed.py','q4_initial':'benchmark.py','q4_main':'benchmark_joint.py',
    'q4_official':'analyze_official_results.py','q4_cells':'benchmark_cells.py',
    'q4_share':'benchmark_task_sharing.py',
}


def main():
    parser=argparse.ArgumentParser(description='按实验运行；代码、结果和说明在各自实验目录。')
    parser.add_argument('study',choices=sorted(list(ENTRIES)+['prepare','verify']))
    parser.add_argument('--script',help='该实验中要运行的另一个脚本名')
    parser.add_argument('--tests',action='store_true',help='运行该实验归属的全部本地单元检查')
    args,extra=parser.parse_known_args()
    if extra and extra[0]=='--':extra=extra[1:]
    if args.study=='verify':
        counts,failures=frozen_checks()
        print(json.dumps(dict(verified=not failures,files=counts,failures=failures),ensure_ascii=False))
        return 1 if failures else 0
    runtime=assemble()
    if args.study=='prepare':
        print(runtime)
        return 0
    records=[r for r in index()['files'] if r['study']==args.study]
    if args.tests:
        tests=[r for r in records if '/code/' in r['path'] and Path(r['logical']).name.startswith('test_') and r['logical'].endswith('.py')]
        if not tests:raise ValueError('这个实验组没有 test_ 脚本。')
        parents={str((runtime/r['logical']).parent) for r in tests}
        if len(parents)!=1:raise ValueError('测试目录不唯一。')
        # Separate process keeps same-named modules in different questions isolated.
        command=[sys.executable,'-X','utf8','-m','unittest','-v']+[Path(r['logical']).stem for r in tests]
        return subprocess.run(command,cwd=parents.pop()).returncode
    name=args.script or ENTRIES[args.study]
    match=[r for r in records if '/code/' in r['path'] and Path(r['logical']).name==name and r['logical'].endswith('.py')]
    if not match and args.script:
        # A few historical audits are saved beside the batch they certify.
        match=[r for r in records if Path(r['logical']).name==name and r['logical'].endswith('.py')]
    if len(match)!=1:raise ValueError(f'该实验没有唯一脚本 {name}')
    if name in ('official_client.py','joint_official_client.py','q4_official_client.py','q4_share25_client.py'):
        parser.error('官方会话请使用项目根目录的对应运行入口。')
    before=artifact_snapshot(runtime)
    result=subprocess.run([sys.executable,'-X','utf8',str(runtime/match[0]['logical']),*extra],cwd=ROOT)
    if result.returncode==0:
        outputs=collect_outputs(runtime,args.study,before)
        if outputs:print('已归档生成结果：\n'+'\n'.join(outputs))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
