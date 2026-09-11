"""Exercise the new CLI only against an ephemeral local HTTP test double."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from test_client import Scenario, LocalSimulator
from environment import LocalArena
from benchmark import multi_case
from joint_solver import solve_multi, DEFAULT_SCHEDULE


class ArenaScenario(Scenario):
    def __init__(self):
        super().__init__()
        self.arena=LocalArena(multi_case(1000),1000,'hash')
        self.drop_first_measure=True

    def dispatch(self,path,raw):
        if path not in ('/measure','/clear'):
            return super().dispatch(path,raw)
        request=json.loads(raw)
        self.requests.append((path,raw,request))
        key=request['request_id']
        if key in self.cache:
            old_path,old_raw,response=self.cache[key]
            assert (path,raw)==(old_path,old_raw)
            return response,False
        self.executions[path]+=1
        q=[request['position']['x'],request['position']['y']]
        reply=(self.arena.measure if path=='/measure' else self.arena.clear)(q,request['channel'])
        self.virtual=self.arena.time_s
        response=dict(accepted=True,real_timestamp_ms=int(time.time()*1000),virtual_time_s=self.virtual,**reply)
        self.cache[key]=(path,raw,response)
        drop=self.drop_first_measure and path=='/measure'
        if drop:self.drop_first_measure=False
        return response,drop


def main():
    root=Path(__file__).resolve().parent
    scenario=ArenaScenario()
    with tempfile.TemporaryDirectory(dir=root/'tmp') as temp, LocalSimulator(scenario) as server:
        assert ':2026' not in server.url
        log=Path(temp)/'joint.jsonl'
        command=[sys.executable,'-X','utf8',str(root/'joint_official_client.py'),
                 '--robot-id','LOCAL-ONLY-TEST','--url',server.url,'--log',str(log)]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=90)
        assert result.returncode==0,result.stderr
        events=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
        summary=next(e for e in events if e.get('type')=='session_summary')
        assert summary['complete'] and summary['exit_accepted']
        assert summary['schedule']==DEFAULT_SCHEDULE
        assert summary['solver_entry']=='joint_solver.solve_multi'
        assert 'joint_solver.py' in summary['joint_source_sha256']
        assert scenario.arena.evaluation()['all_cleared']
        direct=LocalArena(multi_case(1000),1000,'hash')
        solve_multi(direct)
        assert direct.events==scenario.arena.events
        assert scenario.executions['/measure']==scenario.arena.counts['measure']
        assert len([r for r in scenario.requests if r[0]=='/measure'])>scenario.executions['/measure']
        output=dict(passed=True,official_simulator_used=False,ephemeral_local_http_only=True,
            cli_executed=True,dropped_response_retried_without_double_charge=True,
            exact_events_equal_direct_solver=True,schedule=DEFAULT_SCHEDULE,
            total_s=direct.time_s,sources=direct.evaluation()['source_count'],
            joint_source_sha256=summary['joint_source_sha256'])
        (root/'results'/'joint_http_validation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        print(json.dumps(output))


if __name__=='__main__':
    (Path(__file__).resolve().parent/'tmp').mkdir(exist_ok=True)
    main()
