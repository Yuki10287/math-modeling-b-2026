"""Replay a saved policy decision with a warm cache; record CPU and wall time."""
import argparse
import json
import time
from pathlib import Path

import bootstrap
from validate_model import BoundedArena, PublicAPI
from conditional_preview import ConditionalSelector, solve_conditional


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    assert not args.out.exists()
    saved=json.loads(args.case.read_text(encoding='utf-8'))
    row=saved['result']
    forecasts=[r for r in saved['trace'] if r.get('phase')=='conditional_forecast']
    selector=ConditionalSelector(cache={r['state_key']:r['prediction'] for r in forecasts},
                                 risk=forecasts[0]['risk'])
    selection_times=[]
    def timed_selector(*params):
        wall,cpu=time.perf_counter(),time.process_time()
        result=selector(*params)
        selection_times.append(dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu))
        return result
    arena=BoundedArena(saved['sources'],row['seed'],row['field'],max_actions=3000)
    trace=[]
    wall,cpu=time.perf_counter(),time.process_time()
    result=solve_conditional(PublicAPI(arena),trace=trace,selector=timed_selector)
    elapsed=dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu)
    assert result['complete'] and arena.events==saved['events']
    assert all(r['cached'] for r in trace if r.get('phase')=='conditional_forecast')
    payload=dict(local_only=True,official_simulator_used=False,exact_event_replay=True,
                 warm_cache=True,case=row['name'],elapsed=elapsed,
                 selector_wall_s=sum(r['wall_s'] for r in selection_times),
                 selector_cpu_s=sum(r['cpu_s'] for r in selection_times),
                 note='Single diagnostic replay; a warm cache is not representative of a fresh official session.')
    args.out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload),flush=True)


if __name__=='__main__': main()
