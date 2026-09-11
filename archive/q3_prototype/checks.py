"""只检查会影响结论的规则、几何保证及困难端点。"""
import json
import math
from pathlib import Path
import numpy as np
from environment import LocalArena
from experiment import run
from solver import coverage_points,reception_certified,initial_belief


def main():
    src=[dict(channel=1,position=[0.,0.],radius=1000.),dict(channel=2,position=[300.,0.],radius=1500.)]
    env=LocalArena(src,field='constant')
    assert env.measure([1000,0],1)['measure_result']=='direction'
    assert env.measure([1000.001,0],1)['measure_result']=='no_signal'
    first=env.measure([400,100],2)
    assert env.measure([400,100],2)==first
    assert env.measure([5,0],1)['measure_result']=='near'
    before=env.time_s
    assert env.clear([20.001,0],1)['clear_result']=='no_target_in_range'
    assert abs(env.time_s-before-(15.001/5+3))<1e-8
    env.measure([25,0],2)
    assert env.channel==2
    assert env.clear([20,0],1)['clear_result']=='success'
    assert env.channel==2
    assert env.measure([0,0],1)['measure_result']=='no_signal'
    env.evaluation()
    # 与最接近环点夹角最多30°；距离平方对径向距离为凸函数。
    bounds=[math.sqrt(r*r+1200**2-2*r*1200*math.cos(math.pi/6)) for r in [1000,1800]]
    assert max(bounds)<1000
    # 已知收到信号条件下的新接收判据，数值交叉检查随机区域内点。
    rng=np.random.default_rng(872)
    P=initial_belief(np.array([0.,0.]),17.)
    for _ in range(200):
        q=rng.uniform([-500,-500],[1500,1000])
        if reception_certified(P,q,np.zeros(2)):
            w=rng.dirichlet(np.ones(len(P)),size=100)
            g=w@P
            assert np.all(np.linalg.norm(g-q,axis=1)<=np.maximum(1000,np.linalg.norm(g,axis=1))+1e-6)
    results=[]
    cases=[]
    for d in [0.,5.,5.001,20.,999.999,1000.,1499.999,1500.]:
        for a in [0.,359.995]:
            angle=math.radians(a)
            v=np.array([math.cos(angle),math.sin(angle)])
            pos=v*d
            # 三角函数浮点误差可把“恰在圆上”的测试点放到圆外1e-13米。
            # 仅修正测试数据，不放宽环境的接收或清除判据。
            if d>0 and np.linalg.norm(pos)>d:pos=v*(np.nextafter(d,0.)/np.linalg.norm(v))
            assert np.linalg.norm(pos)<=max(1000.,d)
            cases.append((f'distance-{d}-angle-{a}',[dict(channel=1,position=pos.tolist(),radius=max(1000.,d))]))
    for name,sources in cases:
        for policy in ['geometry','time']:
            r,detail=run('single',901,'constant',policy,sources=sources)
            r['edge_case']=name;results.append(r)
    ring=[dict(channel=i+1,position=[1800*math.cos((i+.5)*math.pi/6),1800*math.sin((i+.5)*math.pi/6)],radius=1000.) for i in range(12)]
    colocated=[dict(channel=i+1,position=[1800.,0.],radius=1000.) for i in range(16)]
    origin=[dict(channel=i+1,position=[0.,0.],radius=1000.) for i in range(10)]
    last=[dict(channel=i+1,position=[20.*i,0.],radius=1000.) for i in range(9)]+[dict(channel=20,position=[1800*math.cos(math.pi/6),900.],radius=1000.)]
    for name,sources in [('boundary-ring',ring),('colocated-16',colocated),('origin-10',origin),('last-boundary-source',last)]:
        for policy in ['geometry','time']:
            for schedule in ['immediate','deferred']:
                r,detail=run('multi',991,'constant',policy,schedule,sources=sources)
                r['edge_case']=name;results.append(r)
    out=dict(rules_checked=['1000米接收边界','同点固定误差','5米near','20米清除边界','失败3秒','clear不切频','清除后无信号','时间分项一致'],
             ring_coverage_max_distance_m=max(bounds),reception_random_crosschecks=200,edge_results=results)
    Path('checks-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(edge_cases=len(results),all_cleared=sum(r['all_cleared'] for r in results),
                         errors=[r for r in results if r['error'] or r['belief_violations']],coverage_bound=max(bounds)),ensure_ascii=False))


if __name__=='__main__':main()
