"""B题全向源闭环原型。策略只通过 measure/clear 反馈工作，不接触真值。"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
import numpy as np

ANGLE_EPS = math.pi / 180


def clip(P, n, b):
    if not len(P): return np.empty((0, 2))
    d = P @ n - b
    if np.max(d) <= 1e-8: return P.copy()
    if np.min(d) > 1e-8: return np.empty((0, 2))
    result=[]
    for i in range(len(P)):
        j=(i-1)%len(P)
        if (d[i] <= 1e-8) != (d[j] <= 1e-8):
            result.append(P[j]+d[j]/(d[j]-d[i])*(P[i]-P[j]))
        if d[i] <= 1e-8: result.append(P[i])
    return np.asarray(result)


def wedge(P, s, angle):
    lo, hi = math.radians(angle)-ANGLE_EPS, math.radians(angle)+ANGLE_EPS
    for n in [np.array([math.sin(lo),-math.cos(lo)]),np.array([-math.sin(hi),math.cos(hi)])]:
        P=clip(P,n,float(n@s))
    return P


def circle_polygon(s=(0,0),r=1800.,n=128):
    a=(np.arange(n)+.5)*2*math.pi/n
    return np.asarray(s)+r/math.cos(math.pi/n)*np.column_stack((np.cos(a),np.sin(a)))


def disk_clip(P,s,r=1500.,n=128):
    if not len(P): return P
    if np.max(np.linalg.norm(P-s,axis=1)) <= r: return P
    for a in np.arange(n)*2*math.pi/n:
        normal=np.array([math.cos(a),math.sin(a)])
        P=clip(P,normal,float(r+normal@s))
    return P


def diameter(P):
    d2=np.sum((P[:,None,:]-P[None,:,:])**2,axis=2)
    i,j=np.unravel_index(np.argmax(d2),d2.shape)
    return float(math.sqrt(d2[i,j])),(i,j)


def mec(P):
    """精确枚举支撑圆；先检查最远点对的直径圆，通常立即结束。"""
    if not len(P): raise ValueError('empty belief')
    if len(P)==1:return P[0].copy(),0.
    D,(i,j)=diameter(P)
    c=(P[i]+P[j])/2
    radius=float(np.max(np.linalg.norm(P-c,axis=1)))
    if radius <= D/2+1e-7:return c,radius
    offset=P.mean(axis=0);Q=P-offset
    centers=[c-offset]
    for a,b in itertools.combinations(Q,2): centers.append((a+b)/2)
    for a,b,c0 in itertools.combinations(Q,3):
        M=2*np.array([b-a,c0-a]);det=np.linalg.det(M)
        if abs(det)>1e-10:centers.append(np.linalg.solve(M,[b@b-a@a,c0@c0-a@a]))
    centers=np.asarray(centers)
    radii=np.max(np.linalg.norm(centers[:,None,:]-Q[None,:,:],axis=2),axis=1)
    best=np.argmin(radii)
    return centers[best]+offset,float(radii[best])


def initial_belief(s,angle):
    return disk_clip(wedge(circle_polygon(),s,angle),s)


def update_belief(P,s,angle):
    result=disk_clip(wedge(P,s,angle),s)
    if not len(result):raise RuntimeError('观测与当前可行域矛盾')
    return result


def nearest_certified_clear(P,s):
    """在圆心到当前位置线段上找最近可保证清除点；非全局投影。"""
    c,r=mec(P)
    if r>20-1e-6:return None
    if np.max(np.linalg.norm(P-s,axis=1))<=20-1e-6:return s.copy()
    lo,hi=0.,1.
    for _ in range(35):
        t=(lo+hi)/2;q=c+t*(s-c)
        if np.max(np.linalg.norm(P-q,axis=1))<=20-1e-6:lo=t
        else:hi=t
    return c+lo*(s-c)


def reception_certified(P,q,witness):
    # witness 已经收到该源；如果移动后更接近则仍可接收。
    # 只检查“比 witness 更远”的半平面部分是否完全在 q 的1000米圆内。
    farther=clip(P,2*(q-witness),float(q@q-witness@witness))
    return not len(farther) or np.max(np.linalg.norm(farther-q,axis=1))<=999.999


def candidates(P,s,witness=None):
    witness=s if witness is None else witness
    c,r=mec(P);_,(i,j)=diameter(P)
    axis=P[j]-P[i];axis/=max(np.linalg.norm(axis),1e-10)
    perp=np.array([-axis[1],axis[0]])
    points=[c]
    for longitudinal,lateral in [(0,.3),(0,-.3),(0,.65),(0,-.65),
                                  (-.35,.3),(-.35,-.3),(.35,.3),(.35,-.3)]:
        points.append(c+r*(longitudinal*axis+lateral*perp))
    for fraction in [.25,.5,.75]:
        for lateral in [0,.15,-.15]:
            points.append(witness+fraction*(c-witness)+lateral*r*perp)
    valid=[]
    for q in points:
        if reception_certified(P,q,witness) and np.linalg.norm(q-s)>0.1:
            if all(np.linalg.norm(q-p)>0.1 for p in valid):valid.append(q)
    if not valid:raise RuntimeError('没有可保证接收的候选点')
    return valid


def scenario_points(P):
    # 有限离散评估，不宣称给出连续可行域的最坏情形证明。
    indices=np.linspace(0,len(P)-1,min(5,len(P)),dtype=int)
    return np.vstack((P[indices],P.mean(axis=0)))


def continuation_cost(P,s,g,steps=3):
    """固定后续规则的短期展开。后续读数用零误差；只是排名估计。"""
    total=0.
    for _ in range(steps):
        c,r=mec(P)
        if r<=20-1e-6:return total+np.linalg.norm(c-s)/5+5
        _,(i,j)=diameter(P);v=P[j]-P[i];v/=max(np.linalg.norm(v),1e-10)
        q=c+.3*r*np.array([-v[1],v[0]])
        total+=np.linalg.norm(q-s)/5+5
        if np.linalg.norm(g-q)<=5:return total+5
        angle=math.degrees(math.atan2(*(g-q)[::-1]))%360
        P=wedge(P,q,angle);s=q
    # 未终止展开显式加尾项；属于启发式估计，不是已发生的清除时间。
    c,r=mec(P)
    return total+np.linalg.norm(c-s)/5+5+2*r/5+10


def choose_measure(P,s,policy,witness=None):
    options=candidates(P,s,witness)
    if policy=='midpoint':
        c,r=mec(P);_,(i,j)=diameter(P);v=P[j]-P[i];v/=max(np.linalg.norm(v),1e-10)
        desired=c+.3*r*np.array([-v[1],v[0]])
        return min(options,key=lambda q:np.linalg.norm(q-desired))
    hypotheses=scenario_points(P)
    best=None
    for q in options:
        worst=0.
        for g in hypotheses:
            delta=g-q
            if np.linalg.norm(delta)<=5:
                value=5. if policy=='time' else 10.
                worst=max(worst,value);continue
            bearing=math.degrees(math.atan2(delta[1],delta[0]))
            for err in [-1.,0.,1.]:
                next_P=wedge(P,q,bearing+err)
                if not len(next_P):raise RuntimeError('预测观测产生空集')
                if policy=='geometry':value=diameter(next_P)[0]
                elif policy=='time':value=continuation_cost(next_P,q,g)
                else:raise ValueError(policy)
                worst=max(worst,value)
        movement=np.linalg.norm(q-s)/5+5
        score=worst+movement if policy=='time' else worst
        rank=(float(score),float(movement))
        if best is None or rank<best[0]:best=(rank,q)
    return best[1]


def solve_source(api,channel,first,policy,trace,max_measures=30):
    s=api.position.copy()
    if first['measure_result']=='near':
        return api.clear(s,channel)['clear_result']=='success'
    P=initial_belief(s,first['svd_deg'])
    for step in range(max_measures):
        c,r=mec(P)
        trace.append(dict(channel=channel,step=step,position=s.tolist(),
                          radius_m=r,polygon=P.tolist(),phase='localize'))
        q=nearest_certified_clear(P,s)
        if q is not None:
            result=api.clear(q,channel)
            if result['clear_result']!='success':raise RuntimeError('保证清除失败，需检查模型')
            return True
        q=choose_measure(P,s,policy)
        result=api.measure(q,channel);s=q
        if result['measure_result']=='near':return api.clear(s,channel)['clear_result']=='success'
        if result['measure_result']=='no_signal':raise RuntimeError('保证接收的位置返回无信号')
        P=update_belief(P,s,result['svd_deg'])
    return False


def search_grid(spacing=1600.):
    """保留与目标圆盘相交的三角形的全部顶点。外接半径≤1000。"""
    h=spacing*math.sqrt(3)/2
    coords={(i,j):np.array([spacing*(i+j/2),h*j]) for i in range(-4,5) for j in range(-3,4)}
    triangles=[];active=set()
    def segment_distance(a,b):
        v=b-a;t=np.clip(-(a@v)/(v@v),0,1)
        return np.linalg.norm(a+t*v)
    for i in range(-4,4):
        for j in range(-3,3):
            for keys in [[(i,j),(i+1,j),(i,j+1)],[(i+1,j),(i+1,j+1),(i,j+1)]]:
                T=np.array([coords[k] for k in keys])
                # 原点恰为网格顶点；其余与圆盘相交的三角形必有边/顶点距原点≤1800。
                if min(segment_distance(T[k],T[(k+1)%3]) for k in range(3))<=1800+1e-8:
                    active.update(keys);triangles.append(T.tolist())
    points=np.array([coords[k] for k in sorted(active)])
    # 这些三角形覆盖目标圆盘，等边三角形任意点到最近顶点≤spacing/√3。
    return points,triangles


def coverage_points():
    # 原点 + 半径1200米正六边形。对ρ∈[1000,1800]，与最近环点夹角≤30°。
    # 距离平方为ρ的凸函数，最大值在端点；两端均小于1000²，故连续覆盖。
    a=np.arange(6)*math.pi/3
    return np.vstack(([0.,0.],1200*np.column_stack((np.cos(a),np.sin(a)))))


def solve_multi(api,policy='time',schedule='immediate',trace=None):
    trace=[] if trace is None else trace
    grid=coverage_points();todo=list(range(len(grid)))
    cleared=set();found={};unseen=set(range(1,21))
    while todo:
        index=min(todo,key=lambda i:np.linalg.norm(grid[i]-api.position));todo.remove(index)
        p=grid[index]
        trace.append(dict(phase='scan',position=p.tolist(),remaining_channels=len(unseen)))
        # 同一检测点扫完未知频道，再离开；避免追踪途中漏扫该点。
        channels=sorted(unseen,key=lambda c:(c!=api.channel,c))
        discovered=[]
        for channel in channels:
            result=api.measure(p,channel)
            if result['measure_result']!='no_signal':
                unseen.remove(channel);found[channel]=(p.copy(),result);discovered.append(channel)
        if schedule=='immediate':
            while discovered:
                # 优先处理粗略位置最近的源；只是当前发现批次内的排序。
                def priority(channel):
                    pos,res=found[channel]
                    c=pos if res['measure_result']=='near' else mec(initial_belief(pos,res['svd_deg']))[0]
                    return np.linalg.norm(c-api.position)
                channel=min(discovered,key=priority);discovered.remove(channel)
                pos,res=found.pop(channel)
                # 之前的读数有效；定位器需要携带当时检测点，而不是当前机器狗位置。
                success=solve_source_from_history(api,channel,pos,res,policy,trace)
                if not success:return dict(status='localization_limit',cleared=len(cleared))
                cleared.add(channel)
        if len(cleared)==16:return dict(status='complete_by_count',cleared=16)
    # 每个未发现频道均在全部覆盖点测过；至此可以证明不存在未发现全向源。
    while found:
        def priority(channel):
            pos,res=found[channel]
            c=pos if res['measure_result']=='near' else mec(initial_belief(pos,res['svd_deg']))[0]
            return np.linalg.norm(c-api.position)
        channel=min(found,key=priority);pos,res=found.pop(channel)
        success=solve_source_from_history(api,channel,pos,res,policy,trace)
        if not success:return dict(status='localization_limit',cleared=len(cleared))
        cleared.add(channel)
    return dict(status='complete_by_coverage',cleared=len(cleared),absent_channels=len(unseen),scan_points=len(grid))


def solve_source_from_history(api,channel,first_position,first,policy,trace):
    if first['measure_result']=='near':
        return api.clear(first_position,channel)['clear_result']=='success'
    P=initial_belief(first_position,first['svd_deg']);witness=first_position.copy()
    for step in range(30):
        s=api.position.copy();c,r=mec(P)
        trace.append(dict(channel=channel,step=step,position=s.tolist(),radius_m=r,polygon=P.tolist(),phase='localize'))
        q=nearest_certified_clear(P,s)
        if q is not None:
            if api.clear(q,channel)['clear_result']!='success':raise RuntimeError('保证清除失败')
            return True
        q=choose_measure(P,s,policy,witness);result=api.measure(q,channel)
        if result['measure_result']=='near':return api.clear(q,channel)['clear_result']=='success'
        if result['measure_result']=='no_signal':raise RuntimeError('保证接收失败')
        P=update_belief(P,q,result['svd_deg']);witness=q.copy()
    return False
