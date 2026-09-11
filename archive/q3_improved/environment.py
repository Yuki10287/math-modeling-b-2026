"""仅用于自建验证。误差场是明确指定的测试条件，不代表官方分布。"""
import hashlib
import math
import numpy as np


class LocalArena:
    def __init__(self,sources,seed=0,field='smooth',start=(0.,0.)):
        self._sources={int(x['channel']):dict(x) for x in sources}
        self._removed=set();self._seed=seed;self._field=field
        self.position=np.array(start,float);self.channel=1
        self.time_s=0.;self.distance_m=0.;self.counts=dict(measure=0,switch=0,clear_success=0,clear_fail=0)
        self.events=[]

    def _move(self,q):
        q=np.asarray(q,float)
        if q.shape!=(2,) or not np.all(np.isfinite(q)) or np.any(np.abs(q)>2_000_000):raise ValueError('invalid position')
        d=float(np.linalg.norm(q-self.position));self.time_s+=d/5;self.distance_m+=d;self.position=q.copy()

    def _error(self,q,channel):
        phase=(self._seed*.754877666+channel*.569840291)*2*math.pi
        if self._field=='smooth':return .99*math.sin(q[0]/210+phase)*math.cos(q[1]/170-phase)
        if self._field=='extreme':return .99*(1 if math.sin(q[0]/130+q[1]/110+phase)>=0 else -1)
        if self._field=='constant':return .99
        key=f'{self._seed}|{channel}|{q[0]:.8f}|{q[1]:.8f}'.encode()
        u=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')/2**64
        return 1.98*u-.99

    def measure(self,q,channel):
        self._move(q)
        if channel!=self.channel:self.channel=channel;self.counts['switch']+=1;self.time_s+=1
        self.counts['measure']+=1;self.time_s+=5
        source=self._sources.get(channel);result={'measure_result':'no_signal'}
        if source and channel not in self._removed:
            g=np.array(source['position']);d=np.linalg.norm(g-self.position)
            orientation=source.get('orientation')
            visible=orientation is None or np.dot(self.position-g,[math.cos(orientation),math.sin(orientation)])>=-1e-9
            if d<=source['radius'] and visible:
                if d<=5:result={'measure_result':'near'}
                else:
                    angle=math.degrees(math.atan2(g[1]-self.position[1],g[0]-self.position[0]))%360
                    result={'measure_result':'direction','svd_deg':round((angle+self._error(self.position,channel))%360,2)%360}
        self.events.append(dict(action='measure',position=self.position.tolist(),channel=channel,time_s=self.time_s,**result))
        return result

    def clear(self,q,channel):
        self._move(q)
        source=self._sources.get(channel)
        success=source is not None and channel not in self._removed and np.linalg.norm(np.array(source['position'])-self.position)<=20
        self.time_s+=5 if success else 3
        self.counts['clear_success' if success else 'clear_fail']+=1
        if success:self._removed.add(channel)
        result={'clear_result':'success' if success else 'no_target_in_range'}
        self.events.append(dict(action='clear',position=self.position.tolist(),channel=channel,time_s=self.time_s,**result))
        return result

    def evaluation(self):
        parts={'move':self.distance_m/5,'measure':5*self.counts['measure'],'switch':self.counts['switch'],
               'clear_success':5*self.counts['clear_success'],'clear_fail':3*self.counts['clear_fail']}
        assert abs(sum(parts.values())-self.time_s)<1e-6
        return dict(source_count=len(self._sources),cleared=len(self._removed),all_cleared=len(self._removed)==len(self._sources),
                    total_s=self.time_s,average_s=self.time_s/max(1,len(self._removed)),distance_m=self.distance_m,
                    counts=self.counts.copy(),time_parts_s=parts)


def audit_trace(arena,trace):
    violations=[]
    for row in trace:
        if row.get('phase')!='localize':continue
        g=np.array(arena._sources[row['channel']]['position']);P=np.array(row['polygon'])
        # 多边形由外接正多边形裁切，保持逆时针。
        v=np.roll(P,-1,axis=0)-P;w=g-P
        cross=v[:,0]*w[:,1]-v[:,1]*w[:,0]
        if np.min(cross)<-1e-5:violations.append({'channel':row['channel'],'step':row['step']})
    return violations
