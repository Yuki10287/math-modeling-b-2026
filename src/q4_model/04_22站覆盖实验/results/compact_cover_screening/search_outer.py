"""Slightly expanded outer rings; screening is not a continuous proof."""
from search import *

start=time.monotonic(); coarse=grid(180,31); rows=[]; passed=[]
for m in range(8,12):
    for n in range(7,12):
        center=True
        if m+n+center>22:continue
        for r in range(900,1151,25):
            for scale in (1.02,1.04,1.06,1.08,1.10,1.12,1.16):
                for fraction in (0.,.5,1.):
                    phase=fraction*np.pi/math.lcm(m,n)
                    q=stations(m,n,r,phase,center);q[:m]*=scale
                    a=assess(q,coarse)
                    row=dict(m=m,n=n,r=r,phase=phase,center=center,outer_scale=scale,stations=len(q),**a);rows.append(row)
                    if a['bad']==0:
                        order=open_route(q,np.zeros(2));row['route_m']=route_length(q,np.zeros(2),order)
                        passed.append(row);print('PASS',row,flush=True)
    print('DONE',m,'elapsed',time.monotonic()-start,'passes',len(passed),flush=True)
passed.sort(key=lambda x:x['route_m']);rows.sort(key=lambda x:(x['bad'],x['stations'],x['max_gap_deg']))
out=dict(elapsed_s=time.monotonic()-start,coarse_points=len(coarse),coarse_passes=passed,best_failed=[x for x in rows if x['bad']][:30])
(HERE/'screening_outer.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for row in passed[:6]:
    q=stations(row['m'],row['n'],row['r'],row['phase'],row['center']);q[:row['m']]*=row['outer_scale'];dense=grid(1440,361)
    bad=0;worst=None
    for offset in range(0,len(dense),20000):
        a=assess(q,dense[offset:offset+20000]);bad+=a['bad']
        if worst is None or a['max_gap_deg']>worst['max_gap_deg']:worst=a
    row['dense_bad']=bad;row['dense_points']=len(dense);row['dense_worst']=worst
    print('DENSE',row,flush=True)
    (HERE/'screening_outer.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
