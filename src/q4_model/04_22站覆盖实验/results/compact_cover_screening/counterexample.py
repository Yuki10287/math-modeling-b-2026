"""Record local geometric counterexamples to superficially near-complete covers."""
from search import *

q=stations(11,10,1000,0,True)
dense=grid(2160,181)
worst=None;bad=0
for offset in range(0,len(dense),20000):
    a=assess(q,dense[offset:offset+20000]);bad+=a['bad']
    if worst is None or a['max_gap_deg']>worst['max_gap_deg']:worst=a
g=np.array(worst['worst_point']);g*=1799.99/np.linalg.norm(g)
v=q-g;dist=np.linalg.norm(v,axis=1);angles=np.sort(np.arctan2(v[dist<=1000,1],v[dist<=1000,0]))
extended=np.r_[angles,angles[0]+2*np.pi];gap=int(np.argmax(np.diff(extended)))
phi=float((extended[gap]+extended[gap+1])/2);normal=np.array([np.cos(phi),np.sin(phi)])
dots=v@normal
assert np.all((dist>1000)|(dots<0))
order=open_route(q,np.zeros(2))
central_q=stations(11,10,1025,0,True);cg=np.array([10.,0.]);cv=central_q-cg
assert np.all((np.linalg.norm(cv,axis=1)>1000)|(cv[:,0]<0))
out=dict(
    description='Negative research result; these layouts are invalid and must not be integrated.',
    screened_minimal_outer_candidates=sum(1 for m in range(7,12) for n in range(6,12) for center in (False,True) if m+n+center<=22)*13*5,
    screened_expanded_outer_candidates=sum(1 for m in range(8,12) for n in range(7,12) if m+n+1<=22)*11*7*3,
    coarse_points=5401,passes=0,
    boundary_counterexample=dict(layout=dict(m=11,n=10,r=1000,phase=0,center=True),stations=q.tolist(),source=g.tolist(),source_radius=float(np.linalg.norm(g)),direction_deg=float(np.degrees(phi)%360),range_m=1000.,distances=dist.tolist(),direction_dot_products=dots.tolist(),max_dot_within_range=float(dots[dist<=1000].max()),minimum_out_of_range_distance=float(dist[dist>1000].min()),unseen_at_every_station=True,dense_points=len(dense),dense_bad=bad,dense_worst=worst,route_m=route_length(q,np.zeros(2),order)),
    central_counterexample=dict(layout=dict(m=11,n=10,r=1025,phase=0,center=True),source=cg.tolist(),direction_deg=0.,range_m=1000.,within_range_indices=np.flatnonzero(np.linalg.norm(cv,axis=1)<=1000).tolist(),unseen_at_every_station=True),
    conclusion='No screened regular two-ring layout with outer count 7–11, inner count 6–11 and at most 22 stations passed even the coarse necessary pointwise check. This is not a proof that all such layouts or fewer-station covers are impossible. A 99.8% pointwise hit rate can hide valid-source failures. There is no new candidate for continuous certification or full-task integration from this bounded search.')
(HERE/'research_conclusion.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('boundary_counterexample','central_counterexample')},ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out['boundary_counterexample'].items() if k not in ('stations','distances','direction_dot_products')},ensure_ascii=False,indent=2))
