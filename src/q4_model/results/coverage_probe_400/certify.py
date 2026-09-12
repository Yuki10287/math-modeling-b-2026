"""Try finite continuous cell certificates for the 21-station candidate."""
import json, pathlib, time
import sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from probe import stations, grid, assess

ROOT=pathlib.Path(__file__).parent

def certificate(q,t,radius=1000.-1e-6):
    d=np.linalg.norm(t[:,None,:]-q[None,:,:],axis=2).max(axis=0)
    good=np.flatnonzero(d<=radius)
    if len(good)<3:return None
    h=ConvexHull(q[good])
    signed=t@h.equations[:,:2].T+h.equations[:,2]
    if signed.max()>1e-8:return None
    return dict(witnesses=good.tolist(),max_distance=float(d[good].max()),hull_outside_error=float(signed.max()))

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 9
    r=float(sys.argv[2]) if len(sys.argv)>2 else 960.
    q=stations(12,n,r,0,True)
    dense=grid(1440,361)
    worst=None; totalbad=0; worst_required=0; worst_g=None
    for offset in range(0,len(dense),20000):
        g=dense[offset:offset+20000]
        s=assess(q,g); totalbad+=s['bad']
        if worst is None or s['max_gap_deg']>worst['max_gap_deg']:worst=s
    print('dense',len(dense),totalbad,worst,flush=True)
    (ROOT/f'dense-{len(q)}-{r:g}.json').write_text(json.dumps(dict(stations=q.tolist(),count=len(q),inner_radius=r,points=len(dense),bad=totalbad,worst=worst),indent=2),encoding='utf-8')
    if totalbad:return
    # Delaunay triangles tile the full convex hull, which contains the target disk.
    initial=Delaunay(q).simplices
    pending=[(q[t].copy(),0) for t in initial]
    good=[]; failed=[]; start=time.time()
    while pending:
        t,depth=pending.pop()
        c=certificate(q,t)
        if c is not None:
            good.append(dict(vertices=t.tolist(),depth=depth,**c));continue
        if depth>=12:
            failed.append(dict(vertices=t.tolist(),depth=depth));continue
        lens=np.linalg.norm(np.roll(t,-1,axis=0)-t,axis=1)
        i=int(np.argmax(lens));j=(i+1)%3;k=(i+2)%3;mid=(t[i]+t[j])/2
        pending.extend([(np.array([t[i],mid,t[k]]),depth+1),(np.array([mid,t[j],t[k]]),depth+1)])
        if len(good)+len(failed)+len(pending)>200000:
            raise RuntimeError('bounded probe exceeded 200000 cells')
    result=dict(description='Floating-point sufficient cell certificate over whole convex hull; not yet integrated or independently audited.',station_count=len(q),stations=q.tolist(),initial_triangles=initial.tolist(),dense=dict(points=len(dense),bad=totalbad,worst=worst),certified_cells=len(good),failed_cells=len(failed),max_depth=max(x['depth'] for x in good+failed),max_witness_distance=max(x['max_distance'] for x in good),hull_apothem=float(-ConvexHull(q).equations[:,2].min()),elapsed_s=time.time()-start,cells=good,failed=failed)
    (ROOT/f'certificate-{len(q)}-{r:g}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('stations','initial_triangles','cells','failed')},flush=True)

if __name__=='__main__':main()
