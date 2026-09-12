"""Finite pointwise coverage screening only; no official calls or solver edits."""
import json, math, pathlib, time
import numpy as np

ROOT=pathlib.Path(__file__).parent

def stations(m,n,r,phase=0,center=False):
    a=np.arange(m)*2*np.pi/m
    b=np.arange(n)*2*np.pi/n+phase
    ro=(1800.0001)/np.cos(np.pi/m)
    p=np.vstack((ro*np.c_[np.cos(a),np.sin(a)],r*np.c_[np.cos(b),np.sin(b)]))
    if center:p=np.vstack((p,[[0.,0.]]))
    return p

def grid(nang=240,nrad=37):
    a=np.arange(nang)*2*np.pi/nang+.000137
    r=np.linspace(0,1800,nrad)
    return np.vstack((np.zeros((1,2)),(r[1:,None,None]*np.c_[np.cos(a),np.sin(a)][None,:,:]).reshape(-1,2)))

def assess(q,g):
    v=q[None,:,:]-g[:,None,:]
    d=np.linalg.norm(v,axis=2)
    a=np.where(d<=1000+1e-9,np.arctan2(v[:,:,1],v[:,:,0]),10.)
    a.sort(axis=1)
    count=(d<=1000+1e-9).sum(axis=1)
    gaps=np.diff(a,axis=1)
    gaps=np.where(np.arange(len(q)-1)[None,:]<count[:,None]-1,gaps,0)
    last=a[np.arange(len(g)),np.maximum(0,count-1)]
    maxgap=np.maximum(gaps.max(axis=1),a[:,0]+2*np.pi-last)
    maxgap=np.where(d.min(axis=1)<1e-8,0,maxgap)
    maxgap=np.where(count==0,2*np.pi,maxgap)
    bad=maxgap>np.pi+1e-10
    idx=int(np.argmax(maxgap))
    return dict(bad=int(bad.sum()),total=len(g),fraction=float(1-bad.mean()),max_gap_deg=float(maxgap[idx]*180/np.pi),worst_point=g[idx].tolist())

def main():
    t=time.time(); g=grid(); rows=[]
    for m,n in [(12,12),(12,11),(12,10),(12,9),(12,8),(13,10),(13,9),(13,8),(14,9),(14,8),(14,7),(15,8),(15,7),(16,7),(16,6)]:
        for r in [750,825,900,925,950,960,975,1000,1050,1100]:
            for phase in [0.,np.pi/m,np.pi/n]:
                for center in [False,True]:
                    if m+n+center>=25:continue
                    q=stations(m,n,r,phase,center)
                    res=assess(q,g)
                    rows.append(dict(m=m,n=n,r=r,phase=phase,center=center,stations=len(q),**res))
        print(m,n,'best',min((x for x in rows if x['m']==m and x['n']==n),key=lambda x:(x['bad'],x['max_gap_deg'])),flush=True)
    rows.sort(key=lambda x:(x['bad'],x['stations'],x['max_gap_deg']))
    result=dict(description='Finite pointwise screening, not continuous proof',elapsed_s=time.time()-t,grid_size=len(g),best=rows[:40],all_rows=rows)
    (ROOT/'screening.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINISHED',len(rows),time.time()-t,rows[:5],flush=True)

if __name__=='__main__': main()
