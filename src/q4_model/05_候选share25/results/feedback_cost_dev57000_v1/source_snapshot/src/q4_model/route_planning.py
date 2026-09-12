"""Open route over required scan stops and currently known source proxies."""
import numpy as np


def route_length(points,start,order):
    if not order:return 0.
    return float(np.linalg.norm(points[order[0]]-start)+sum(np.linalg.norm(points[b]-points[a]) for a,b in zip(order,order[1:])))


def open_route(points,start):
    points=np.asarray(points,float);start=np.asarray(start,float)
    if not len(points):return []
    distances=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    initial=np.linalg.norm(points-start,axis=1)
    best=None
    for first in np.argsort(initial,kind='stable')[:min(4,len(points))]:
        order=[int(first)];remaining=set(range(len(points)))-set(order)
        while remaining:
            k=min(remaining,key=lambda k:(distances[order[-1],k],k))
            order.append(k);remaining.remove(k)
        # Open-path 2-opt: endpoint reversal never introduces a return-to-origin cost.
        for sweep in range(12):
            improvement=0.;change=None
            for i in range(len(order)-1):
                for j in range(i+1,len(order)):
                    old=initial[order[i]] if i==0 else distances[order[i-1],order[i]]
                    new=initial[order[j]] if i==0 else distances[order[i-1],order[j]]
                    if j+1<len(order):
                        old+=distances[order[j],order[j+1]]
                        new+=distances[order[i],order[j+1]]
                    if old-new>improvement+1e-8:
                        improvement=old-new;change=(i,j)
            if change is None:break
            i,j=change;order[i:j+1]=order[i:j+1][::-1]
        value=route_length(points,start,order)
        rank=(value,order)
        if best is None or rank<best:best=rank
    return best[1]

