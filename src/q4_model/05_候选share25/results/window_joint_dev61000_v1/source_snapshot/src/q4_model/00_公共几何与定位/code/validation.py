"""Offline independent feedback, timing, truth-containment and coverage checks."""
import math
import numpy as np
from scipy.spatial import ConvexHull


def polygon_contains(P,g):
    P=np.asarray(P,np.longdouble);g=np.asarray(g,np.longdouble)
    if len(P)==1:return np.linalg.norm(P[0]-g)<=1e-6
    area=abs(np.sum(P[:,0]*np.roll(P[:,1],-1)-P[:,1]*np.roll(P[:,0],-1)))
    if len(P)==2 or area<1e-12:
        distances=np.sum((P[:,None,:]-P[None,:,:])**2,axis=2)
        i,j=np.unravel_index(np.argmax(distances),distances.shape)
        v=P[j]-P[i];length2=np.sum(v*v)
        t=np.clip(np.sum((g-P[i])*v)/length2,0,1) if length2>0 else 0
        return bool(np.linalg.norm(g-(P[i]+t*v))<=1e-6)
    edge=np.roll(P,-1,axis=0)-P
    lengths=np.sqrt(np.sum(edge*edge,axis=1))
    cross=edge[:,0]*(g[1]-P[:,1])-edge[:,1]*(g[0]-P[:,0])
    return bool(np.all(cross>=-1e-6*np.maximum(1,lengths)))


def validate_run(sources,events,trace,result,stations,triangles):
    truth={x['channel']:x for x in sources}
    removed=set();position=np.zeros(2);channel=1;time_s=0.
    negative={c:[] for c in range(1,21)}
    counts=dict(measure=0,switch=0,clear_success=0,clear_fail=0)
    max_time_error=0.
    for event in events:
        q=np.array(event['position']);c=event['channel'];source=truth.get(c)
        time_s+=float(np.linalg.norm(q-position))/5
        exists=source is not None and c not in removed
        distance=float(np.linalg.norm(np.array(source['position'])-q)) if exists else math.inf
        if event['action']=='measure':
            counts['measure']+=1;counts['switch']+=int(c!=channel)
            time_s+=5+int(c!=channel);channel=c
            visible=False
            if exists:
                angle=source.get('orientation')
                visible=angle is None or (q[0]-source['position'][0])*math.cos(angle)+(q[1]-source['position'][1])*math.sin(angle)>=-1e-9
            expected='no_signal' if not exists or not visible or distance>source['radius'] else 'near' if distance<=5 else 'direction'
            assert event['measure_result']==expected, 'feedback mismatch'
            if expected=='no_signal':negative[c].append(q)
            if expected=='direction':
                g=source['position'];bearing=math.degrees(math.atan2(g[1]-q[1],g[0]-q[0]))%360
                error=(event['svd_deg']-bearing+180)%360-180
                assert abs(error)<=1.0050002,'bearing error out of bounds'
        else:
            success=exists and distance<=20
            assert event['clear_result']==('success' if success else 'no_target_in_range')
            time_s+=5 if success else 3
            counts['clear_success' if success else 'clear_fail']+=1
            if success:removed.add(c)
        max_time_error=max(max_time_error,abs(time_s-event['time_s']))
        assert abs(time_s-event['time_s'])<1e-6,'timing mismatch'
        position=q
    polygons=plans=0
    for row in trace:
        if row['phase']=='belief':
            source=truth[row['channel']];g=np.array(source['position']);polygons+=1
            assert polygon_contains(row['polygon'],g),'true source excluded'
            n=None if source.get('orientation') is None else np.array([math.cos(source['orientation']),math.sin(source['orientation'])])
            for p in map(np.array,row['positives']):
                assert np.linalg.norm(g-p)<=source['radius']+1e-6
                assert n is None or (p-g)@n>=-1e-7
            for p in map(np.array,row['negatives']):
                assert np.linalg.norm(g-p)>source['radius']-1e-6 or (n is not None and (p-g)@n<1e-7)
        if row['phase']=='optical_plan':
            plans+=1;P=np.array(row['polygon']);R=np.array(row['rotation']);origin=np.array(row['origin'])
            lo=np.array(row['lower']);hi=np.array(row['upper']);nx,ny=row['cells']
            assert np.allclose(R.T@R,np.eye(2),atol=1e-10)
            local=(P-origin)@R
            assert np.all(local>=lo-1e-7) and np.all(local<=hi+1e-7)
            assert np.linalg.norm((hi-lo)/[nx,ny])/2<20-1e-6
            expected=np.array([[lo[0]+(i+.5)*(hi[0]-lo[0])/nx,lo[1]+(j+.5)*(hi[1]-lo[1])/ny] for i in range(nx) for j in range(ny)])
            expected=origin+expected@R.T;actual=np.array(row['path'])
            assert len(expected)==len(actual)
            assert np.max(np.min(np.linalg.norm(expected[:,None,:]-actual,axis=2),axis=1))<1e-6,'incomplete rectangle cover'
    certificate_checks=0
    assert result['complete'],result
    assert set(result['cleared'])==removed
    assert removed==set(truth),'not all hidden sources cleared'
    cert=result['certificate']
    assert np.allclose(stations,cert['stations']) and np.array_equal(triangles,cert['triangles'])
    if cert['basis']=='count_upper_bound':
        assert len(removed)==16
    else:
        assert len(removed)+len(cert['absent'])==20
        for c in cert['absent']:
            assert c not in truth
            record=cert['channels'][str(c)];points=np.array(record['negative_points'])
            assert all(any(np.linalg.norm(p-q)<=1e-7 for q in negative[c]) for p in points)
            assert len(record['triangle_witnesses'])==len(triangles)
            for t,indices in record['triangle_witnesses'].items():
                T=stations[triangles[int(t)]];w=points[indices]
                assert np.max(np.linalg.norm(T[:,None,:]-w,axis=2))<=1000-1e-7
                # Independent library, not the solver's monotone-chain hull.
                h=ConvexHull(w)
                assert np.max(T@h.equations[:,:2].T+h.equations[:,2])<=1e-7
                certificate_checks+=1
    return dict(passed=True,events=len(events),polygons=polygons,optical_plans=plans,
                triangle_certificates=certificate_checks,max_time_error_s=max_time_error,
                total_s=time_s,counts=counts)
