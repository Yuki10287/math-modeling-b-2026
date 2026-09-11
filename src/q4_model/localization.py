"""Mixed-type hypothesis scoring with a separate certified optical fallback."""
import math
import numpy as np
from shared import core


def samples(P, count=16):
    # Deterministic quadrature, not a posterior guaranteed by the problem.
    if len(P) == 1:
        return P.copy()
    center = P.mean(axis=0)
    a,b=P-center,np.roll(P,-1,axis=0)-center
    areas = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    if areas.sum() < 1e-9:
        _, (i,j) = core.diameter(P)
        return P[i] + np.linspace(0,1,count)[:,None]*(P[j]-P[i])
    cumulative = np.cumsum(areas)/areas.sum()
    points = []
    for k in range(count):
        t = min(int(np.searchsorted(cumulative,(k+.5)/count)),len(P)-1)
        u = math.sqrt(((k+1)*.61803398875)%1)
        v = ((k+1)*.41421356237)%1
        points.append((1-u)*center+u*((1-v)*P[t]+v*P[(t+1)%len(P)]))
    return np.array(points)


def optical_plan(P, start, quadrature=None):
    """Cover an enclosing oriented rectangle by full small rectangular cells.

    Each cell's half diagonal is <20 m, so coverage is continuous; no sampled
    position or inferred orientation is used to assert clearance completeness.
    """
    _, (i,j) = core.diameter(P)
    u = P[j]-P[i]
    u = u/np.linalg.norm(u) if np.linalg.norm(u)>1e-10 else np.array([1.,0.])
    rotation = np.column_stack((u,[-u[1],u[0]]))
    origin = P.mean(axis=0)
    local = (P-origin)@rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    side = math.sqrt(2)*(20-1e-4)
    nx, ny = np.maximum(1,np.ceil((hi-lo)/side).astype(int))
    x = lo[0]+(np.arange(nx)+.5)*(hi[0]-lo[0])/nx
    y = lo[1]+(np.arange(ny)+.5)*(hi[1]-lo[1])/ny
    # Long-axis snake; four endpoint choices preserve the same cover.
    paths = []
    for ys in (y,y[::-1]):
        for reverse in (False,True):
            line=[]
            for k, yy in enumerate(ys):
                xs = x[::-1] if bool(k%2)^reverse else x
                line.extend((xx,yy) for xx in xs)
            paths.append(origin+np.array(line)@rotation.T)
    points = samples(P) if quadrature is None or not len(quadrature) else quadrature
    best = None
    for path in paths:
        cumulative=np.cumsum(np.linalg.norm(np.diff(np.vstack((start,path)),axis=0),axis=1))/5
        hits=np.linalg.norm(points[:,None,:]-path,axis=2)<=20
        if not hits.any(axis=1).all():
            raise RuntimeError('optical rectangle failed sample sanity check')
        first=hits.argmax(axis=1)
        costs=cumulative[first]+3*first+5
        worst=float(cumulative[-1]+3*(len(path)-1)+5)
        score=.8*float(costs.mean())+.2*worst
        if best is None or score<best['score']:
            best=dict(path=path, score=score, worst_s=worst, mean_s=float(costs.mean()),
                      origin=origin, rotation=rotation, lower=lo, upper=hi,
                      cells=[int(nx),int(ny)],
                      cell_radius=float(np.linalg.norm((hi-lo)/[nx,ny])/2))
    return best


class Belief:
    def __init__(self, q, feedback):
        self.P = core.disk_clip(core.circle_polygon(),q,5.) if feedback['measure_result']=='near' else core.initial_belief(q,feedback['svd_deg'])
        self.positives = [np.asarray(q).copy()]
        self.negatives = []
        self.failed_clears = []
        self.measured = [np.asarray(q).copy()]
        self.near = np.asarray(q).copy() if feedback['measure_result']=='near' else None

    def update(self,q,feedback):
        kind=feedback['measure_result']
        self.measured.append(np.asarray(q).copy())
        if kind=='no_signal':
            self.negatives.append(np.asarray(q).copy())
            # No Q3 disk exclusion or radius bisector is valid here.
            return
        if kind=='near':
            P=core.disk_clip(self.P,q,5.)
            self.near=np.asarray(q).copy()
        else:
            P=core.disk_clip(core.wedge(self.P,q,feedback['svd_deg']),q)
        if not len(P) or not np.isfinite(P).all():
            raise RuntimeError('positive observations conflict')
        self.P=P
        self.positives.append(np.asarray(q).copy())

    def scenarios(self):
        """Discrete position/type/heading scenarios with feasible radius intervals.

        For an assumed heading, positives bound R below; only negatives visible
        under that heading bound R above. This is ranking quadrature only.
        """
        gs=samples(self.P)
        angles=np.arange(24)*2*math.pi/24
        headings=np.column_stack((np.cos(angles),np.sin(angles)))
        rows=[]
        for index,g in enumerate(gs):
            if any(np.linalg.norm(g-q)<=20 for q in self.failed_clears):
                continue
            distances=np.linalg.norm(np.array(self.positives)-g,axis=1)
            lower=max(1000.,float(distances.max()))
            for mode in range(25):
                n=None if mode==0 else headings[mode-1]
                if n is not None and np.any((np.array(self.positives)-g)@n < -1e-8):
                    continue
                upper=1500.
                for q in self.negatives:
                    if n is None or (q-g)@n>=0:
                        upper=min(upper,float(np.linalg.norm(q-g))-1e-7)
                if lower<=upper:
                    # Equal initial type mass, not 24-times directional prior mass.
                    weight=.5 if n is None else .5/24
                    rows.append((index,g,n,(lower+upper)/2,weight))
        return gs,rows


def choose_measure(belief,start,channel,current_channel):
    gs, scenarios=belief.scenarios()
    if not scenarios:
        return None
    center,r=core.mec(belief.P)
    _,(i,j)=core.diameter(belief.P)
    u=belief.P[j]-belief.P[i]
    u=u/np.linalg.norm(u) if np.linalg.norm(u)>1e-9 else np.array([1.,0.])
    v=np.array([-u[1],u[0]])
    # Finite geometrically scaled transverse views; observations may be negative.
    candidates=[center, (center+start)/2]
    for along in (0.,-.35*r):
        for offset in (-100.,-40.,40.,100.):
            candidates.append(center+along*u+offset*v)
    best=None
    total_weight=sum(row[4] for row in scenarios)
    for q in candidates:
        if any(np.linalg.norm(q-p)<=.1 for p in belief.measured):
            continue
        hit_weights=np.zeros(len(gs))
        for index,g,n,radius,weight in scenarios:
            if np.linalg.norm(q-g)<=radius and (n is None or (q-g)@n>=0):
                hit_weights[index]+=weight/total_weight
        probability=float(hit_weights.sum())
        if probability<.05:
            continue
        negative_cost=optical_plan(belief.P,q,gs)['score']
        positive_cost=0.
        for k in np.flatnonzero(hit_weights):
            g=gs[k]
            if np.linalg.norm(g-q)<=5:
                cost=5.
            else:
                angle=math.degrees(math.atan2(g[1]-q[1],g[0]-q[0]))%360
                values=[]
                for error in (-1.,0.,1.):
                    bearing=round((angle+error)%360,2)%360
                    P=core.disk_clip(core.wedge(belief.P,q,bearing),q)
                    if not len(P):
                        raise RuntimeError('empty predictive branch')
                    values.append(optical_plan(P,q)['score'])
                cost=float(np.mean(values))
            positive_cost+=hit_weights[k]*cost
        score=float(np.linalg.norm(q-start))/5+5+int(channel!=current_channel)
        score+=positive_cost+(1-probability)*negative_cost
        if best is None or score<best['score']:
            best=dict(q=q,score=score,signal_mass=probability,scenarios=len(scenarios))
    return best
