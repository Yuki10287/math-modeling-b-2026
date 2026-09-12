"""A 25-station continuous directional cover, with the existing certificate."""
import math
import numpy as np
from directional_cover import DirectionalCover


class PolarCover(DirectionalCover):
    def __init__(self):
        # The outer regular dodecagon circumscribes the 1800 m target disk.
        # A staggered inner ring joins it with triangles whose edges are <1000 m.
        self.outer_radius=(1800.+1e-4)/math.cos(math.pi/12)
        self.inner_radius=960.
        a=np.arange(12)*math.pi/6
        outer=self.outer_radius*np.column_stack((np.cos(a),np.sin(a)))
        inner=self.inner_radius*np.column_stack((np.cos(a+math.pi/12),np.sin(a+math.pi/12)))
        self.stations=np.vstack((np.zeros((1,2)),inner,outer))
        triangles=[]
        for i in range(12):
            j=(i+1)%12
            triangles.extend([(0,1+i,1+j),(1+i,13+i,13+j),(1+i,13+j,1+j)])
        self.indices=np.array(triangles)
        self.triangles=self.stations[self.indices]
        self.spacing=float(np.max(np.linalg.norm(np.roll(self.triangles,-1,axis=1)-self.triangles,axis=2)))
        if self.spacing>=1000-1e-5:
            raise ValueError('polar cover violates guaranteed reception radius')
        self.negative={c:[] for c in range(1,21)}
        self.covered={c:np.zeros(len(triangles),bool) for c in range(1,21)}
        self.witnesses={c:{} for c in range(1,21)}

