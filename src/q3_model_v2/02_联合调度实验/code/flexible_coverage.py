"""Coordinate descent with reassignment of overlapping scan responsibilities."""
import numpy as np

from joint_planning import shortest_visit, route_cost
from scan_planning import build_scan_tasks
from v1_solver import rolling_route


def flexible_route(tracker, channels, start, sources):
    scans = build_scan_tasks(tracker, channels, start, sources) if channels else []
    if not scans and not sources:
        return [], 0.
    route, _ = rolling_route(np.asarray(start), scans+sources)
    if not scans:
        return route, route_cost(start, route)
    required = np.any(~tracker._excluded[[tracker._index(c) for c in channels]], axis=0)
    radius = tracker.signal_r-tracker.cell_half_diagonal-1e-9
    for _ in range(3):
        previous_cost = route_cost(start, route)
        # At each coordinate update, other stations keep their actual positions.
        # Only cells they do not cover constrain the moving station. Updating
        # sequentially is essential: moving all stations against stale masks
        # could lose their overlapping cells simultaneously.
        for task in list(route):
            if task['kind'] != 'scan':
                continue
            others = [tracker._mask(s['position']) for s in route if s['kind']=='scan' and s is not task]
            covered = np.any(others, axis=0) if others else np.zeros(tracker.total_cells,dtype=bool)
            unique = required & ~covered
            if not np.any(unique):
                route.remove(task)
                continue
            i = next(i for i,s in enumerate(route) if s is task)
            before = start if i==0 else route[i-1]['position']
            after = route[i+1]['position'] if i+1<len(route) else None
            q = shortest_visit(before,after,tracker.centers[unique],radius,task['position'])
            if np.all(tracker._mask(q)[unique]):
                task['position'] = q.tolist()
        ordered, _ = rolling_route(np.asarray(start), route)
        if route_cost(start,ordered) < route_cost(start,route):
            route = ordered
        if route_cost(start,route) >= previous_cost-1e-5:
            break
    masks = [tracker._mask(s['position']) for s in route if s['kind']=='scan']
    assert masks and np.all(np.any(masks,axis=0)[required])
    return route, route_cost(start,route)
