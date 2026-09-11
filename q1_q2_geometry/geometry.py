"""B题第一问几何核与第二测点示例。依赖 numpy、scipy；不调用正式模拟器。"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

EPS = 1e-8


def wedge_halfplanes(position, bearing_deg, half_width_deg=1.0):
    """返回 A @ g <= b；用叉积处理竖直线及 0/360 度跨界。"""
    low, high = np.deg2rad([bearing_deg-half_width_deg,
                           bearing_deg+half_width_deg])
    A = np.array([[math.sin(low), -math.cos(low)],
                  [-math.sin(high), math.cos(high)]])
    return A, A @ np.asarray(position, float)


def polygon_from_measurements(measurements):
    """纯示向度交会：先用 LP 区分空/无界，再枚举边界交点。

    measurements: [(position, bearing_deg), ...]
    返回 status 及凸包顶点；有限点集也可退化为点或线段。
    """
    if not measurements:
        return {"status": "unbounded", "vertices": []}
    pairs = [wedge_halfplanes(s, a) for s, a in measurements]
    A, b = np.vstack([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    opts = dict(A_ub=A, b_ub=b, bounds=[(None, None)]*2,
                method="highs", options={"primal_feasibility_tolerance": 1e-9})
    feasible = linprog([0, 0], **opts)
    if feasible.status == 2:
        return {"status": "empty", "vertices": []}
    if not feasible.success:
        raise RuntimeError(feasible.message)
    for direction in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        result = linprog(direction, **opts)
        if result.status == 3:
            return {"status": "unbounded", "vertices": []}
        if not result.success:
            raise RuntimeError(result.message)
    vertices = []
    for i, j in itertools.combinations(range(len(b)), 2):
        M = A[[i, j]]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, b[[i, j]])
        if np.all(A @ p <= b + 1e-7):
            if all(np.linalg.norm(p-v) > 1e-7 for v in vertices):
                vertices.append(p)
    P = np.asarray(vertices)
    if len(P) == 0:
        raise RuntimeError("有限非空集未获得顶点；需要检查数值精度")
    if len(P) >= 3 and np.linalg.matrix_rank(P-P.mean(axis=0), tol=1e-7) == 2:
        P = P[ConvexHull(P).vertices]
    elif len(P) > 2:
        i, j = diameter(P)[1]
        P = P[[i, j]]
    return {"status": "bounded", "vertices": P.tolist()}


def diameter(vertices):
    P = np.asarray(vertices, float)
    if len(P) == 0:
        raise ValueError("空集未定义此处的直径")
    d2 = np.sum((P[:, None, :] - P[None, :, :])**2, axis=2)
    ij = np.unravel_index(np.argmax(d2), d2.shape)
    return float(np.sqrt(d2[ij])), tuple(map(int, ij))


def minimum_enclosing_circle(vertices):
    """小规模透明实现：枚举点、两点直径圆及三点外接圆。

    仅为少量顶点的参考实现，最坏 O(n^4)；不是大规模优化版本。
    """
    P = np.asarray(vertices, float)
    if len(P) == 0:
        raise ValueError("最小包围圆需要非空点集")
    origin = P.mean(axis=0)
    Q = P-origin
    best_c, best_r = np.zeros(2), float(np.max(np.linalg.norm(Q, axis=1)))

    def consider(c, r):
        nonlocal best_c, best_r
        actual = float(np.max(np.linalg.norm(Q-c, axis=1)))
        if actual <= r + 1e-7 and actual < best_r:
            best_c, best_r = c, actual

    for p in Q:
        consider(p, 0)
    for p, q in itertools.combinations(Q, 2):
        consider((p+q)/2, float(np.linalg.norm(p-q)/2))
    for p, q, r in itertools.combinations(Q, 3):
        M = 2*np.array([q-p, r-p])
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        c = np.linalg.solve(M, [q@q-p@p, r@r-p@p])
        consider(c, float(np.linalg.norm(c-p)))
    return best_c+origin, best_r


def clip_polygon(vertices, normal, bound):
    P = np.asarray(vertices, float)
    if not len(P):
        return P
    out = []
    previous = P[-1]
    d_previous = previous @ normal-bound
    for current in P:
        d_current = current @ normal-bound
        previous_in, current_in = d_previous <= EPS, d_current <= EPS
        if previous_in != current_in:
            t = d_previous/(d_previous-d_current)
            out.append(previous+t*(current-previous))
        if current_in:
            out.append(current)
        previous, d_previous = current, d_current
    return np.asarray(out)


def clip_wedge(vertices, position, angle):
    A, b = wedge_halfplanes(position, angle)
    for normal, bound in zip(A, b):
        vertices = clip_polygon(vertices, normal, bound)
    return vertices


def outer_disk(center=(0, 0), radius=1500., sides=720):
    """圆盘的外接正多边形，保证不错误删去真实可行点。"""
    a = (np.arange(sides)+.5)*2*np.pi/sides
    return np.asarray(center) + radius/np.cos(np.pi/sides)*np.column_stack((np.cos(a), np.sin(a)))


def clip_disk_outer(vertices, center, radius=1500., sides=720):
    if np.max(np.linalg.norm(vertices-center, axis=1)) <= radius:
        return vertices
    for angle in np.arange(sides)*2*np.pi/sides:
        n = np.array([np.cos(angle), np.sin(angle)])
        vertices = clip_polygon(vertices, n, radius+n @ center)
        if not len(vertices):
            break
    return vertices


def second_point_case(q, target_distance=1000., second_error_deg=0.):
    """首测读数0度；源在(距离,0)；第二测角按两位小数输出。

    第二误差是确定性指定的读数误差。本实验不假定误差服从某分布。
    地点不同的实验属于不同反事实案例；不对同一点重新抽样降噪。
    """
    q = np.asarray(q, float)
    target = np.array([target_distance, 0.])
    P1 = clip_wedge(outer_disk(), [0, 0], 0.)
    true_angle = math.degrees(math.atan2(target[1]-q[1], target[0]-q[0])) % 360
    # 测试误差网格仅用 -0.99/0/0.99，避免四舍五入使预设误差越过±1。
    observed = round((true_angle+second_error_deg) % 360, 2) % 360
    P2 = clip_wedge(P1, q, observed)
    P2 = clip_disk_outer(P2, q)
    center, radius = minimum_enclosing_circle(P2)
    d, _ = diameter(P2)
    dist_move = float(np.linalg.norm(q))
    # 首测已经发生后的新增耗时：走到第二点 + 第二次测向。
    extra_time = dist_move/5+5
    max_receive_dist = float(np.max(np.linalg.norm(P1-q, axis=1)))
    certified = radius <= 20-1e-7
    # 明确采用圆心作为清除点；这不一定是最近的可保证清除位置。
    # 总计包含原点首测5秒、第二次5秒及成功清除5秒，无切频。
    total = dist_move/5 + np.linalg.norm(center-q)/5 + 15 if certified else None
    return dict(point=q.tolist(), target=target.tolist(), observed_deg=observed,
                second_input_error_deg=second_error_deg,
                second_actual_error_deg=(observed-true_angle+180) % 360-180,
                first_polygon=P1.tolist(), polygon=P2.tolist(),
                center=center.tolist(), radius_m=radius, diameter_m=d,
                extra_time_s=extra_time, two_measure_time_s=extra_time+5,
                sufficient_receive=bool(max_receive_dist <= 1000),
                max_first_region_distance_m=max_receive_dist,
                can_guarantee_one_clear=certified,
                total_via_center_s=None if total is None else float(total))


def checks():
    tri = np.array([[0., 0.], [38., 0.], [19., 19*math.sqrt(3)]])
    stations = [([-1000., 0.], 1.),
                ([538., -500*math.sqrt(3)], 121.),
                ([519., 519*math.sqrt(3)], 241.)]
    result = polygon_from_measurements(stations)
    assert result['status'] == 'bounded'
    assert abs(diameter(result['vertices'])[0]-38) < 1e-6
    center, r = minimum_enclosing_circle(result['vertices'])
    assert abs(r-38/math.sqrt(3)) < 1e-6
    assert np.allclose(center, tri.mean(axis=0))
    assert polygon_from_measurements([([0, 0], 0), ([0, 0], 0)])['status'] == 'unbounded'
    assert polygon_from_measurements([([0, 0], 180), ([10, 0], 0)])['status'] == 'empty'
    assert polygon_from_measurements([([0, 0], 0), ([0, 0], 180)])['status'] == 'bounded'
    # 旋转后观测角跨过0度，平移后结果仍应保持直径及半径。
    angle = np.deg2rad(239.5)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = [(R @ s + [300, -500], (a+239.5) % 360) for s, a in stations]
    other = polygon_from_measurements(transformed)
    assert abs(diameter(other['vertices'])[0]-38) < 1e-6
    assert abs(minimum_enclosing_circle(other['vertices'])[1]-r) < 1e-6
    return {"triangle_diameter_m": 38., "triangle_min_radius_m": r,
            "checks": ["可实现三角形反例", "重复测量无界", "矛盾测量空集", "退化点", "旋转平移与角度跨界"]}


def main():
    checked = checks()
    points = [[750, 0], [0, 300], [750, 300], [950, 100], [1000, 100], [1000, 300], [1000, 800]]
    rows = [second_point_case(q) for q in points]
    grid = [second_point_case(q, distance, error)
            for q in points for distance in [600., 1000., 1400.]
            for error in [-.99, 0., .99]]
    for row in rows+grid:
        target, P = np.array(row['target']), np.array(row['polygon'])
        A, b = wedge_halfplanes(row['point'], row['observed_deg'])
        assert np.all(A @ target <= b+1e-7)
        assert np.linalg.norm(target-np.array(row['center'])) <= row['radius_m']+1e-7
        assert row['diameter_m']/2 <= row['radius_m']+1e-7
        assert row['radius_m'] <= row['diameter_m']/np.sqrt(3)+1e-7
        if row['can_guarantee_one_clear']:
            assert np.max(np.linalg.norm(P-row['center'], axis=1)) <= 20
    data = dict(assumptions={"source_type": "omni", "actual_receive_radius_m": 1500,
             "first_position": [0, 0], "first_bearing_deg": 0,
             "first_error_deg": 0, "error_bound_deg": 1,
             "circle_outer_error_m": 1500*(1/np.cos(np.pi/720)-1),
             "near_hole_omitted": "忽略direction的5米排除区，保留保守外包集合",
             "official_simulator": False, "statistical_performance_claim": False},
             checks=checked, examples=rows, sensitivity=grid)
    dest = Path(__file__).with_name('results.json')
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({"checks": checked, "examples": [
        {k: v for k, v in row.items() if k not in ['first_polygon', 'polygon']}
        for row in rows], "sensitivity_cases": len(grid)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
