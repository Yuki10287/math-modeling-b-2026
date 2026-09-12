"""问题二：公开首测 -> 可靠第二测点 -> 二测后几何区域；不包含动作计时或清除。

固定沿用原型的 128 边外逼近、18 候选和有限最坏直径评分。
复用第一问几何核；不导入第三问求解器、环境或客户端。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from . import geometry as core
except ImportError:
    import geometry as core


SIDES = 128
ANGLE_ERROR_DEG = 1.0
POLICIES = ("geometry", "midpoint", "nearest")


def _point(value, name="position"):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"{name} 必须是两个有限数值组成的坐标")
    if any(isinstance(x, (bool, str)) for x in value):
        raise ValueError(f"{name} 必须是数值坐标")
    try:
        p = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值坐标") from exc
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} 必须是两个有限数值组成的坐标")
    # 题设未限制观测点在 Ω 内；这里只排除已超出该几何计算尺度的数值。
    if np.max(np.abs(p)) > 1e7:
        raise ValueError(f"{name} 超出本程序支持的坐标数值范围 ±10000000 米")
    return p


def normalize_observation(observation):
    if not isinstance(observation, dict):
        raise ValueError("观测必须是 JSON 对象")
    status = observation.get("status")
    if status not in ("direction", "near", "no_signal"):
        raise ValueError("status 必须为 direction、near 或 no_signal")
    p = _point(observation.get("position"))
    result = {"position": p.tolist(), "status": status}
    if status == "direction":
        angle = observation.get("angle")
        if isinstance(angle, (bool, str)) or not isinstance(angle, (int, float)):
            raise ValueError("direction 观测必须提供有限数值 angle（度）")
        if not math.isfinite(angle):
            raise ValueError("angle 必须为有限数值")
        result["angle"] = float(angle % 360)
    return result


def clip(P, normal, bound):
    """含新增交点的凸多边形裁剪；同样支持点和线段。"""
    P = np.asarray(P, float).reshape((-1, 2))
    if not len(P):
        return P
    d = P @ normal - bound
    if np.max(d) <= core.EPS:
        return P.copy()
    if np.min(d) > core.EPS:
        return np.empty((0, 2))
    return core.clip_polygon(P, normal, bound).reshape((-1, 2))


def wedge(P, position, angle):
    # 与原型使用相同的角度运算次序，减少对同分候选的舍入扰动。
    lo, hi = math.radians(angle) - math.pi / 180, math.radians(angle) + math.pi / 180
    for n in (np.array([math.sin(lo), -math.cos(lo)]),
              np.array([-math.sin(hi), math.cos(hi)])):
        P = clip(P, n, float(n @ position))
    return P


def disk_clip(P, position, radius):
    if not len(P) or np.max(np.linalg.norm(P-position, axis=1)) <= radius:
        return P.copy()
    for a in np.arange(SIDES) * 2 * math.pi / SIDES:
        n = np.array([math.cos(a), math.sin(a)])
        P = clip(P, n, float(radius + n @ position))
        if not len(P):
            break
    return P


def enclosing_circle(P):
    """先检查直径圆，其余情形复用第一问支撑圆枚举。"""
    D, (i, j) = core.diameter(P)
    center = (P[i] + P[j]) / 2
    radius = float(np.max(np.linalg.norm(P-center, axis=1)))
    if radius <= D / 2 + 1e-7:
        return center, radius
    return core.minimum_enclosing_circle(P)


def region_summary(P):
    if not len(P):
        return None
    D, _ = core.diameter(P)
    c, r = enclosing_circle(P)
    return {
        "vertices": P.tolist(),
        "diameter_m": D,
        "enclosing_circle": {"center": c.tolist(), "radius_m": r},
        "representation": "conservative_convex_polygon",
        "circle_sides": SIDES,
    }


def initial_region(first_observation):
    first = normalize_observation(first_observation)
    if first["status"] == "no_signal":
        raise ValueError("问题二要求首测成功；no_signal 不能据此产生第二测点")
    s = np.asarray(first["position"])
    P = core.outer_disk((0, 0), 1800., SIDES)
    if first["status"] == "near":
        # 先从小圆开始，避免近距离分支的无用大多边形裁剪。
        P = disk_clip(core.outer_disk(s, 5., SIDES), np.zeros(2), 1800.)
    else:
        P = disk_clip(wedge(P, s, first["angle"]), s, 1500.)
    return P


def reception_certificate(P, point, witness):
    """首测成功的证据约束：检查比首点更远的那一半区域。"""
    q, s = np.asarray(point, float), np.asarray(witness, float)
    farther = clip(P, 2 * (q-s), float(q @ q-s @ s))
    maximum = float(np.max(np.linalg.norm(farther-q, axis=1))) if len(farther) else None
    return {
        "certified": maximum is None or maximum <= 999.999,
        "farther_region_vertices": farther.tolist(),
        "farther_max_distance_m": maximum,
        "threshold_m": 999.999,
    }


def candidate_points(P, first_position):
    s = np.asarray(first_position, float)
    D, (i, j) = core.diameter(P)
    if D <= 1e-8:
        return []
    c, r = enclosing_circle(P)
    axis = (P[j]-P[i]) / D
    perp = np.array([-axis[1], axis[0]])
    raw = [c]
    for longitudinal, lateral in ((0, .3), (0, -.3), (0, .65), (0, -.65),
                                  (-.35, .3), (-.35, -.3), (.35, .3), (.35, -.3)):
        raw.append(c + r * (longitudinal*axis + lateral*perp))
    for fraction in (.25, .5, .75):
        for lateral in (0, .15, -.15):
            raw.append(s + fraction*(c-s) + lateral*r*perp)
    result = []
    for q in raw:
        if np.linalg.norm(q-s) <= .1 or any(np.linalg.norm(q-p) <= .1 for p in result):
            continue
        if reception_certificate(P, q, s)["certified"]:
            result.append(q)
    return result


def scenario_points(P):
    indices = np.linspace(0, len(P)-1, min(5, len(P)), dtype=int)
    return [*P[indices], P.mean(axis=0)]


def geometry_score(P, point):
    """有限情景最大预测直径，不是连续不确定集上的精确最坏值。"""
    q = np.asarray(point, float)
    worst = 0.
    for h in scenario_points(P):
        if np.linalg.norm(h-q) <= 5:
            worst = max(worst, 10.)
            continue
        bearing = math.degrees(math.atan2(h[1]-q[1], h[0]-q[0]))
        for error in (-1., 0., 1.):
            posterior = wedge(P, q, bearing+error)
            if not len(posterior):
                raise RuntimeError("预测区域为空：检查几何数值精度")
            worst = max(worst, core.diameter(posterior)[0])
    return worst


def plan_second_measurement(first_observation, policy="geometry"):
    if policy not in POLICIES:
        raise ValueError(f"policy 必须为 {', '.join(POLICIES)}")
    first = normalize_observation(first_observation)
    P = initial_region(first)
    result = {
        "schema": "q2-plan-v1", "status": "ready",
        "first_observation": first, "policy": policy,
        "angle_error_deg": ANGLE_ERROR_DEG,
        "region": region_summary(P),
        "candidate_region": {
            "definition": "intersection_{g in P1} B(g, max(1000, ||g-s1||))",
            "test": "P1 intersect {2(q-s1).g <= ||q||^2-||s1||^2} inside B(q,999.999)",
            "finite_candidates_are_subset": True,
        },
        "candidates": [], "selected_point": None,
    }
    if not len(P):
        result.update(status="inconsistent", reason="首测与源位置范围不相容")
        return result
    if first["status"] == "near":
        result.update(status="near", near_constraint={"center": first["position"], "radius_m": 5.})
        return result
    if result["region"]["diameter_m"] <= 1e-8:
        result.update(status="localized", reason="区域直径小于等于数值退化阈值 1e-8 米")
        return result
    s = np.asarray(first["position"])
    options = candidate_points(P, s)
    if not options:
        result.update(status="no_candidate", reason="当前离散候选构造不足；不代表不存在可行测点")
        return result
    c, r = enclosing_circle(P)
    D, (i, j) = core.diameter(P)
    a = (P[j]-P[i]) / D
    desired = c + .3*r*np.array([-a[1], a[0]])
    ranks = []
    for q in options:
        score, distance = geometry_score(P, q), float(np.linalg.norm(q-s))
        result["candidates"].append({
            "point": q.tolist(), "geometry_score_m": score, "distance_m": distance,
            "receive_certificate": reception_certificate(P, q, s),
        })
        if policy == "geometry":
            ranks.append((score, distance))
        elif policy == "midpoint":
            ranks.append((float(np.linalg.norm(q-desired)),))
        else:
            ranks.append((distance,))
    index = min(range(len(options)), key=lambda k: ranks[k])
    result["selected_point"] = options[index].tolist()
    result["selected_index"] = index
    return result


def update_second_measurement(plan, second_observation):
    if not isinstance(plan, dict) or plan.get("schema") != "q2-plan-v1" or plan.get("status") != "ready":
        raise ValueError("update 需要 status=ready 的 q2-plan-v1 计划")
    if plan.get("angle_error_deg") != ANGLE_ERROR_DEG:
        raise ValueError("计划的角误差配置与当前程序不一致")
    first = normalize_observation(plan.get("first_observation"))
    if first["status"] != "direction":
        raise ValueError("update 需要正常首测")
    second = normalize_observation(second_observation)
    q = _point(plan.get("selected_point"), "selected_point")
    if np.linalg.norm(q-np.asarray(second["position"])) > 1e-6:
        raise ValueError("二测位置与计划选定位置不一致（容差 1e-6 米）")
    # 从公开首测重建 P1，避免将外部 JSON 中改动过的顶点当作可信几何状态。
    P = initial_region(first)
    if not len(P) or not reception_certificate(P, q, first["position"])["certified"]:
        raise ValueError("计划点未通过首测区域的接收认证")
    D1 = core.diameter(P)[0]
    result = {
        "schema": "q2-result-v1", "status": second["status"],
        "first_observation": first, "second_observation": second,
        "initial_diameter_m": D1, "region": None, "diameter_ratio": None,
    }
    if second["status"] == "no_signal":
        result.update(status="inconsistent", reason="已认证接收的点返回 no_signal；请检查输入或模型假设")
        return result
    if second["status"] == "near":
        posterior = disk_clip(P, q, 5.)
        result["near_constraint"] = {"center": q.tolist(), "radius_m": 5.}
    else:
        posterior = disk_clip(wedge(P, q, second["angle"]), q, 1500.)
    if not len(posterior):
        result.update(status="inconsistent", reason="二测与首测几何区域不相容")
        return result
    result["region"] = region_summary(posterior)
    if D1 > 1e-8:
        result["diameter_ratio"] = result["region"]["diameter_m"] / D1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="读取首测 JSON 并选择第二测点")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--policy", choices=POLICIES, default="geometry")
    update = commands.add_parser("update", help="读取计划和实际二测 JSON，输出几何区域")
    update.add_argument("--plan", type=Path, required=True)
    update.add_argument("--input", type=Path, required=True)
    for sub in (plan, update):
        sub.add_argument("--output", type=Path, required=True, help="新 JSON 路径；拒绝覆盖已有文件")
    args = parser.parse_args(argv)
    try:
        observation = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if args.command == "plan":
            result = plan_second_measurement(observation, args.policy)
        else:
            saved_plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
            result = update_second_measurement(saved_plan, observation)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"输入或计算失败：{exc}\n")
    print(f"{result['status']}: {args.output}")
    return 1 if result["status"] in ("inconsistent", "no_candidate") else 0


if __name__ == "__main__":
    raise SystemExit(main())
