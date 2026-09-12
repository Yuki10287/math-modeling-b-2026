"""第三问改进实验：公开反馈驱动的定位恢复与滚动访问顺序。

baseline 保持原代码；safe 仅补局部失败恢复；dynamic 联合安排已知源和必访扫描点。
路线用未知源候选区域的包围圆圆心作代理点，不把它当作真实源位置。
"""
from __future__ import annotations

import numpy as np
import baseline_solver as core
from recovery import locate_source


class EvidenceAPI:
    """仅记录已接受的公开反馈，完成证书不读取环境真值。"""
    def __init__(self, api):
        self.api = api
        self.grid = core.coverage_points()
        self.cleared = set()
        self.discovered = set()
        self.negatives = {c: set() for c in range(1, 21)}

    @property
    def position(self):
        return self.api.position

    @property
    def channel(self):
        return self.api.channel

    def measure(self, q, channel):
        result = self.api.measure(q, channel)
        if result.get('accepted', True) is not True:
            raise RuntimeError('检测未被接受，不能更新搜索证据')
        if result['measure_result'] == 'no_signal':
            for i, p in enumerate(self.grid):
                if np.linalg.norm(np.asarray(q) - p) <= 1e-6:
                    self.negatives[channel].add(i)
        else:
            self.discovered.add(channel)
        return result

    def clear(self, q, channel):
        result = self.api.clear(q, channel)
        if result.get('accepted', True) is not True:
            raise RuntimeError('清除未被接受，不能更新完成证据')
        if result['clear_result'] == 'success':
            self.cleared.add(channel)
        return result

    def certificate(self):
        absent = [c for c in range(1, 21)
                  if c not in self.cleared and c not in self.discovered
                  and len(self.negatives[c]) == 7]
        by_count = len(self.cleared) == 16
        by_coverage = len(self.cleared) + len(absent) == 20
        return dict(valid=by_count or by_coverage,
                    basis='count_upper_bound' if by_count else 'seven_point_coverage' if by_coverage else 'incomplete',
                    cleared_channels=sorted(self.cleared), absent_channels=absent,
                    unresolved_channels=sorted(set(range(1, 21)) - self.cleared - set(absent)),
                    negative_scan_indices={str(c): sorted(self.negatives[c]) for c in range(1, 21)},
                    assumptions='问题3：全向固定源，唯一频道，接收半径至少1000米，源位于1800米圆域')


def route_length(start, route, points):
    if not route:
        return 0.
    positions = np.vstack((start, points[route]))
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def improve_route(start, route, points):
    """开放路径2-opt；没有回原点约束。"""
    route = list(route)
    for _ in range(20):
        best_delta, best_pair = -1e-7, None
        for i in range(len(route) - 1):
            before = start if i == 0 else points[route[i - 1]]
            a = points[route[i]]
            for j in range(i + 1, len(route)):
                b = points[route[j]]
                delta = np.linalg.norm(before - b) - np.linalg.norm(before - a)
                if j + 1 < len(route):
                    after = points[route[j + 1]]
                    delta += np.linalg.norm(a - after) - np.linalg.norm(b - after)
                if delta < best_delta:
                    best_delta, best_pair = float(delta), (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        route[i:j + 1] = reversed(route[i:j + 1])
    return route


def rolling_route(start, tasks):
    """有限多起点最近邻 + 2-opt。只优化当前已知任务的代理访问距离。"""
    points = np.array([task['position'] for task in tasks])
    starts = sorted(range(len(tasks)), key=lambda i: (np.linalg.norm(points[i] - start), i))[:3]
    best = None
    for first in starts:
        route = [first]
        remaining = set(range(len(tasks))) - {first}
        while remaining:
            nxt = min(remaining, key=lambda i: (np.linalg.norm(points[i] - points[route[-1]]), i))
            route.append(nxt)
            remaining.remove(nxt)
        route = improve_route(start, route, points)
        rank = (route_length(start, route, points), tuple(route))
        if best is None or rank < best[0]:
            best = (rank, route)
    return [tasks[i] for i in best[1]], best[0][0]


def _scan(api, index, grid, unseen, found, trace):
    p = grid[index]
    trace.append(dict(phase='scan', position=p.tolist(), scan_index=index, remaining_channels=len(unseen)))
    discovered = []
    for channel in sorted(unseen, key=lambda c: (c != api.channel, c)):
        result = api.measure(p, channel)
        if result['measure_result'] != 'no_signal':
            unseen.remove(channel)
            # 优先保留原始正反馈；代理点计算失败不能阻断后续光学恢复。
            if result['measure_result'] == 'near':
                center = p.copy()
            else:
                try:
                    center = core.mec(core.initial_belief(p, result['svd_deg']))[0]
                    if not np.all(np.isfinite(center)):
                        raise ValueError('nonfinite proxy center')
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    a = np.radians(result['svd_deg'])
                    center = p + 750*np.array([np.cos(a), np.sin(a)])
                    trace.append(dict(phase='proxy_recovery', channel=channel, reason=str(exc)))
            found[channel] = dict(position=p.copy(), first=result, center=center)
            discovered.append(channel)
    return discovered


def _service(api, channel, found, policy, trace, max_local_steps):
    item = found[channel]
    success = locate_source(api, channel, item['position'], item['first'],
                            policy=policy, trace=trace, max_steps=max_local_steps)
    if success:
        del found[channel]
    return success


def _solve_safe(api, policy, trace, max_local_steps):
    grid = core.coverage_points()
    todo, found, unseen = list(range(7)), {}, set(range(1, 21))
    while todo:
        index = min(todo, key=lambda i: np.linalg.norm(grid[i] - api.position))
        todo.remove(index)
        discovered = _scan(api, index, grid, unseen, found, trace)
        while discovered:
            channel = min(discovered, key=lambda c: np.linalg.norm(found[c]['center'] - api.position))
            discovered.remove(channel)
            if not _service(api, channel, found, policy, trace, max_local_steps):
                return dict(status='recovery_exhausted')
        if len(api.cleared) == 16:
            return dict(status='complete_by_count')
    return dict(status='complete_by_coverage')


def _solve_dynamic(api, policy, trace, max_local_steps):
    grid = core.coverage_points()
    todo, found, unseen = set(range(7)), {}, set(range(1, 21))
    while todo or found:
        if len(api.cleared) == 16:
            return dict(status='complete_by_count')
        tasks = [dict(kind='scan', key=i, position=grid[i].tolist()) for i in sorted(todo)]
        tasks += [dict(kind='source', key=c, position=found[c]['center'].tolist()) for c in sorted(found)]
        route, distance = rolling_route(api.position, tasks)
        trace.append(dict(phase='plan', position=api.position.tolist(), route=route,
                          estimated_remaining_distance_m=distance))
        task = route[0]
        if task['kind'] == 'scan':
            todo.remove(task['key'])
            _scan(api, task['key'], grid, unseen, found, trace)
        elif not _service(api, task['key'], found, policy, trace, max_local_steps):
            return dict(status='recovery_exhausted')
    return dict(status='complete_by_coverage')


def solve_multi(api, policy='time', schedule='dynamic', trace=None, max_local_steps=30):
    if policy not in ('time', 'geometry', 'midpoint'):
        raise ValueError('unknown policy')
    if schedule not in ('baseline', 'safe', 'dynamic'):
        raise ValueError('unknown schedule')
    if isinstance(max_local_steps, bool) or not isinstance(max_local_steps, int) or max_local_steps < 0:
        raise ValueError('max_local_steps must be a nonnegative integer')
    trace = [] if trace is None else trace
    observed = EvidenceAPI(api)
    if schedule == 'baseline':
        result = core.solve_multi(observed, policy=policy, schedule='immediate', trace=trace)
    elif schedule == 'safe':
        result = _solve_safe(observed, policy, trace, max_local_steps)
    else:
        result = _solve_dynamic(observed, policy, trace, max_local_steps)
    certificate = observed.certificate()
    declared = result['status'] in ('complete_by_count', 'complete_by_coverage')
    complete = declared and certificate['valid']
    if declared and not complete:
        result['status'] = 'certificate_incomplete'
    return {**result, 'complete': complete, 'cleared': len(observed.cleared),
            'completion_certificate': certificate,
            'fallback_count': sum(row.get('phase') == 'fallback_start' for row in trace)}
