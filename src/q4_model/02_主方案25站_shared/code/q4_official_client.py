"""Q4 client entry point. Importing or requesting --help makes no connection."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shared import ROOT, load_file
from polar_cover import PolarCover

# Reuse the frozen, previously tested HTTP transport, not its Q3 solver or stop rule.
transport = load_file('_b_q4_http_transport', ROOT/'q3_model_v2/official_client.py')
ClientError = transport.ClientError


class Q4Client(transport.OfficialClient):
    def __init__(self, robot_id, url='http://127.0.0.1:2026', log=None, **kwargs):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or parsed.port is None or not 1 <= parsed.port <= 65535):
            raise ValueError('接口地址必须是 http://127.0.0.1:端口。')
        super().__init__(robot_id, url, log, **kwargs)

    def measure(self, q, channel):
        point, extra = self._action(q, channel)
        result = self.post('/measure', extra)
        self.position = point.copy()
        self.channel = int(channel)
        if result['measure_result'] == 'no_signal':
            if self.channel not in self.cleared_channels:
                self.negative_points[self.channel].append(point.tolist())
        else:
            self.detected_channels.add(self.channel)
        return result


def solver_api(client):
    """Only the task's public action surface is supplied to the decision model."""
    class Public:
        __slots__ = ()
        @property
        def position(self):
            return client.position.copy()
        @property
        def channel(self):
            return client.channel
        def measure(self, q, c):
            return client.measure(q, c)
        def clear(self, q, c):
            return client.clear(q, c)
    return Public()


def _channels(value, name):
    if (not isinstance(value, list)
            or any(type(c) is not int or not 1 <= c <= 20 for c in value)
            or len(value) != len(set(value))):
        raise ClientError(f'{name} 必须是不重复的 1 至 20 整数频道列表。')
    return set(value)


def check_completion(result, api):
    """Recheck absence from accepted observations using an independent hull library."""
    if not isinstance(result, dict):
        raise ClientError('算法返回值必须是对象。')
    if result.get('complete') is not True:
        return False
    cert = result.get('certificate')
    if not isinstance(cert, dict):
        raise ClientError('算法声称完成，却未提供第四问证书。')
    cleared = _channels(cert.get('cleared'), '证书已清除频道')
    if cleared != api.cleared_channels or _channels(result.get('cleared'), '算法已清除频道') != cleared:
        raise ClientError('声称清除的频道与实际成功反馈不符。')
    if len(cleared) > 16:
        raise ClientError('成功清除数量超过题目上限。')
    if cert.get('basis') == 'count_upper_bound':
        if len(cleared) != 16:
            raise ClientError('数量证书需要实际清除 16 个不同频道。')
        return True
    if cert.get('basis') != 'directional_triangle_cover':
        raise ClientError('未知的第四问完成证书。')
    absent = _channels(cert.get('absent'), '无源频道')
    if absent & cleared or absent | cleared != set(range(1, 21)):
        raise ClientError('证书未完整且无重复地划分 20 个频道。')
    if absent & api.detected_channels:
        raise ClientError('已发现而未清除的频道不能宣告无源。')
    mesh = PolarCover()
    try:
        stations = np.asarray(cert['stations'], float)
        triangles = np.asarray(cert['triangles'])
        if (stations.shape != mesh.stations.shape or triangles.shape != mesh.indices.shape
                or not np.allclose(stations, mesh.stations, rtol=0, atol=1e-7)
                or not np.array_equal(triangles, mesh.indices)):
            raise ValueError('mesh mismatch')
    except (ValueError, TypeError, KeyError) as exc:
        raise ClientError('证书网格与第四问已验证网格不符。') from exc
    for c in sorted(absent):
        # Ignore claimed synthetic witnesses; use only the transport's accepted log ledger.
        points = np.asarray(api.negative_points[c], float).reshape(-1, 2)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            raise ClientError(f'频道 {c} 的实际无信号观测不足。')
        for triangle in mesh.triangles:
            eligible = np.max(np.linalg.norm(triangle[:, None, :]-points, axis=2), axis=0) <= 1000-1e-7
            witnesses = points[eligible]
            if len(witnesses) < 3:
                raise ClientError(f'频道 {c} 未覆盖全部三角形。')
            try:
                hull = ConvexHull(witnesses)
            except QhullError as exc:
                raise ClientError(f'频道 {c} 的观测不能形成二维包围。') from exc
            if np.max(triangle @ hull.equations[:, :2].T + hull.equations[:, 2]) > 1e-7:
                raise ClientError(f'频道 {c} 的实际观测未包围目标区域。')
    return True


def source_hashes():
    names = ('shared.py', 'directional_cover.py', 'polar_cover.py', 'localization.py',
             'solver.py', 'joint_solver.py', 'route_planning.py', 'guarded_policy.py',
             'q4_official_client.py')
    paths = [Path(__file__).parent/name for name in names]
    paths += [ROOT/'q3_model_v2/geometry.py', ROOT/'q3_model_v2/official_client.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_session(api, solver=None):
    if solver is None:
        from joint_solver import solve_multi
        solver = solve_multi
    trace = []
    started = time.monotonic()
    original_error = None
    try:
        snapshot = source_hashes()
        api.record(dict(type='session_start', problem=4, variant='shared', source_sha256=snapshot))
        api.enter()
        algorithm_start = time.monotonic()
        result = solver(solver_api(api), variant='shared', trace=trace)
        algorithm_wall = time.monotonic()-algorithm_start
        complete = check_completion(result, api)
        exited = api.exit() if api.can_exit else None
        if source_hashes() != snapshot:
            raise ClientError('运行过程中源文件发生变化，请保留日志核查。')
        summary = dict(result, complete=complete, problem=4, variant='shared',
            source_sha256=snapshot, successful_clear_count=len(api.cleared_channels),
            total_virtual_s=api.time_s,
            average_virtual_s_per_cleared=api.time_s/len(api.cleared_channels) if api.cleared_channels else None,
            exit_accepted=exited is not None and exited.get('accepted') is True,
            solver_wall_time_s=algorithm_wall, local_session_wall_time_s=time.monotonic()-started,
            local_session_wall_time_definition='本地从会话开始至退出反馈的墙钟时间，包含 HTTP 往返及证书核验。',
            official_program_runtime_s=None,
            official_program_runtime_note='正式汇总程序运行时间须从模拟器填写；接口未提供。')
        api.record(dict(type='session_summary', **summary))
        return summary
    except BaseException as exc:
        original_error = exc
        try:
            api.error_event(exc, 'q4_session', local_session_wall_time_s=time.monotonic()-started)
        except Exception as logging_error:
            exc.add_note(f'错误日志保存失败：{logging_error!r}')
        if api.can_exit:
            try:
                api.exit()
            except BaseException as cleanup_error:
                exc.add_note(f'结束测试时另有异常：{cleanup_error!r}')
        raise
    finally:
        try:
            api.save_trace(trace)
        except BaseException as trace_error:
            if original_error is not None:
                original_error.add_note(f'轨迹保存失败：{trace_error!r}')
            else:
                raise
        finally:
            api.close_log()


def main():
    parser = argparse.ArgumentParser(description='第四问 shared 版；由用户先手动启动问题4演练。')
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:2026')
    parser.add_argument('--log', required=True, help='新的 JSONL 日志路径，已有文件不会覆盖。')
    args = parser.parse_args()
    api = Q4Client(args.robot_id, args.url, args.log)
    summary = run_session(api)
    print('第四问运行结束。')
    print('完整清除证书：' + ('通过' if summary['complete'] else '未通过，请检查日志'))
    print(f'成功清除数：{summary["successful_clear_count"]}')
    print(f'总虚拟时间：{summary["total_virtual_s"]:.6f} 秒')
    print('退出请求：' + ('已接受' if summary['exit_accepted'] else '未确认'))
    print(f'指令与反馈日志：{api.log.resolve()}')
    print(f'定位轨迹：{api.trace_path.resolve()}')
    return 0 if summary['complete'] and summary['exit_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
