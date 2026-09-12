"""问题 3 官方演练接口适配器。导入本模块不会连接或启动模拟器。"""
from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import math
import socket
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class ClientError(RuntimeError):
    """接口或运行预算错误。"""


class ActionRejected(ClientError):
    pass


class ResponseProtocolError(ClientError):
    pass


class TransportFailure(ClientError):
    pass


class DeadlineBudgetExceeded(TimeoutError):
    pass


class OfficialClient:
    def __init__(self, robot_id, url="http://127.0.0.1:2026", log=None,
                 http_timeout_s=5.0, max_attempts=3, exit_reserve_s=5.0):
        if log is None:
            raise ValueError("必须指定新的日志路径。")
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("robot_id 必须是非空参赛队号。")
        if not math.isfinite(http_timeout_s) or http_timeout_s <= 0:
            raise ValueError("HTTP 超时时间必须为正数。")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts 必须是正整数。")
        if not math.isfinite(exit_reserve_s) or exit_reserve_s < 0:
            raise ValueError("退出预留时间必须为非负数。")
        self.robot_id = robot_id
        self.url = url.rstrip("/")
        self.position = np.zeros(2)
        self.channel = 1
        self.time_s = 0.0
        self.deadline = math.inf
        self.http_timeout_s = float(http_timeout_s)
        self.max_attempts = max_attempts
        self.exit_reserve_s = float(exit_reserve_s)
        self.active = False
        self.closed = False
        self._exit_safe = True
        self.cleared_channels = set()
        self.detected_channels = set()
        self.negative_scan_indices = {c: set() for c in range(1, 21)}
        self.negative_points = {c: [] for c in range(1, 21)}
        angles = np.arange(6) * math.pi / 3
        self.scan_points = np.vstack(([0., 0.], 1200*np.column_stack((np.cos(angles), np.sin(angles)))))
        self.log = Path(log)
        self.trace_path = Path(str(self.log) + ".beliefs.json")
        # 独占创建两个文件。绝不覆盖旧运行日志或旧轨迹。
        for path in (self.log, self.trace_path):
            if path.exists():
                raise FileExistsError(f"文件已存在，请更换 --log：{path}")
        self._log_stream = self.log.open("x", encoding="utf-8")
        try:
            with self.trace_path.open("x", encoding="utf-8") as stream:
                stream.write("[]\n")
        except BaseException:
            self._log_stream.close()
            raise

    def record(self, record):
        self._log_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._log_stream.flush()

    def error_event(self, exc, phase, **fields):
        self.record({"type": "client_error", "phase": phase,
                     "error_type": type(exc).__name__, "error_message": str(exc), **fields})

    def save_trace(self, trace):
        # 此文件已由本实例独占创建。异常路径同样保存已经获得的轨迹。
        self.trace_path.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n",
                                   encoding="utf-8")

    def close_log(self):
        self._log_stream.close()

    @property
    def can_exit(self):
        return (self.active and not self.closed and self._exit_safe
                and time.monotonic() < self.deadline)

    def _timeout_for(self, path, call_deadline):
        reserve = self.exit_reserve_s if path not in ("/enter", "/exit") else 0.0
        remaining = min(call_deadline, self.deadline - reserve) - time.monotonic()
        if remaining <= 0:
            raise DeadlineBudgetExceeded("本次 HTTP 请求已无安全的现实时间预算。")
        return min(self.http_timeout_s, remaining)

    def _validate_response(self, path, result):
        if not isinstance(result, dict):
            raise ResponseProtocolError("响应必须是 JSON 对象。")
        if result.get("accepted") is not True:
            raise ActionRejected(f"接口未接受动作：{result}")
        value = result.get("virtual_time_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ResponseProtocolError("响应缺少有效的 virtual_time_s。")
        if value < self.time_s - 1e-6:
            raise ResponseProtocolError("已接受动作的虚拟时间发生倒退。")
        if path == "/enter":
            remaining = result.get("remaining_real_duration_s")
            if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                    or not math.isfinite(remaining) or remaining < 0):
                raise ResponseProtocolError("/enter 未返回有效的剩余现实时间。")
        elif path == "/measure":
            kind = result.get("measure_result")
            if kind not in ("direction", "near", "no_signal"):
                raise ResponseProtocolError("未知的 measure_result。")
            if kind == "direction":
                angle = result.get("svd_deg")
                if (isinstance(angle, bool) or not isinstance(angle, (int, float))
                        or not math.isfinite(angle) or not 0 <= angle < 360):
                    raise ResponseProtocolError("示向度缺失或超出 [0, 360)。")
        elif path == "/clear":
            if result.get("clear_result") not in ("success", "no_target_in_range"):
                raise ResponseProtocolError("未知的 clear_result。")
        elif path == "/exit" and result.get("exit_reason") != "user_exit":
            raise ResponseProtocolError("/exit 未返回协议规定的 user_exit。")

    def post(self, path, extra=None):
        if path not in ("/enter", "/measure", "/clear", "/exit"):
            raise ValueError(f"未知指令：{path}")
        if self.closed:
            raise ClientError("已完成退出，不再向关闭的测试发送指令。")
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": str(uuid.uuid4())}
        if extra:
            if set(extra) & set(payload):
                raise ValueError("附加字段不得覆盖身份字段或 request_id。")
            payload.update(extra)
        # 一次新动作只生成一次 ID 和请求体；任何重试复用完全相同的字节。
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        call_deadline = time.monotonic() + self.http_timeout_s * self.max_attempts
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                timeout = self._timeout_for(path, call_deadline)
            except DeadlineBudgetExceeded as exc:
                if last is not None:
                    self._exit_safe = False
                self.error_event(exc, "budget", path=path, request=payload, attempt=attempt)
                if last is not None:
                    raise exc from last
                raise
            try:
                request = Request(self.url + path, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
            except HTTPError as exc:
                self._exit_safe = False
                self.error_event(exc, "http", path=path, request=payload,
                                 attempt=attempt, http_status=exc.code)
                raise
            except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                last = exc
                # 请求可能已经执行但响应丢失。仅允许重试原动作，禁止新动作清理。
                self._exit_safe = False
                self.error_event(exc, "transport", path=path, request=payload, attempt=attempt)
                continue
            except (ValueError, UnicodeError, http.client.HTTPException) as exc:
                self._exit_safe = False
                self.error_event(exc, "response_decode", path=path, request=payload, attempt=attempt)
                raise ResponseProtocolError("响应无法按协议解析；原始异常见日志及异常链。") from exc
            # 正常请求记录保留原有 path/request/response 三字段格式。
            self.record({"path": path, "request": payload, "response": result})
            try:
                self._validate_response(path, result)
            except ClientError as exc:
                self._exit_safe = False
                self.error_event(exc, "response_validation", path=path, request=payload, attempt=attempt)
                raise
            self.time_s = float(result["virtual_time_s"])
            self._exit_safe = True
            return result
        raise TransportFailure("接口连接重试已耗尽，测试可能已关闭；停止发送新动作。") from last

    def enter(self):
        start = time.monotonic()
        result = self.post("/enter")
        # 从发出进入请求前计预算，保守地计入 HTTP 往返及重试时间。
        self.deadline = start + float(result["remaining_real_duration_s"])
        self.active = True
        return result

    @staticmethod
    def _action(q, channel):
        point = np.asarray(q, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any(np.abs(point) > 2_000_000):
            raise ValueError("位置必须是两个有限坐标，绝对值不超过 2000000 米。")
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, np.integer)) or not 1 <= channel <= 20:
            raise ValueError("频道必须是 1 到 20 的整数。")
        return point, {"position": {"x": float(point[0]), "y": float(point[1])}, "channel": int(channel)}

    def measure(self, q, channel):
        point, extra = self._action(q, channel)
        result = self.post("/measure", extra)
        self.position = point.copy()
        self.channel = int(channel)
        if result['measure_result'] == 'no_signal':
            if int(channel) not in self.cleared_channels:
                self.negative_points[int(channel)].append(point.tolist())
            for i, q in enumerate(self.scan_points):
                if np.linalg.norm(point-q) <= 1e-6:
                    self.negative_scan_indices[int(channel)].add(i)
        else:
            self.detected_channels.add(int(channel))
        return result

    def clear(self, q, channel):
        point, extra = self._action(q, channel)
        result = self.post("/clear", extra)
        self.position = point.copy()
        if result["clear_result"] == "success":
            self.cleared_channels.add(int(channel))
        # /clear 的目标频道不改变测向机频道。
        return result

    def exit(self):
        if not self.can_exit:
            raise ClientError("测试未进入、已结束、状态不明或预算已耗尽；跳过 /exit。")
        result = self.post("/exit")
        self.active = False
        self.closed = True
        return result


def _completion(result, api):
    certificate = result.get("completion_certificate")
    if result.get("complete") is not True:
        return False
    if not isinstance(certificate, dict):
        raise ClientError("算法声称全清，但没有提供 completion_certificate。")
    basis = certificate.get("basis")
    if certificate.get('valid') is not True or basis not in ('adaptive_coverage', 'count_upper_bound'):
        raise ClientError("算法的完备证书状态无效。")
    claimed = certificate.get("cleared_channels")
    if not isinstance(claimed, list) or len(claimed) != len(set(claimed)) or set(claimed) != api.cleared_channels:
        raise ClientError("完备证书的已清除频道与接口成功响应不一致。")
    if basis == 'count_upper_bound' and len(api.cleared_channels) != 16:
        raise ClientError("按数量全清需要已成功清除 16 个不同频道。")
    if basis == 'adaptive_coverage':
        absent = certificate.get("absent_channels")
        if (not isinstance(absent, list) or len(absent) != len(set(absent))
                or set(absent) & api.cleared_channels
                or set(absent) | api.cleared_channels != set(range(1, 21))):
            raise ClientError("覆盖完备证书未完整划分 20 个频道。")
        from coverage_model import CoverageTracker
        coverage = CoverageTracker()
        for channel in absent:
            if channel in api.detected_channels:
                raise ClientError('已发现但未清除的频道不能宣告不存在。')
            for q in api.negative_points[channel]:
                coverage.observe(channel, q)
            if not coverage.complete(channel):
                raise ClientError('覆盖证书缺少实际无信号测点的完整区域覆盖。')
    return True


def run_session(api, solver, *, policy="time", schedule="lean", max_local_steps=30):
    """执行已由用户启动的演练；任何异常均保存轨迹并保留原始异常。"""
    trace = []
    start = time.monotonic()
    original_error = None
    try:
        api.enter()
        algorithm_start = time.monotonic()
        result = solver(api, policy=policy, schedule=schedule, trace=trace,
                        max_local_steps=max_local_steps)
        solver_wall_time = time.monotonic() - algorithm_start
        complete = _completion(result, api)
        exit_result = api.exit() if api.can_exit else None
        module_root = Path(__file__).parent
        hashes = {name: hashlib.sha256((module_root/name).read_bytes()).hexdigest()
                  for name in ('baseline_solver.py', 'solver.py', 'recovery.py', 'official_client.py',
                               'geometry.py', 'belief_model.py', 'local_policy.py',
                               'coverage_model.py', 'scan_planning.py', 'v1_solver.py')}
        summary = {**result, "complete": complete,
                   'policy': policy, 'schedule': schedule, 'max_local_steps': max_local_steps,
                   'source_sha256': hashes,
                   "total_virtual_s": api.time_s,
                   "successful_clear_count": len(api.cleared_channels),
                   "average_virtual_s_per_cleared": (api.time_s / len(api.cleared_channels)
                                                     if api.cleared_channels else None),
                   "exit_accepted": exit_result is not None and exit_result.get("accepted") is True,
                   "solver_wall_time_s": solver_wall_time,
                   "local_session_wall_time_s": time.monotonic() - start,
                   "local_session_wall_time_definition": "本地从发送 /enter 前到 /exit 返回后的墙钟时间，包含 HTTP 往返；若未退出则截至本地停止。",
                   "official_program_runtime_s": None,
                   "official_program_runtime_note": "接口不返回正式汇总程序运行时间，请从模拟器测试汇总填写。"}
        api.record({"type": "session_summary", **summary})
        return summary
    except BaseException as exc:
        original_error = exc
        try:
            api.error_event(exc, "session", local_session_wall_time_s=time.monotonic() - start)
        except Exception as logging_error:
            exc.add_note(f"错误日志保存失败：{logging_error!r}")
        # 仅本地算法异常且接口仍确认可用时退出；不向已关闭/状态不明的测试盲发 /exit。
        if api.can_exit:
            try:
                api.exit()
            except BaseException as cleanup_error:
                exc.add_note(f"结束测试时另有异常，原始异常予以保留：{cleanup_error!r}")
        raise
    finally:
        try:
            api.save_trace(trace)
        except BaseException as trace_error:
            if original_error is not None:
                original_error.add_note(f"轨迹保存失败：{trace_error!r}")
            else:
                raise
        finally:
            api.close_log()


def main():
    parser = argparse.ArgumentParser(description="请先在官方模拟器手动启动问题 3 演练，并等待接口就绪。")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:2026")
    parser.add_argument("--policy", choices=["time"], default="time")
    parser.add_argument("--schedule", choices=["lean"], default="lean")
    parser.add_argument("--max-local-steps", type=int, default=30)
    parser.add_argument("--log", required=True, help="本次运行的新 JSONL 文件名，不允许覆盖已有文件。")
    args = parser.parse_args()
    if args.max_local_steps < 1:
        parser.error("--max-local-steps 必须是正整数。")
    from solver import solve_multi
    api = OfficialClient(args.robot_id, args.url, args.log)
    summary = run_session(api, solve_multi, policy=args.policy, schedule=args.schedule,
                          max_local_steps=args.max_local_steps)
    print('本轮第三问运行已结束。')
    print('完整清除证书：' + ('通过' if summary['complete'] else '未通过，请检查日志'))
    print(f"成功清除数：{summary['successful_clear_count']}")
    print(f"总虚拟时间：{summary['total_virtual_s']:.6f} 秒")
    print('退出请求：' + ('已接受' if summary['exit_accepted'] else '未确认'))
    print(f'指令与反馈日志：{api.log.resolve()}')
    print(f'定位轨迹：{api.trace_path.resolve()}')


if __name__ == "__main__":
    main()
