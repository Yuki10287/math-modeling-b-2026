"""Independent Q4 share25 entry; import and --help never connect to a server."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from q4_official_client import Q4Client, ClientError, solver_api, check_completion, transport
from shared import ROOT


VARIANT = 'share25'


def source_hashes():
    """Hash the candidate and the unchanged geometry, transport and proof checker."""
    names = ('shared.py', 'directional_cover.py', 'polar_cover.py', 'localization.py',
             'solver.py', 'task_sharing_solver.py', 'route_planning.py', 'guarded_policy.py',
             'q4_official_client.py', 'q4_share25_client.py')
    paths = [Path(__file__).parent/name for name in names]
    paths += [ROOT/'q3_model_v2/geometry.py', ROOT/'q3_model_v2/official_client.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_session(api, solver=None):
    """Use the frozen share25 policy and the original Q4 completion checker."""
    trace = []
    started = time.monotonic()
    original_error = None
    try:
        snapshot = source_hashes()
        if solver is None:
            from task_sharing_solver import solve_multi
            solver = solve_multi
        api.record(dict(type='session_start', problem=4, variant=VARIANT,
                        source_sha256=snapshot))
        api.enter()
        algorithm_start = time.monotonic()
        result = solver(solver_api(api), variant=VARIANT, trace=trace)
        algorithm_wall = time.monotonic()-algorithm_start
        complete = check_completion(result, api)
        exited = api.exit() if api.can_exit else None
        if source_hashes() != snapshot:
            raise ClientError('运行过程中源文件发生变化，请保留日志核查。')
        summary = dict(result, complete=complete, problem=4, variant=VARIANT,
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
            api.error_event(exc, 'q4_share25_session', local_session_wall_time_s=time.monotonic()-started)
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
    parser = argparse.ArgumentParser(description='第四问 share25 候选；由用户先手动启动问题4演练。')
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:2026')
    parser.add_argument('--log', required=True, help='新的 JSONL 日志路径，已有文件不会覆盖。')
    args = parser.parse_args()
    api = Q4Client(args.robot_id, args.url, args.log)
    summary = run_session(api)
    print('第四问 share25 候选运行结束。')
    print('完整清除证书：' + ('通过' if summary['complete'] else '未通过，请检查日志'))
    print(f'成功清除数：{summary["successful_clear_count"]}')
    print(f'总虚拟时间：{summary["total_virtual_s"]:.6f} 秒')
    print('退出请求：' + ('已接受' if summary['exit_accepted'] else '未确认'))
    print(f'指令与反馈日志：{api.log.resolve()}')
    print(f'定位轨迹：{api.trace_path.resolve()}')
    return 0 if summary['complete'] and summary['exit_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
