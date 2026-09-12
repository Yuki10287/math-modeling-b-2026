"""Q4 share25 with certified ordered optical pruning; independent CLI entry.

Import and --help never connect. The frozen transport and 25-station completion
checker are reused. Only this entry selects the new pruning policy; prior Q4
entries keep their existing behavior.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from q4_official_client import (
    Q4Client as BaseQ4Client, ClientError, solver_api, check_completion, transport,
)
from shared import ROOT


VARIANT = 'share25_prune'
BASE_VARIANT = 'share25'


class Q4Client(BaseQ4Client):
    """Keep validated action feedback separate from the solver's claimed trace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_actions = []

    def measure(self, q, channel):
        reply = super().measure(q, channel)
        event = dict(action='measure', position=self.position.tolist(),
                     channel=int(channel), measure_result=reply['measure_result'],
                     time_s=self.time_s)
        if reply['measure_result'] == 'direction':
            event['svd_deg'] = reply['svd_deg']
        self.accepted_actions.append(event)
        return reply

    def clear(self, q, channel):
        reply = super().clear(q, channel)
        self.accepted_actions.append(dict(
            action='clear', position=self.position.tolist(), channel=int(channel),
            clear_result=reply['clear_result'], time_s=self.time_s))
        return reply


def source_hashes():
    """Fingerprint the complete runtime dependency set, using src-relative paths."""
    names = (
        'shared.py', 'directional_cover.py', 'polar_cover.py', 'localization.py',
        'solver.py', 'route_planning.py', 'guarded_policy.py',
        'negative_region.py', 'ordered_optical_prune.py',
        'negative_region_proof.py', 'prune_proof.py',
        'q4_official_client.py', 'q4_share25_prune_client.py',
    )
    paths = [Path(__file__).parent / name for name in names]
    paths += [ROOT / 'q3_model_v2/geometry.py', ROOT / 'q3_model_v2/official_client.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def run_session(api, solver=None):
    """Run the isolated policy, verify its proofs, and preserve failure evidence."""
    trace = []
    started = time.monotonic()
    original_error = None
    try:
        snapshot = source_hashes()
        if solver is None:
            from ordered_optical_prune import solve_multi
            solver = solve_multi
        from prune_proof import verify_trace
        api.record(dict(type='session_start', problem=4, variant=VARIANT,
                        base_variant=BASE_VARIANT, source_sha256=snapshot))
        api.enter()
        algorithm_start = time.monotonic()
        result = solver(solver_api(api), variant=BASE_VARIANT, trace=trace)
        algorithm_wall = time.monotonic() - algorithm_start
        complete = check_completion(result, api)
        audit_start = time.monotonic()
        if complete:
            try:
                prune_audit = verify_trace(trace, api.accepted_actions)
                if prune_audit.get('passed') is not True:
                    raise ValueError('optical pruning proof audit did not pass')
            except (ValueError, AssertionError, KeyError, TypeError, IndexError) as exc:
                raise ClientError('保序删点证书与实际反馈核验未通过，请保留两份日志。') from exc
        else:
            prune_audit = dict(passed=False, reason='session_incomplete')
        proof_wall = time.monotonic() - audit_start
        exited = api.exit() if api.can_exit else None
        if source_hashes() != snapshot:
            raise ClientError('运行过程中源文件发生变化，请保留日志核查。')
        summary = dict(
            result, complete=complete, problem=4, variant=VARIANT,
            base_variant=BASE_VARIANT, source_sha256=snapshot,
            successful_clear_count=len(api.cleared_channels),
            total_virtual_s=api.time_s,
            average_virtual_s_per_cleared=(api.time_s / len(api.cleared_channels)
                                          if api.cleared_channels else None),
            exit_accepted=exited is not None and exited.get('accepted') is True,
            prune_audit=prune_audit,
            optical_prune_skipped_before_success=prune_audit.get('skipped_before_success'),
            optical_prune_removed_planned=prune_audit.get('removed_planned'),
            solver_wall_time_s=algorithm_wall,
            pruning_proof_check_wall_time_s=proof_wall,
            local_session_wall_time_s=time.monotonic() - started,
            local_session_wall_time_definition='本地从会话开始至退出反馈的墙钟时间，包含 HTTP 往返及证书核验。',
            official_program_runtime_s=None,
            official_program_runtime_note='正式汇总程序运行时间须从模拟器填写；接口未提供。',
        )
        api.record(dict(type='session_summary', **summary))
        return summary
    except BaseException as exc:
        original_error = exc
        try:
            api.error_event(exc, 'q4_share25_prune_session',
                            local_session_wall_time_s=time.monotonic() - started)
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
    parser = argparse.ArgumentParser(
        description='第四问25站＋保序删点候选；由用户先手动启动问题4演练。')
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:2026')
    parser.add_argument('--log', required=True, help='新的 JSONL 日志路径，已有文件不会覆盖。')
    args = parser.parse_args()
    api = Q4Client(args.robot_id, args.url, args.log)
    summary = run_session(api)
    print('第四问25站＋保序删点候选运行结束。')
    print('完整清除证书：' + ('通过' if summary['complete'] else '未通过，请检查日志'))
    print('保序删点核验：' + ('通过' if summary['prune_audit']['passed'] else '未通过'))
    print(f'成功清除数：{summary["successful_clear_count"]}')
    print(f'总虚拟时间：{summary["total_virtual_s"]:.6f} 秒')
    if summary['average_virtual_s_per_cleared'] is not None:
        print(f'平均单源时间：{summary["average_virtual_s_per_cleared"]:.6f} 秒')
    if summary['optical_prune_skipped_before_success'] is not None:
        print(f'成功之前已跳过的必失败点：{summary["optical_prune_skipped_before_success"]}')
    print('退出请求：' + ('已接受' if summary['exit_accepted'] else '未确认'))
    print(f'指令与反馈日志：{api.log.resolve()}')
    print(f'定位及删点证据：{api.trace_path.resolve()}')
    return 0 if summary['complete'] and summary['exit_accepted'] and summary['prune_audit']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
