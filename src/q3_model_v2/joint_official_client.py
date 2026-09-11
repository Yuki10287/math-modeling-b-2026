"""Manual entry for the locally evaluated joint scheduling candidate."""
import argparse
import hashlib
from pathlib import Path

from official_client import OfficialClient, run_session
from joint_solver import solve_multi, DEFAULT_SCHEDULE


def recorded_solver(api, **options):
    result=solve_multi(api, **options)
    root=Path(__file__).resolve().parent
    names=['joint_solver.py','joint_planning.py','flexible_coverage.py','joint_official_client.py']
    result['joint_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    result['solver_entry']='joint_solver.solve_multi'
    return result


def main():
    parser=argparse.ArgumentParser(description='第三问联合调度候选版；须由你先手动启动演练。')
    parser.add_argument('--robot-id',required=True)
    parser.add_argument('--url',default='http://127.0.0.1:2026')
    parser.add_argument('--log',required=True)
    parser.add_argument('--schedule',choices=[DEFAULT_SCHEDULE],default=DEFAULT_SCHEDULE)
    parser.add_argument('--max-local-steps',type=int,default=30)
    args=parser.parse_args()
    if args.max_local_steps<1:parser.error('--max-local-steps 必须是正整数')
    api=OfficialClient(args.robot_id,args.url,args.log)
    result=run_session(api,recorded_solver,schedule=args.schedule,max_local_steps=args.max_local_steps)
    print('联合调度候选版运行结束。')
    print('完整清除证书：'+('通过' if result['complete'] else '未通过，请检查日志'))
    print(f"成功清除数：{result['successful_clear_count']}")
    print(f"总虚拟时间：{result['total_virtual_s']:.6f} 秒")
    print('退出请求：'+('已接受' if result['exit_accepted'] else '未确认'))
    print(f'指令与反馈日志：{api.log.resolve()}')
    print(f'定位轨迹：{api.trace_path.resolve()}')


if __name__=='__main__':main()
