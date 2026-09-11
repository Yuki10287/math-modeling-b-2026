import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle


def load(path):return json.loads(Path(path).read_text())


def aggregate(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in groups.items():
        out.append(dict(zip(keys,key))|dict(cases=len(rs),all_cleared=sum(r['all_cleared'] for r in rs),
                 total_sources=sum(r['source_count'] for r in rs),
                 mean_total_s=float(np.mean([r['total_s'] for r in rs])),
                 median_total_s=float(np.median([r['total_s'] for r in rs])),
                 p95_total_s=float(np.percentile([r['total_s'] for r in rs],95)),
                 mean_per_source_s=float(np.mean([r['average_s'] for r in rs])),
                 mean_measures=float(np.mean([r['counts']['measure'] for r in rs])),
                 mean_runtime_s=float(np.mean([r['runtime_s'] for r in rs])),
                 max_runtime_s=max(r['runtime_s'] for r in rs),
                 belief_violations=sum(len(r['belief_violations']) for r in rs),
                 clear_failures=sum(r['counts']['clear_fail'] for r in rs),
                 mean_time_parts_s={k:float(np.mean([r['time_parts_s'][k] for r in rs])) for k in rs[0]['time_parts_s']}))
    return out


def main():
    single=aggregate(load('single-benchmark/summary.json'),['policy'])
    multi=aggregate(load('multi-benchmark/summary.json'),['policy','schedule'])
    checks=load('checks-results.json');http=load('http-check-results.json')
    s={r['policy']:r for r in single};m={(r['policy'],r['schedule']):r for r in multi}
    improvement=1-m['time','immediate']['mean_total_s']/m['geometry','immediate']['mean_total_s']
    output=dict(single=single,multi=multi,single_time_reduction=1-s['time']['mean_total_s']/s['geometry']['mean_total_s'],
                multi_time_reduction=improvement,checks=checks,http_check=http)
    Path('aggregate.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    font_path=Path('/usr/share/fonts/truetype/droid/DroidSansFallback.ttf')
    if font_path.exists():font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':['Droid Sans Fallback','DejaVu Sans'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    figdir=Path('figures');figdir.mkdir(exist_ok=True)
    order=[('geometry','deferred'),('geometry','immediate'),('time','deferred'),('time','immediate')]
    labels=['几何优先 · 扫完再处理','几何优先 · 扫点后处理','时间优先 · 扫完再处理','时间优先 · 扫点后处理']
    fig,ax=plt.subplots(figsize=(9,4.4));left=np.zeros(4)
    for key,label,color in [('move','移动','#567e89'),('measure','测向','#b0c9be'),('switch','切频','#d1b47d'),('clear_success','清除','#bda2bc')]:
        values=np.array([m[k]['mean_time_parts_s'][key] for k in order])/60
        ax.barh(np.arange(4),values,left=left,label=label,color=color,height=.58);left+=values
    for i,val in enumerate(left):ax.text(val+1,i,f'{val:.1f} 分钟',va='center')
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('平均每局虚拟任务时间（分钟）')
    ax.set_xlim(0,max(left)*1.2);ax.legend(ncol=4,loc='upper left',bbox_to_anchor=(0,-.22),frameon=False)
    ax.set_title('第三问本地测试：60组相同条件，四种组合均全部清除',loc='left',pad=15)
    fig.tight_layout();fig.savefig(figdir/'time-comparison.png',dpi=180);plt.close(fig)
    # 固定展示预先使用的第一组保留测试布局，不挑选最大收益案例。
    fig,axes=plt.subplots(1,2,figsize=(10,4.9),sharex=True,sharey=True)
    for ax,policy,title in zip(axes,['geometry','time'],['几何优先','时间优先']):
        case=load(f'multi-benchmark/multi-1000-smooth-{policy}-immediate.json')
        points=np.array([[0,0]]+[e['position'] for e in case['events']])
        sources=np.array([s0['position'] for s0 in case['sources']])
        ax.add_patch(Circle((0,0),1800,fill=False,color='#aaaaaa',lw=1))
        ax.plot(points[:,0],points[:,1],color='#567e89',lw=1,alpha=.75)
        scan=np.array([x['position'] for x in case['trace'] if x['phase']=='scan'])
        ax.scatter(scan[:,0],scan[:,1],marker='s',facecolors='none',edgecolors='#7d617b',s=35,label='扫描点')
        ax.scatter(sources[:,0],sources[:,1],c='#c98463',marker='x',s=42,label='源真值（仅评估）')
        ax.scatter([0],[0],c='#333333',s=28,label='出发点')
        ax.set_aspect('equal');ax.set_xlim(-2000,2000);ax.set_ylim(-2000,2000)
        ax.set_xticks([-1800,0,1800]);ax.set_yticks([-1800,0,1800]);ax.set_xlabel('东向坐标（米）')
        ax.set_title(f'{title}\n{case["result"]["total_s"]/60:.1f} 分钟，全部清除')
    axes[0].set_ylabel('北向坐标（米）');axes[1].legend(loc='lower left',fontsize=9,frameon=False)
    fig.suptitle('同一案例的完整轨迹：种子1000，平滑误差场',y=1.01)
    fig.tight_layout();fig.savefig(figdir/'paired-trajectories.png',dpi=180,bbox_inches='tight');plt.close(fig)
    lines=['# B题闭环原型本地测试结果','',
      '本轮完成了单源闭环、第三问多源完整搜索清除及HTTP适配器的本地替身验证。未运行官方模拟器；所有虚拟时间均由自建环境计算。','',
      '## 比较设计','',
      '- 单源：40种随机布局 × 3种固定地点误差场 = 120组条件，两种策略各运行一次，共240次。源初始可检测；距离在6米至该源接收半径之间取样。',
      '- 多源：独立的20种布局（种子1000—1019）× 3种误差场 = 60组条件，四种组合各运行一次，共240次。每局10—16个源，布局按目标圆盘面积均匀生成；接收半径在1000—1500米内均匀生成。',
      '- 三种误差场为平滑空间变化、位置哈希及接近误差界的正负跳变；均在同一地点保持固定。它们仅为压力测试条件，不代表官方分布。',
      '- 两种选点策略共享候选点、位置集合、接收保证、清除判据和最多30次局部迭代限制。全部失败记录均保留；均值没有排除困难案例。','',
      '## 单源结果','',
      '| 选点策略 | 成功条件数 | 平均总时间/秒 | 平均测向次数 | 平均本地计算时间/秒 |',
      '|---|---:|---:|---:|---:|']
    for r in single:lines.append(f'| {r["policy"]} | {r["all_cleared"]}/{r["cases"]} | {r["mean_total_s"]:.2f} | {r["mean_measures"]:.2f} | {r["mean_runtime_s"]:.4f} |')
    lines+=['',f'时间优先的平均任务时间减少 {output["single_time_reduction"]:.2%}。它增加少量测向，换取更短的移动路径。','',
      '## 第三问结果','',
      '| 选点 | 调度 | 全部清除 | 每局平均总时间/秒 | 每源平均时间/秒 | 单局本地计算最大值/秒 |',
      '|---|---|---:|---:|---:|---:|']
    for r in multi:lines.append(f'| {r["policy"]} | {r["schedule"]} | {r["all_cleared"]}/{r["cases"]} | {r["mean_total_s"]:.2f} | {r["mean_per_source_s"]:.2f} | {r["max_runtime_s"]:.3f} |')
    lines+=['',
      'geometry：在有限源位置和当前误差场景上，最小化预测的最坏剩余直径。time：用同一组场景，计入移动、测量和固定后续规则的三步展开费用。immediate：在每个扫描点先扫完尚未发现的频道，再处理该批源。deferred：扫完全部覆盖点再集中处理。',
      '',f'时间优先＋扫点后处理相比几何优先的对应方案，平均总时间减少 {improvement:.2%}，在60组条件中全部更快。每种组合对应共804个源实例；这是20种布局重复施加3种误差场，不能称为60种独立布局。',
      '', '每源平均时间采用每局T/该局清除数量，再对各局取算术平均；不与“所有时间相加除以所有源数量”混用。本地计算时间包含策略和进程内环境动作，排除导入和结果落盘，不等同于正式测试的程序运行时间，也未包含真实HTTP延迟。',
      '', '![时间分解](figures/time-comparison.png)','',
      '![示例轨迹](figures/paired-trajectories.png)','',
      '## 为什么七个扫描点足够覆盖全向源','',
      '扫描点为原点及半径1200米正六边形的6个顶点。距原点不超过1000米的源在原点必能检测；其余源与最近环点的方向差不超过30°。当源径向距离ρ在[1000,1800]时，距离平方最多为ρ²+1200²−2400ρ cos30°；该式关于ρ凸，检查两个端点得到最远距离不超过968.902米，小于最小接收半径1000米。',
      '', '因此，每个未发现频道在这七点扫描后都可以获得不存在源的证书。已经发现的频道仍必须清除成功；清除16个源也可依据数量上限提前结束。覆盖证书只适用于全向源。',
      '', '## 校验及边界记录','',
      '- 所有基准实验的可行域均未排除真实源；成功清除判据未产生失败清除。',
      '- 检查了1000米接收边界、5米near、20米清除、同点固定误差、失败3秒、clear不切频及分项计时。',
      '- 48个边界/特殊布局运行全部清除，包括圆域边缘最小接收半径、16源重合、10源位于原点、最后一个源在边缘及角度跨0°。',
      '- 初次边界测试有4次未进入定位：用三角函数构造的“圆上”坐标因浮点误差超出接收半径约1e-13米。已修正测试点生成，环境接收判据和求解策略均未放宽。原始结果仍保留。',
      '- HTTP适配通过本地替身服务清除16源；人工丢弃第一次measure的响应后，客户端复用原请求重试，核对没有重复执行或重复计时。该测试不是官方模拟器验证。',
      '', '## 当前可得出的结论与下一步','',
      '已经有可运行的第三问基线，且在列出的自建案例中无遗漏。实验支持继续研究把未来清除成本纳入选点；目前最好组合约83%的虚拟时间仍用于移动，跨源排序、顺路处理、共享扫描位置及更短的搜索路径值得优先改进。',
      '', '现有结果不能证明最优性或对任意合法场景100%成功。选点评分是离散近似，短期展开后的尾项是启发式，局部30步上限也未给出普适收敛证明。正式比较之前应先接入官方第三问演练并复查反馈与计时。第四问需要另外处理定向遮蔽，不能沿用七点全向覆盖证书。']
    Path('report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('report and figures ready')


if __name__=='__main__':main()
