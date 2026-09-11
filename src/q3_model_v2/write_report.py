"""Write the user-facing experiment report from completed frozen evidence."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))


def main():
    data = read('results/holdout_summary.json')
    validation = read('results/validation.json')
    development = read('results/development-final/summary.json')
    replay = read('results/holdout_replay_validation.json')
    assert replay['passed']
    assert validation['passed'] and validation['completed_cases'] == validation['requested_cases']
    assert development['code_hashes_unchanged']
    chosen = data['chosen_schedule']
    groups = data['groups']
    a, b = data['comparisons']['baseline'], data['comparisons']['v1']
    rows = []
    for key, label in [('baseline', '初版'), ('v1', '上一版动态调度'), (chosen, '本轮模型改进版')]:
        g = groups[key]
        rows.append(f"| {label} | {g['mean_total_s']:.2f} | {g['mean_distance_m']:.1f} | {g['mean_measures']:.2f} | {g['clear_failures']/g['runs']:.2f} | {g['all_cleared']}/{g['runs']} |")
    table = '\n'.join(rows)
    worst = b['worst_cases'][0]
    g = groups[chosen]
    interval = b['layout_cluster_bootstrap_95pct_interval']
    fieldtext = '；'.join(f"{name} {b['by_field'][field]['reduction_pct']:.2f}%" for field,name in [('smooth','平滑场'),('hash','位置哈希场'),('extreme','接近误差边界的跳变场')])
    stress_sources = sum(c['source_count'] for c in validation['cases'])
    note = f'''# 本轮模型改进与本地实验结果

本轮默认采用 `{chosen}`。在20种新布局、每种3个固定位置误差场的60组配对条件中，本轮平均总时间比上一版降低 **{b['aggregate_reduction_pct']:.2f}%**，每局平均少 **{b['mean_saved_s']:.2f}秒**；相对初版降低 **{a['aggregate_reduction_pct']:.2f}%**。三个版本均完成所有源的清除。全部为本地测试，没有连接官方模拟器。

## 1. 哪些模型不足得到调整

1. **把未知但固定的接收半径纳入约束。** 同源在w收到、在q收不到，说明 `||g-w|| <= R < ||g-q||`，因此得到垂直平分线一侧的线性限制。正负观测不再各自独立处理。这条改动来自题目物理条件，不需要位置分布假设。
2. **把单源孤立定位改成持续的多源位置估计。** 到达某个位置后，按收益筛选其他频道的观测；其他源若已能保证在当前位置20米内，就顺手清除。每次检测和切频照常计费，只节省移动。
3. **把“先精定位再清除”改成测向与清除动作联合选择。** 允许一组20米光学圆完整覆盖当前候选区域；比较继续测向与这一组有限尝试的时间。每次失败的3秒及移动都计入，失败也会更新位置约束。
4. **允许更短的横向测向距离。** 原候选步长主要随候选区域尺度放大；新增20、40、80米横向偏移候选，并用接收保证筛选。面积与尾部风险评分帮助选择，不声称是官方分布下的真实期望最优。
5. **搜索点由剩余区域决定。** 任意位置的无信号都可积累为排除证据，满足全区域覆盖条件的扫描点可以移动。达到16个已发现频道后，立即停止搜索其他频道；全部清除后才能完成。

具体推导、连续几何证明与实现松弛见 [模型推导与适用边界](模型推导与适用边界.md)。

## 2. 冻结版本的配对结果

留出种子4000—4019，20种独立布局，每种重复使用smooth、hash、extreme三个误差场。源位置按圆域面积均匀生成，接收半径均匀取1000—1500米；这属于自建测试总体，不代表官方生成分布。三版始终使用同布局和同一固定位置误差函数，比较计入最后证明无遗漏的时间。

| 版本 | 平均每局总时间/秒 | 平均移动/米 | 平均检测次数 | 平均失败清除次数 | 全清局数 |
|---|---:|---:|---:|---:|---:|
{table}

- 相对上一版：{b['faster']}组更快、{b['slower']}组更慢、{b['equal']}组持平。三种误差场平均降幅分别为：{fieldtext}。
- 相对初版：{a['faster']}组更快、{a['slower']}组更慢。不同参考版本的降幅均由本次同一批配对条件计算，没有拿不同布局或之前官方演练的绝对时间比较。
- 相对上一版最差一组为种子{worst['seed']}、{worst['field']}：上一版{worst['reference_s']:.2f}秒，本轮{worst['candidate_s']:.2f}秒，多{-worst['saved_s']:.2f}秒。完整保留所有变慢案例。
- 按布局整体重抽样、将同布局三个误差场作为一组，得到平均降幅的95% bootstrap区间约为[{interval[0]:.2f}%, {interval[1]:.2f}%]。它只描述这套自建总体的不确定性，不能据此推断官方成绩。
- 本轮60组共清除{g['source_instances']}个源实例，未触发光学后备的局数为{g['runs'] if g['fallback_count']==0 else '见原始记录'}；累计有意光学失败{g['clear_failures']}次，均按题目3秒收费。最大本地计算耗时{g['max_runtime_s']:.2f}秒/局，不含官方通信时间。

![配对结果及耗时组成](figures/holdout_comparison.png)

轨迹示例按“最接近相对上一版收益中位数”的固定规则选择，没有挑选最佳案例：

![代表布局三版路径](figures/median_trajectory_comparison.png)

## 3. 为什么这仍是针对题目的合理模型

时间目标始终为 `移动距离/5 + 5×检测次数 + 实际切频次数 + 5×成功清除次数 + 3×失败清除次数`。每个真实动作都经过公开接口，决策没有读取真源位置、真实半径或真实源数。

位置区域保持外逼近；负圆仅减去其内接多边形，再对所有剩余分支取凸包，避免选择性丢弃可能位置。光学多圆覆盖有连续区域检查，不只检查原多边形顶点。无遗漏证书使用25800个覆盖整个1800米圆域的闭方格；只有整个方格包含于某次1000米负观测圆内才被排除。

独立验证通过{validation['completed_cases']}/{validation['requested_cases']}个压力布局，共{stress_sources}个源，包含圆域边界、最小接收半径、源密集/重合、远端最后一个频道、16源和额外的量化边界场景。验证器隐藏真值接口，并从实际反馈重新核算时间、负观测覆盖和全过程候选区域包含性。另检验负圆裁切的10298个圆外可行点、1000组固定半径正负约束、异常反馈不入证据，以及发现16个源不能提前宣告清除完成。

全部60个留出案例另做了独立动作重放：7914条反馈与保存记录完全一致；6469个候选多边形、499个完整光学计划及1200份频道证据均通过核验。只重放既有动作，不重新运行策略或调整参数。详细结果见 `results/holdout_replay_validation.json`。

在误差按最近0.01度量化的解释下，v2采用1.0050001度几何余量；原版对照保留原1度设定。七个固定扫描点和152点有限光学覆盖保留为后备，其几何保证与用于排名的面积先验分离。

## 4. 哪些尝试没有直接保留为最终方案

开发过程中“只增加沿途搜索”反而增加检测开销，说明多测并不自动省时。模型逐步加入共享定位、联合光学行动、短横向候选和移动扫描点后，再比较沿途扫描阈值0.15与0.30，最终选取0.30。选择依据和源码冻结时间记录在 `results/selection.json`。

393个单源开发条件中，短横向联合模型将平均剩余单源处理时间从164.76秒降至142.95秒，四种局部策略均全清；这项13.24%的单源降幅不能替代上面的整局比较。最终版本另在30组开发条件上重跑，平均总时间{development['groups'][chosen]['mean_total_s']:.2f}秒，全清30/30，源码指纹稳定。

部分早期开发记录标记 `code_hashes_unchanged=false`，原因是运行期间其他模型文件仍在编辑；它们仅保留为探索过程，不用于宣称冻结版本表现。最终留出180次运行的源码指纹全部一致，测试后没有根据留出结果再调整策略。

## 5. 仍然存在的模型不足

本版仍是滚动启发式求解，并非全局最优：路线以候选区域圆心作代理；局部未来成本只做有限展开；源位置的凸外包会丢失部分排除信息；扫描计划统一以每站最多6秒/未知频道估计检测开销，而实际会跳过无新增覆盖的频道。未来可以在这些明确近似上继续研究，不能把本轮降幅理解成已经达到理论下界。

本轮平均收益不意味着每局都更快，更不等于官方保证。第三问依赖全向辐射，第四问定向源需要重写负观测与搜索覆盖模型；这里不直接推广。

程序与复现说明见 [README](README.md)，机器可读汇总见 [holdout_summary.json](results/holdout_summary.json)，独立检查见 [validation.json](results/validation.json)。
'''
    (ROOT/'本轮改进与实验结果.md').write_text(note, encoding='utf8')
    print(str(ROOT/'本轮改进与实验结果.md'))


if __name__ == '__main__':
    main()
