# 官方演练与时间诊断：按批次查看

| 材料 | 版本与范围 | 对应文件 |
|---|---|---|
| 旧批5次汇总 | shared，63/63清除；含逐动作、费用、几何和回放核查 | [official_q4_20260911_analysis.json](official_q4_20260911_analysis.json) |
| 旧批目标评估 | 400秒/源目标；适用条件见报告 | [target400_assessment.json](target400_assessment.json) |
| 新批4次汇总 | shared，50/50清除；平均整局6304.11秒，合并504.33秒/源 | [official_q4_20260912_batch4_analysis.json](official_q4_20260912_batch4_analysis.json) |
| 新批第04次回放补充 | 两处等价路线的条件核查；不是严格复现或新成绩 | [official_q4_20260912_batch4_replay_detail.json](official_q4_20260912_batch4_replay_detail.json) |
| 新批条件回放原代码 | 仅作来源快照，复核使用code中的可复用入口 | [历史生成脚本](source_snapshots/batch4_replay/README.md) |
| 本次归档复核 | 分析逐字节一致，补充回放全字段一致，冻结版本核对 | [batch4_organization_check.json](batch4_organization_check.json) |

报告、源代码和本地复核方法见[本阶段导航](../README.md)。原始ZIP和个人客户端日志仅在local_data保存；这里的JSON已去除身份与请求编号。
