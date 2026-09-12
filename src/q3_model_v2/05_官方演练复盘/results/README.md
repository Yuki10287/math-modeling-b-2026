# 本阶段实验记录

各批次按原名保留。`dev/development` 表示开发，`holdout` 表示留出验证，`stress` 表示压力条件；不把不同批次均值混算。

| 批次或汇总 | 文件 |
|---|---|
| official_practice_20260911_analysis.json | [查看汇总](official_practice_20260911_analysis.json) |
| 2026-09-12正式测试前五次演练 | [完整诊断](official_practice_20260912_batch10_analysis.json) |
| 同批原始反馈独立审计 | [独立核验](official_practice_20260912_batch10_independent_audit.json) |
| 题设与实现的有限边界探针 | [模型核查](model_assumptions_audit_v1.json) |
| 同批成功点导出的乐观性能下界 | [下界诊断](q3_information_lower_bound_v1.json) |

新批用户提供实际源数，63/63全清；两个新JSON分析的是同一批五次演练，不能重复计为十局。跨批汇总仅作同一冻结算法的描述性统计，不能当作配对提速证据。详见[本批报告](../reports/正式测试前五次演练分析与版本建议.md)。
