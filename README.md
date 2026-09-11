# 2026数学建模B题求解

**第三问主方案保持 [`q3_model_v2`](q3_model_v2/README.md) 的 `lean` 配置。** 前三问已整理正文草稿；第四问已完成[25站覆盖与联合路线的本地改进](q4_model/README.md)、压力测试及独立接入程序，尚无第四问官方成绩。第三问算法、参数和手动启动入口保持此前版本。

优先阅读：

- [方案冻结与版本选择](方案冻结与版本选择.md)：采用哪个版本、选择依据及时间口径。
- [前三问模型与验证整理稿](论文整理/前三问模型与验证整理稿.md)：推导、算法、实验与局限的正文草稿。
- [证据索引与交付清单](论文整理/证据索引与交付清单.md)：数据出处、可复核项及待补材料。

lean在60组本地留出条件中全部清除，平均总时间3215.22秒，相对上一版v1下降11.30%；合并单源时间234.69秒。用户提供的5次官方演练日志均通过现有资料可支持的核查，共清除63源，合并单源时间239.29秒，逐局单源时间的平均为246.30秒。

| 目录 | 内容 |
|---|---|
| [`q1_q2_geometry`](q1_q2_geometry/README.md) | 第一、二问的几何核与测点实验 |
| [`q3_prototype`](q3_prototype/README.md) | 第三问最初闭环原型，保留作历史对照 |
| [`q3_improved`](q3_improved/README.md) | 上一版动态调度、光学后备与本地验证 |
| [`q3_model_v2`](q3_model_v2/README.md) | 当前模型、官方接口入口和完整本地验证 |
| [`q4_model`](q4_model/README.md) | 第四问混合类型、25站方向覆盖、联合路线与本地验证 |

第三问模型说明见 [模型推导与适用边界](q3_model_v2/模型推导与适用边界.md)，本地结果见 [本轮改进与实验结果](q3_model_v2/本轮改进与实验结果.md)。第四问[最新24组配对验证](q4_model/第四问覆盖与联合路线改进.md)全部全清，平均总时间相对第四问初版下降31.12%。不能直接用第三问程序测试第四问。

## 本地运行

Python 3.11或更新版本。在仓库根目录执行：

```powershell
python -X utf8 q3_model_v2/verify_main_solution.py
python -m pip install -r q3_model_v2/requirements.txt
python -X utf8 q3_model_v2/benchmark.py --start-seed 1000 --seeds 1 --fields smooth --schedules baseline,v1,lean --split development --out q3_model_v2/results/my-local-check
python -X utf8 q3_model_v2/validate_model.py --output q3_model_v2/results/my-validation.json
```

第一条只读取文件并检查主版本指纹，不运行求解器、不连接模拟器。实验输出目录需使用新名字，避免覆盖旧记录。第一、二问另需安装 `q1_q2_geometry/requirements.txt` 中的依赖。

## 手动接入官方第三问

先阅读 [官方测试操作说明](q3_model_v2/官方测试操作说明.md)，再双击 `q3_model_v2/运行第三问.cmd`。由使用者登录模拟器、选择问题3并启动测试，倒计时结束后再确认程序连接。

官方接口适配已通过本地HTTP替身检查。模型对照成绩来自自建本地环境，不能当作官方成绩；用户提供的五次官方演练另见[演练结果复盘](q3_model_v2/官方演练五次结果分析与下一步.md)。个人演练日志自动保存在 `q3_model_v2/official_runs`，不会提交到Git。

第四问使用独立的[运行第四问入口](q4_model/运行第四问.cmd)，先阅读[第四问操作说明](q4_model/第四问官方演练操作说明.md)。30组压力条件、14项本地HTTP检查和3项阈值检查全部通过，尚未运行第四问官方演练。

## 补充实验

[联合调度](q3_model_v2/联合调度改进与本地验证.md)相对lean平均改善1.09%；[成本预测修正](q3_model_v2/成本预测诊断与本地验证.md)相对联合调度平均变慢0.38%；[全局动作价值](q3_model_v2/全局动作价值实验与本地验证.md)相对联合调度平均改善0.30%。三轮分别使用新布局，不能叠加收益或直接比较跨轮均值。代码及负结果均保留作模型对照，未替换主方案。

## 版本同步

本地修改后，先检查改动并选取本次需要同步的文件，再提交和推送：

```powershell
git status
git add <本次确认的文件或目录>
git diff --cached --stat
git commit -m "描述本次模型或程序改动"
git push
```

尖括号内容需替换为实际路径。在另一台电脑首次使用，克隆仓库；之后在没有未处理本地改动时用 `git pull --ff-only` 获取更新。遇到本地改动或冲突时先处理，不强制覆盖工作。

仓库保留代码、模型说明、图表和可复现的本地实验记录。原始题目附件、个人演练日志、环境缓存及历史压缩包留在本地，由 `.gitignore` 排除。核心模型的冻结校验值在 `q3_model_v2/results/selection.json` 与留出验证记录中保留。
