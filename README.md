# 2026数学建模B题求解

当前推荐第三问入口是 **[`q3_model_v2`](q3_model_v2/README.md)**，默认策略为 `lean`。

| 目录 | 内容 |
|---|---|
| [`q1_q2_geometry`](q1_q2_geometry/README.md) | 第一、二问的几何核与测点实验 |
| [`q3_prototype`](q3_prototype/README.md) | 第三问最初闭环原型，保留作历史对照 |
| [`q3_improved`](q3_improved/README.md) | 上一版动态调度、光学后备与本地验证 |
| [`q3_model_v2`](q3_model_v2/README.md) | 当前模型、官方接口入口和完整本地验证 |

第三问模型说明见 [模型推导与适用边界](q3_model_v2/模型推导与适用边界.md)，本地结果见 [本轮改进与实验结果](q3_model_v2/本轮改进与实验结果.md)。第四问尚未实现，不能直接用第三问程序测试第四问。

## 本地运行

Python 3.11或更新版本。在仓库根目录执行：

```powershell
python -m pip install -r q3_model_v2/requirements.txt
python -X utf8 q3_model_v2/benchmark.py --start-seed 1000 --seeds 1 --fields smooth --schedules baseline,v1,lean --split development --out q3_model_v2/results/my-local-check
python -X utf8 q3_model_v2/validate_model.py --output q3_model_v2/results/my-validation.json
```

输出目录需使用新名字，避免覆盖旧实验。第一、二问另需安装 `q1_q2_geometry/requirements.txt` 中的依赖。

## 手动接入官方第三问

先阅读 [官方测试操作说明](q3_model_v2/官方测试操作说明.md)，再双击 `q3_model_v2/运行第三问.cmd`。由使用者登录模拟器、选择问题3并启动测试，倒计时结束后再确认程序连接。

官方接口适配已通过本地HTTP替身检查。仓库中的成绩均来自自建本地环境，不能当作官方成绩。个人演练日志自动保存在 `q3_model_v2/official_runs`，不会提交到Git。

## 版本同步

本地修改后，先检查改动，再提交和推送：

```powershell
git status
git add .
git diff --cached --stat
git commit -m "描述本次模型或程序改动"
git push
```

在另一台电脑首次使用，克隆仓库；之后在没有未处理本地改动时用 `git pull --ff-only` 获取更新。遇到本地改动或冲突时先处理，不强制覆盖工作。

仓库保留代码、模型说明、图表和可复现的本地实验记录。原始题目附件、个人演练日志、环境缓存及历史压缩包留在本地，由 `.gitignore` 排除。核心模型的冻结校验值在 `q3_model_v2/results/selection.json` 与留出验证记录中保留。
