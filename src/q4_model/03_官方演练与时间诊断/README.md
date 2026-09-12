# 第四问：官方演练与时间诊断

**状态：实测复盘。** 五次官方结果、时间开销及400秒目标分析。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [官方演练五次结果分析](reports/官方演练五次结果分析.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [analyze_official_results.py](code/analyze_official_results.py): 结果分析、核查或绘图。
- [assess_time_target.py](code/assess_time_target.py): 结果分析、核查或绘图。
- [plot_official_results.py](code/plot_official_results.py): 结果分析、核查或绘图。

## 使用了哪些前序成果

[第四问：25站shared](../02_主方案25站_shared/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q4_official -- --help
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
