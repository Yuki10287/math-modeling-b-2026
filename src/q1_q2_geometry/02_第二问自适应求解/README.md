# 第二问：自适应求解

**状态：当前求解。** 输入首测信息、生成测点、接收反馈并更新区域。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [第二问专项验证](reports/第二问专项验证.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 早期记录或顶层验证汇总 | [records/](records/) |
| 可运行的示例输入 | [examples/](examples/) |

主要文件：

- [benchmark_q2.py](code/benchmark_q2.py): 实验或压力测试驱动。
- [q2_solver.py](code/q2_solver.py): 模型或运行依赖。
- [test_q2_solver.py](code/test_q2_solver.py): 本地行为检查。

## 使用了哪些前序成果

[第一问：几何核](../00_第一问几何核/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q12_current -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q12_current --tests
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
