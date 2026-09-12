# 第三问：早期局部选点

**状态：开发探索。** 保存局部选点、半径与局部费用诊断。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

本组入口和文件用途见下表；结论以保存的实验汇总为准。

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 早期记录或顶层验证汇总 | [records/](records/) |

主要文件：

- [local_candidate_experiment.py](code/local_candidate_experiment.py): 模型或运行依赖。
- [local_checks.py](code/local_checks.py): 模型或运行依赖。

## 使用了哪些前序成果

[第三问：time/lean](../00_主方案_lean/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q3_local -- --help
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
