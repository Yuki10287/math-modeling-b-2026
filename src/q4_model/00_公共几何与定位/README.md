# 第四问：公共几何与定位

**状态：共享基础。** 各版本共用的几何、定位、路线与独立审计。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

本组入口和文件用途见下表；结论以保存的实验汇总为准。

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |

主要文件：

- [directional_cover.py](code/directional_cover.py): 模型或运行依赖。
- [guarded_policy.py](code/guarded_policy.py): 模型或运行依赖。
- [localization.py](code/localization.py): 模型或运行依赖。
- [route_planning.py](code/route_planning.py): 模型或运行依赖。
- [shared.py](code/shared.py): 模型或运行依赖。
- [validation.py](code/validation.py): 模型或运行依赖。

## 使用了哪些前序成果

[第三问：time/lean](../../q3_model_v2/00_主方案_lean/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
