# 本地运行与目录维护

在项目根目录使用Python 3.11或更新版本。源码只在各版本或实验的 `code` 中编辑，运行时自动把依赖复制到 `tmp/study_runs/` 的独立目录，保持原导入关系及冻结文件字节。

## 常用命令

```powershell
python -X utf8 tools/check_structure.py
python -X utf8 tools/run_study.py verify
python -X utf8 tools/run_study.py q3_main --tests
python -X utf8 tools/run_study.py q4_share --tests
python -X utf8 tools/run_study.py q12_current --script q2_solver.py -- --help
```

`check_structure.py` 只读核对目录、文档链接、三个入口及冻结文件；`verify` 只核对原发布指纹。`--tests` 运行该组code里的本地检查，不重复执行结果目录里的历史源码快照。

运行其他脚本时，把组README中的实验标识和脚本名填入：

```powershell
python -X utf8 tools/run_study.py q4_share --script benchmark_task_sharing.py -- --help
```

`--` 后面的参数原样交给原脚本。命令的工作目录是项目根目录，所以输入、输出路径按项目根目录填写。程序仍保持原来的参数，不要假定所有历史脚本都支持相同选项。

## 结果保存

建议使用原脚本的 `--out`、`--output` 或 `--output-dir` 参数，指定本实验组results中的新批次目录。具体参数以该脚本帮助为准。

原脚本若默认往自己的目录写结果，工具会把成功运行产生的JSON、CSV、Markdown和图表保存到该组 `results/local_runs/运行编号/`，保留原来的路径层次。已保存的历史结果和冻结证据不会被自动覆盖。失败运行保留在对应tmp运行目录，便于检查。

如果显式给出了项目根目录下的输出路径，原脚本直接写到该处。不要主动指定覆盖历史批次。个人官方日志统一进入 `local_data/official_runs/`，不使用本地实验运行工具启动官方客户端。

## 新增实验

向已有组code加入脚本后，工具会自动发现；同一题目不要使用与其他组重复的模块文件名，避免导入冲突。新增结果文件也会被发现；results根目录的README是导航，不作为实验输入。

若要新建一个独立实验组，在 `src/studies.json` 的groups中登记路径、状态、用途和沿用的logical_root，再在 `run_study.py` 的ENTRIES中登记入口。`files` 保存已有文件迁移前的路径和指纹，用于原始证据溯源；不要为“通过校验”而改写冻结哈希。

`tmp/study_runs/` 是可重新生成的副本，没有需要在里面编辑的源代码。程序全部结束、输出归档后，可以清理其中不再需要的旧运行目录；个人官方日志不在这里。
