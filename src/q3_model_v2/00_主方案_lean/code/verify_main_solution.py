"""Read-only check of the selected Q3 release; never imports or runs its clients."""
import ast
import hashlib
import json
from pathlib import Path


def check(root):
    manifest = json.loads((root / "results/main_solution_freeze.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["file_sha256"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"缺少文件：{name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"文件与冻结版本不同：{name}")
    for filename, function in [("solver.py", "solve_multi"), ("official_client.py", "run_session")]:
        try:
            tree = ast.parse((root / filename).read_text(encoding="utf-8-sig"))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], node.args.defaults))
            defaults.update(zip([a.arg for a in node.args.kwonlyargs], node.args.kw_defaults))
            if ast.literal_eval(defaults["schedule"]) != "lean" or ast.literal_eval(defaults["policy"]) != "time":
                failures.append(f"主配置发生变化：{filename}:{function}")
        except (OSError, SyntaxError, StopIteration, KeyError, ValueError, TypeError) as error:
            failures.append(f"无法检查默认配置：{filename} ({type(error).__name__})")
    return manifest, failures


if __name__ == "__main__":
    try:
        manifest, failures = check(Path(__file__).resolve().parent)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"无法读取冻结记录：{error}")
    if failures:
        print("主版本校验未通过：\n" + "\n".join(failures))
        raise SystemExit(1)
    print(f"主版本校验通过：{manifest['chosen_schedule']}，{len(manifest['file_sha256'])}个文件指纹一致，默认配置为time/lean。")
    print("仅核对文件，未运行求解器，未连接官方模拟器。")
