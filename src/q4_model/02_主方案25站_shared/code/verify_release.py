"""Read-only release checks; never import a client or connect to a simulator."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'q4_model/results/client_release.json').read_text(encoding='utf-8'))
    problems = []
    for name, expected in manifest['source_sha256'].items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(name)
    if problems:
        print('第四问已验证版本校验未通过：')
        for name in problems:
            print(name)
        print('请先检查改动并完成相应本地验证，再更新发布记录。')
        return 1
    print(f'第四问 shared 版本校验通过：{len(manifest["source_sha256"])} 个文件一致。')
    print('只读取本地文件，未连接或启动模拟器。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
