"""Read-only candidate release verification, using only the standard library."""
import hashlib
import json
from pathlib import Path


def verify_release(project_root=None):
    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root).resolve()
    manifest_path = root/'src/q4_model/results/share25_client_release.json'
    problems = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('problem') != 4 or manifest.get('variant') != 'share25':
            raise ValueError('发布记录不是第四问 share25 候选。')
        for group in ('source_sha256', 'evidence_sha256'):
            items = manifest[group]
            if not isinstance(items, dict) or not items:
                raise ValueError(f'发布记录缺少 {group}。')
            for name, expected in items.items():
                path = (root/name).resolve()
                if not path.is_relative_to(root):
                    problems.append(f'发布记录路径越出项目：{name}')
                elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    problems.append(name)
        # Check the unchanged model against the original pre-holdout selection.
        selection = json.loads((root/'src/q4_model/results/cell_selection.json').read_text(encoding='utf-8'))
        if selection['selected_variant'] != 'share25':
            problems.append('冻结候选配置不符')
        for name, expected in selection['code_hashes'].items():
            release_name = 'src/'+name
            if release_name in manifest['source_sha256'] and manifest['source_sha256'][release_name] != expected:
                problems.append(f'求解依赖与留出前冻结不一致：{name}')
        return manifest, problems
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, problems+[f'发布记录无法核验：{exc}']


def main():
    manifest, problems = verify_release()
    if problems:
        print('第四问 share25 候选校验未通过：')
        for problem in problems:
            print(problem)
        print('尚未连接模拟器；请先核对文件与本地验证记录。')
        return 1
    print(f'第四问 share25 候选校验通过：{len(manifest["source_sha256"])} 个发布文件、'
          f'{len(manifest["evidence_sha256"])} 份验证记录一致。')
    print('仅核对本地文件；未连接或启动模拟器，也不代表已有官方成绩。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
