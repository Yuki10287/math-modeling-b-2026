"""Fair 25+prune controls from frozen 50000/53000 logs; no solver runs.

Reuse the frozen ordered deletion selector. Retain every measurement and
successful clear, and every retained action's coordinates and outcome.
The independent v2 optical audit is run separately by another entry point.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import numpy as np
import ordered_optical_prune_experiment as frozen

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
ROOT = HERE.parents[3]
BATCHES = {
    'holdout50000': 'station22_ordered_prune_holdout50000',
    'pressure53000': 'station22_ordered_prune_pressure53000',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def action_costs(events):
    parts = dict(move=0., measure=0., switch=0., clear_success=0., clear_fail=0.)
    position, channel = (0., 0.), 1
    for event in events:
        parts['move'] += math.dist(position, event['position']) / 5
        if event['action'] == 'measure':
            parts['measure'] += 5
            parts['switch'] += int(channel != event['channel'])
            channel = event['channel']
        else:
            parts['clear_success' if event['clear_result'] == 'success' else 'clear_fail'] += (
                5 if event['clear_result'] == 'success' else 3)
        position = event['position']
    assert abs(sum(parts.values()) - events[-1]['time_s']) < 1e-6
    return parts


def describe(rows):
    mean = statistics.mean
    variants = ('share25', 'share25_prune', 'station22_prune')
    means = {v: mean(r[v + '_s'] for r in rows) for v in variants}
    comparisons = {}
    for a, b in (('share25', 'share25_prune'), ('share25', 'station22_prune'),
                 ('share25_prune', 'station22_prune')):
        key = a + '_to_' + b
        worst = max(rows, key=lambda r: (r[b + '_s'] - r[a + '_s']) / r[a + '_s'])
        delta = [r[b + '_s'] - r[a + '_s'] for r in rows]
        comparisons[key] = dict(
            reduction_percent=100 * (means[a] - means[b]) / means[a],
            mean_saved_s=means[a] - means[b], faster=sum(d < -1e-6 for d in delta),
            slower=sum(d > 1e-6 for d in delta), ties=sum(abs(d) <= 1e-6 for d in delta),
            worst_case=worst['case'], worst_added_s=worst[b + '_s'] - worst[a + '_s'],
            worst_added_percent=100 * (worst[b + '_s'] - worst[a + '_s']) / worst[a + '_s'])
    return dict(condition_pairs=len(rows), independent_layouts=len({r['seed'] for r in rows}),
                sources=sum(r['source_count'] for r in rows), cleared=sum(r['cleared'] for r in rows),
                means_s=means, comparisons=comparisons,
                percentiles_s={v: {str(p): float(np.percentile([r[v+'_s'] for r in rows], p))
                                   for p in (90, 95)} for v in variants},
                mean_time_parts_s={v: {k: mean(r[v+'_parts_s'][k] for r in rows)
                                      for k in rows[0][v+'_parts_s']} for v in variants},
                removed_actual=sum(r['removed_actual'] for r in rows),
                removed_original_only=sum(r['removed_original_only'] for r in rows),
                additional_from_contraction=sum(r['additional_from_contraction'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    assert not out.exists(), 'new output directory required'
    snapshot = frozen.hashes()
    development = json.loads((RESULTS/'ordered_optical_prune_dev48000/manifest.json').read_text('utf-8'))
    assert snapshot == development['code_hashes'], 'frozen selector or dependency changed'
    snapshot[relative(Path(__file__))] = sha(Path(__file__))
    inputs = {}
    for name in BATCHES.values():
        directory = RESULTS / name
        for path in sorted(directory.glob('*-share25.json')):
            inputs[relative(path)] = sha(path)
            peer = path.with_name(path.name.replace('-share25.json', '-station22_prune.json'))
            inputs[relative(peer)] = sha(peer)
    assert len(inputs) == 40
    out.mkdir(parents=True)
    frozen.write(out/'manifest.json', dict(
        code_hashes=snapshot, input_sha256=inputs, local_only=True, official_contacted=False,
        new_solver_runs=0, selection='frozen constrained_replay; geometry and prior real feedback only',
        batches=BATCHES, python=platform.python_version(),
        evidence='Restricted action-deletion reconstruction, not a new simulator or official result.',
        external_v2_audit='Separate artifact, not claimed by this driver.'))
    summaries = {}
    for batch, source_name in BATCHES.items():
        target = out / batch
        target.mkdir()
        rows = []
        paths = sorted((RESULTS/source_name).glob('*-share25.json'))
        assert len(paths) == 10
        for path in paths:
            case = json.loads(path.read_text('utf-8'))
            peer_path = path.with_name(path.name.replace('-share25.json', '-station22_prune.json'))
            peer = json.loads(peer_path.read_text('utf-8'))
            assert case['sources'] == peer['sources'], 'comparison layouts differ'
            assert case['summary']['audit']['passed'] and peer['summary']['audit']['passed']
            replay = frozen.constrained_replay(case)
            replay.update(input_path=relative(path), input_sha256=sha(path),
                          method='frozen constrained_replay; no new actions or observations')
            assert [e for e in case['events'] if e['action'] == 'measure'] == [
                dict(e, time_s=old['time_s']) for old, e in zip(
                    [e for e in case['events'] if e['action'] == 'measure'],
                    [e for e in replay['events'] if e['action'] == 'measure'])]
            assert len([e for e in replay['events'] if e.get('clear_result') == 'success']) == case['summary']['cleared']
            filename = path.stem + '-replay.json'
            frozen.write(target/filename, replay)
            row = dict(case=path.stem, seed=case['summary']['seed'], population=case['summary']['population'],
                       field=case['summary']['field'], source_count=case['summary']['source_count'],
                       cleared=case['summary']['cleared'], input_path=relative(path), input_sha256=sha(path),
                       combination_input_path=relative(peer_path), combination_input_sha256=sha(peer_path),
                       replay_file=filename, share25_s=replay['original_s'], share25_prune_s=replay['candidate_s'],
                       station22_prune_s=peer['summary']['total_s'],
                       share25_parts_s=action_costs(case['events']),
                       share25_prune_parts_s=action_costs(replay['events']),
                       station22_prune_parts_s=action_costs(peer['events']),
                       saved_s=replay['saved_s'], removed_actual=replay['removed_actual'],
                       removed_original_only=replay['removed_original_only'],
                       additional_from_contraction=replay['additional_from_contraction'])
            rows.append(row)
            print(f'{path.stem}: save={replay["saved_s"]:.3f}s, removed={replay["removed_actual"]}', flush=True)
        summary = dict(passed=True, rows=rows, overall=describe(rows),
                       by_field={field: describe([r for r in rows if r['field'] == field])
                                 for field in sorted({r['field'] for r in rows})},
                       note='Pressure conditions share five layouts; do not count as ten independent layouts.')
        frozen.write(target/'summary.json', summary)
        summaries[batch] = summary['overall']
        print(batch, json.dumps(summary['overall'], ensure_ascii=False), flush=True)
    current = frozen.hashes()
    current[relative(Path(__file__))] = sha(Path(__file__))
    assert current == snapshot
    assert all(sha(ROOT/path) == digest for path, digest in inputs.items())
    frozen.write(out/'summary.json', dict(passed=True, code_and_inputs_stable=True, batches=summaries,
                                        new_solver_runs=0, official_contacted=False))


if __name__ == '__main__':
    main()
