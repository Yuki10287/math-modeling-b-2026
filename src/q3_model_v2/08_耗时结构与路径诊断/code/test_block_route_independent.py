"""Compare block-route DP with independent exhaustive permutation checks.

Synthetic action streams test ordering and cost arithmetic, not geometric
simulator validity. No simulator, hidden source map, or online policy is used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from block_route_diagnostic import solve_blocks


def removal_states(events):
    removed, result = set(), {}
    for event in events:
        result[event['id']] = event['channel'] in removed
        if event.get('clear_result') == 'success':
            removed.add(event['channel'])
    return result


def independent_cost(events):
    position, tuned, total = [0., 0.], 1, 0.
    for event in events:
        total += math.dist(position, event['position']) / 5
        if event['action'] == 'measure':
            total += 5 + (event['channel'] != tuned)
            tuned = event['channel']
        else:
            total += 5 if event['clear_result'] == 'success' else 3
        position = event['position']
    return total


def check(seed=84551, trials=36):
    rng = random.Random(seed)
    comparisons, maximum_error, largest_blocks = 0, 0., 0
    for trial in range(trials):
        count = 5 + (trial % 3 == 0)
        largest_blocks = max(largest_blocks, count)
        blocks, removed, event_id = [], set(), 0
        for _ in range(count):
            events = []
            for _ in range(rng.randint(1, 3)):
                channel = rng.randint(1, 4)
                action = 'measure' if rng.random() < .55 else 'clear'
                event = dict(id=event_id, action=action, channel=channel,
                             position=[rng.uniform(-100, 100), rng.uniform(-100, 100)])
                event_id += 1
                if action == 'clear':
                    event['clear_result'] = 'success' if channel not in removed and rng.random() < .6 else 'no_target_in_range'
                    if event['clear_result'] == 'success':
                        removed.add(channel)
                else:
                    event['measure_result'] = 'no_signal' if channel in removed else 'direction'
                events.append(event)
            blocks.append(dict(actions=events))
        original = [event for block in blocks for event in block['actions']]
        before = removal_states(original)
        sequences = {channel: [e['id'] for e in original if e['channel'] == channel]
                     for channel in range(1, 5)}
        for mode in ('channel_order', 'removal_state'):
            optimum = float('inf')
            for order in itertools.permutations(range(count)):
                events = [event for index in order for event in blocks[index]['actions']]
                feasible = removal_states(events) == before
                if mode == 'channel_order':
                    feasible = feasible and all(
                        [e['id'] for e in events if e['channel'] == channel] == sequences[channel]
                        for channel in range(1, 5))
                if feasible:
                    optimum = min(optimum, independent_cost(events))
            result = solve_blocks(blocks, precedence=mode)
            error = abs(optimum - result['total_s'])
            assert error < 1e-8, (trial, mode, optimum, result['total_s'])
            maximum_error = max(maximum_error, error)
            comparisons += 1
    assert comparisons == 2 * trials
    return dict(seed=seed, random_problems=trials, comparisons=comparisons,
                largest_blocks=largest_blocks, maximum_absolute_error_s=maximum_error,
                all_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    base = Path(__file__).resolve().parent
    result.update(
        official_simulator_contacted=False,
        scope='Independent enumeration of all permutations; direct event-state/order filtering and direct action-cost summation.',
        limitation='Synthetic streams test combinatorial ordering and fees, not physical consistency with source geometry.',
        source_sha256={name: hashlib.sha256((base/name).read_bytes()).hexdigest()
                       for name in ('block_route_diagnostic.py', Path(__file__).name)})
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
