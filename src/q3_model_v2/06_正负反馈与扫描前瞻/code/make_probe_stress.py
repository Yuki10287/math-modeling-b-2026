"""Construct Q3 stress layouts from selected development probes; never run a solver.

These cases deliberately depend on the development trajectories. They are not
independent holdout data. Only a selected channel's source position and radius
change: place it 10 m from the origin along its original true bearing, at R=1000.
Preserve the original deterministic first reading and verify silence at the old
probe point. Full lean/probe runs must be performed separately because the new
source can change the route before that point is reached.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT / 'src' / 'q3_model_v2'
if str(Q3) not in sys.path:
    sys.path.append(str(Q3))
from environment import LocalArena


def construct(development: Path):
    cases, excluded, seen, inputs = [], [], set(), {}
    for path in sorted((development / 'cases').glob('*-probe.json')):
        detail = json.loads(path.read_text(encoding='utf-8'))
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        row = detail['result']
        if row['schedule'] != 'probe':
            raise ValueError(f'Unexpected variant in {path.name}')
        for action in detail['trace']:
            if action.get('phase') != 'local_action' or not action.get('uncertain_reception'):
                continue
            channel = action['channel']
            key = (row['name'], channel)
            if key in seen:
                continue
            seen.add(key)
            origin = np.zeros(2)
            source = next(s for s in detail['sources'] if s['channel'] == channel)
            first = next(e for e in detail['events']
                         if e['channel'] == channel and e['action'] == 'measure')
            distance = float(np.linalg.norm(source['position']))
            reason = None
            if first.get('measure_result') != 'direction' or np.linalg.norm(first['position']) > 1e-8:
                reason = 'The first channel measurement is not a directional reading at the origin.'
            elif distance <= 5:
                reason = 'The original source has no usable directional bearing from the origin.'
            if reason:
                excluded.append(dict(base_case=row['name'], channel=channel, reason=reason))
                continue

            expected = dict(measure_result='direction', svd_deg=first['svd_deg'])
            replay = LocalArena(detail['sources'], row['seed'], row['field'])
            original_reading = replay.measure(origin, channel)
            modified = copy.deepcopy(detail['sources'])
            target = next(s for s in modified if s['channel'] == channel)
            target['position'] = (np.asarray(source['position'], float) * (10. / distance)).tolist()
            target['radius'] = 1000.
            arena = LocalArena(modified, row['seed'], row['field'])
            modified_reading = arena.measure(origin, channel)
            q = np.asarray(action['position'], float)
            probe_distance = float(np.linalg.norm(q - np.asarray(target['position'])))
            probe_reading = arena.measure(q, channel)
            if original_reading != expected or modified_reading != expected:
                reason = 'Original replay or replacement does not reproduce the recorded first reading exactly.'
            elif probe_distance <= target['radius'] or probe_reading['measure_result'] != 'no_signal':
                reason = 'Replacement still receives at the originally selected probe point.'
            if reason:
                excluded.append(dict(base_case=row['name'], channel=channel, reason=reason,
                                     expected=expected, original_reading=original_reading,
                                     modified_reading=modified_reading, probe_reading=probe_reading))
                continue

            assert len(modified) == len(detail['sources'])
            assert all(a == b for a, b in zip(modified, detail['sources']) if a['channel'] != channel)
            assert abs(float(np.linalg.norm(target['position'])) - 10.) < 1e-10
            assert 10 <= len(modified) <= 16 and len({s['channel'] for s in modified}) == len(modified)
            assert all(np.linalg.norm(s['position']) <= 1800. + 1e-8 and 1000 <= s['radius'] <= 1500
                       for s in modified)
            cases.append(dict(
                name=f'{row["name"]}-near-origin-ch{channel}', seed=row['seed'], field=row['field'],
                sources=modified,
                construction=dict(base_case=row['name'], changed_channel=channel,
                    original_source=source, replacement_source=copy.deepcopy(target),
                    originally_selected_probe=q.tolist(),
                    original_design_signal_mass=action.get('design_signal_mass'),
                    first_measurement_position=origin.tolist(), expected_first_reading=expected,
                    original_replayed_first_reading=original_reading,
                    replacement_first_reading=modified_reading,
                    first_reading_preserved_exactly=True,
                    replacement_distance_to_original_probe_m=probe_distance,
                    replacement_feedback_at_original_probe=probe_reading,
                    source_count_unchanged=True, other_sources_unchanged=True)))
    if not inputs:
        raise ValueError('No development probe records found')
    return dict(
        split='constructed_stress_from_development', independent_holdout=False,
        local_only=True, official_simulator_used=False,
        source_data_sha256=inputs,
        construction='For each selected development probe, move only that channel source to 10 m '
                     'from the origin along its original true bearing, set its fixed radius to 1000 m, '
                     'and retain all other sources, the seed, and the fixed error field.',
        verification='Replay the original and replacement first reading with LocalArena, require '
                     'exact equality to the recorded origin reading, and require no signal at the '
                     'original probe point. Reject any case failing either check.',
        interpretation='These layouts target the small near-origin feasible region missed by finite '
                       'area samples. They do not estimate typical performance or prove that a full '
                       'rerun will select the same probe. Run both policies from the start with only '
                       'the public API; do not inject the old route or source truth into either solver.',
        unique_selected_case_channel_pairs=len(seen), kept_cases=len(cases),
        excluded_cases=excluded, cases=cases)


def main():
    folder = Path(__file__).resolve().parent
    output = folder / 'results' / 'probe-counterexamples-v1.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    result = construct(folder / 'results' / 'development-v1')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(),
                         selected_pairs=result['unique_selected_case_channel_pairs'],
                         kept_cases=result['kept_cases'], excluded_cases=len(result['excluded_cases']))))


if __name__ == '__main__':
    main()
