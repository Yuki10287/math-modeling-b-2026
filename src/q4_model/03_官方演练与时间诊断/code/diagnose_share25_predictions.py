"""Describe selected measurements and optical tails in saved share25 logs.

Reads supplied ZIP members in memory and a prior audited, deidentified summary.
No network, hidden truth reconstruction, policy modification, or counterfactual
time claims. Predicted signal mass is a finite-scenario score, not a calibrated
probability. Its descriptive discrepancy is reported without significance tests.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    cases = {c['case']: c for c in audit['cases']}
    records, optical = [], []
    with zipfile.ZipFile(args.zip) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith('.jsonl.beliefs.json')):
            case = Path(name.removesuffix('.jsonl.beliefs.json')).name
            actions = cases[case]['actions']
            pending = {}
            for row in json.loads(archive.read(name)):
                phase, channel = row['phase'], row.get('channel')
                if phase == 'shared_prediction':
                    pending[('shared_measure', channel)] = row
                elif phase == 'decision' and row['action'] == 'measure':
                    pending[('source_measure', channel)] = row
                elif phase == 'actual_action' and row['reason'] in ('shared_measure', 'source_measure'):
                    prediction = pending.pop((row['reason'], channel))
                    event = actions[row['event']-1]
                    assert event['channel'] == channel and event['reason'] == row['reason']
                    records.append(dict(case=case, channel=channel, event=row['event'],
                        reason=row['reason'], signal_mass=prediction['signal_mass'],
                        result=row['result'], move_s=event['move_s'], fee_s=event['fee_s'],
                        finite_scenarios=prediction.get('scenarios'),
                        predicted_remaining_s=prediction.get('predicted_s'),
                        predicted_optical_s=prediction.get('optical_s'),
                        predicted_gain_s=prediction.get('gain_s')))
                elif phase == 'optical_plan':
                    selected = [a for a in actions if a['channel'] == channel and a['reason'] == 'optical_cover']
                    assert selected and selected[-1]['clear_result'] == 'success'
                    sides = [h-l for l, h in zip(row['lower'], row['upper'])]
                    optical.append(dict(case=case, channel=channel,
                        rectangle_sides_m=sides, cells=row['cells'], full_path_points=len(row['path']),
                        actual_clear_attempts=len(selected), failed_clears=len(selected)-1,
                        actual_optical_s=sum(a['duration_s'] for a in selected),
                        actual_optical_move_s=sum(a['move_s'] for a in selected),
                        actual_local_total_s=cases[case]['per_channel'][str(channel)]['local_time_s'],
                        predicted_score_s=row['score'], predicted_mean_s=row['mean_s'],
                        predicted_worst_s=row['worst_s']))
            assert not pending
    by_reason = {}
    for reason in ('shared_measure', 'source_measure'):
        group = [r for r in records if r['reason'] == reason]
        near_one = [r for r in group if r['signal_mass'] >= .95]
        all_one = [r for r in group if abs(r['signal_mass']-1) < 1e-12]
        by_reason[reason] = dict(count=len(group),
            results=dict(Counter(r['result'] for r in group)),
            summed_predicted_signal_mass=sum(r['signal_mass'] for r in group),
            observed_positive=sum(r['result'] != 'no_signal' for r in group),
            predictions_at_least_095=len(near_one),
            negatives_among_at_least_095=sum(r['result'] == 'no_signal' for r in near_one),
            predictions_equal_1=len(all_one),
            negatives_among_equal_1=sum(r['result'] == 'no_signal' for r in all_one))
    result = dict(archive_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_simulator_contacted=False, hidden_truth_used=False,
        by_reason=by_reason, measurements=records, optical_plans=optical,
        limitations=['Signal mass is not asserted to be a calibrated probability.',
            'Selected observations are correlated and policy-dependent.',
            'A negative measurement may still inform the scenario model; zero polygon shrink is not zero value.',
            'No responses have been reused for an alternative action or changed policy.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(by_reason, ensure_ascii=False, indent=2))
    print(json.dumps(sorted(optical, key=lambda x: x['actual_local_total_s'], reverse=True)[:2], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
