"""Apply the written decision rule to the frozen candidate's completed local runs."""
import hashlib
import json
from pathlib import Path
from benchmark_task_sharing import hashes


def main():
    root = Path(__file__).parent/'results'
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    plan = read('cell_evaluation_plan.json')
    selection = read('cell_selection.json')
    variant = selection['selected_variant']
    assert variant == 'share25'
    holdout = read('task_holdout_23000/analysis.json')
    holdout_manifest = read('task_holdout_23000/manifest.json')
    stress = read('task_stress_share25/summary.json')
    stress_manifest = read('task_stress_share25/manifest.json')
    assert hashes() == selection['code_hashes'] == holdout_manifest['code_hashes'] == stress_manifest['source_sha256']
    assert selection['evaluation_plan_sha256'] == hashlib.sha256((root/'cell_evaluation_plan.json').read_bytes()).hexdigest()
    cmp = holdout['comparisons_to_baseline25'][variant]
    rules = plan['minimum_evidence_for_recommending_replacement']
    worst_percent = max(0., max(-r['relative_reduction_percent'] for r in cmp['condition_records']))
    gates = dict(
        all_holdout_audits_and_clears=all(holdout['safety_checks'].values()),
        all_stress_audits_and_clears=(stress['runs']==30 and stress['passed']==30 and stress['all_cleared']==30
            and stress['source_snapshot_stable'] and stress['stress_snapshot_stable']),
        mean_gain_at_least_five_percent=cmp['relative_reduction_percent'] >= rules['holdout_mean_total_reduction_at_least_percent'],
        faster_layouts_at_least_six=cmp['layout_aggregated']['outcomes']['win'] >= rules['faster_independent_holdout_layouts_at_least'],
        worst_condition_slowdown_at_most_fifteen_percent=worst_percent <= rules['maximum_single_condition_slowdown_at_most_percent'])
    rr = cmp['layout_aggregated']['records']
    leave_one_out = []
    for omit in range(len(rr)):
        old = sum(r['reference_mean_s'] for k,r in enumerate(rr) if k != omit)
        new = sum(r['candidate_mean_s'] for k,r in enumerate(rr) if k != omit)
        leave_one_out.append(100*(old-new)/old)
    report = dict(selected_variant=variant, candidate_source_snapshot_stable=True,
        local_only=True, official_simulator_used=False,
        holdout_groups=holdout['groups'], paired_comparison=cmp,
        worst_condition_slowdown_percent=worst_percent,
        leave_one_layout_out_reduction_range_percent=[min(leave_one_out),max(leave_one_out)],
        leave_one_out_interpretation='Descriptive sensitivity to deleting one layout, not a confidence interval.',
        stress=stress, decision_gates=gates, meets_predeclared_replacement_criteria=all(gates.values()),
        official_entry_changed=False,
        official_evidence='Only the existing 25-station shared release has the five user-provided official runs. No new candidate has been run officially in this work.',
        runtime_context='Holdout and stress ran concurrently on this computer. Solver wall times exclude auditing and are observations under shared load, not controlled speed comparisons or official program runtime.',
        development_data='Two stages totaling 45 formal runs on ten development layouts, plus repeated smoke checks. Development is excluded from holdout and stress summaries.',
        evidence_files=['cell_evaluation_plan.json','cell_selection.json','cell_development_21000/analysis.json',
            'task_development_22000/analysis.json','task_holdout_23000/analysis.json','task_stress_share25/summary.json'])
    (root/'task_candidate_assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(gain_percent=cmp['relative_reduction_percent'],
        mean_saving_s=cmp['mean_saving_s'], condition_outcomes=cmp['condition_outcomes'],
        layout_outcomes=cmp['layout_aggregated']['outcomes'], worst_slowdown_percent=worst_percent,
        leave_one_out_range=report['leave_one_layout_out_reduction_range_percent'],
        gates=gates, meets_predeclared_replacement_criteria=all(gates.values())), indent=2))


if __name__ == '__main__': main()
