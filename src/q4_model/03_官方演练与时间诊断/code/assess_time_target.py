"""Quantify an exploratory time target from saved records; no simulator access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from polar_cover import PolarCover
from route_planning import open_route, route_length


def main():
    here = Path(__file__).resolve().parent
    source = here/'results/official_q4_20260911_analysis.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    a = data['aggregate']
    target = 400.
    gap = a['total_s']-target*a['cleared']
    stations = PolarCover().stations
    distances = np.linalg.norm(stations[:, None, :]-stations[None, :, :], axis=2)
    mst = float(minimum_spanning_tree(distances).sum())
    route = open_route(stations[1:], stations[0])
    feasible = route_length(stations[1:], stations[0], route)
    assert abs(mst-feasible) < 1e-7
    fees = sum(v for k,v in a['costs_s'].items() if k != 'move')
    fixed_structure_floor = (a['runs']*mst/5+fees)/a['cleared']
    result = dict(target_s_per_source=target,
        comparison_status='User saw an approximately 400-second result; problem, completeness and averaging details are unverified.',
        input_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        current_pooled_s=a['pooled_average_s'], required_reduction_pct=100*gap/a['total_s'],
        required_saving_total_s=gap, required_mean_saving_s=gap/a['runs'],
        required_mean_route_saving_m_if_only_movement=5*gap/a['runs'],
        fixed_station_count=len(stations), scan_mst_lower_bound_m=mst,
        feasible_scan_path_m=feasible, feasible_scan_order=[0]+[int(i)+1 for i in route],
        fixed_observed_nonmovement_fees_s=fees,
        fixed_station_and_observed_action_count_floor_s_per_source=fixed_structure_floor,
        floor_scope='Numerical MST lower bound is attained by the displayed open path. Applies only if all 25 stations and observed action counts are retained; not a lower bound for every valid Q4 model.',
        hindsight_remove_entire_post_last_clear_tail_s_per_source=(a['total_s']-a['post_last_clear_s'])/a['cleared'],
        hindsight_tail_note='Diagnostic only: the solver did not know the source count and could not simply stop there.',
        remove_all_failed_clear_fees_s_per_source=(a['total_s']-a['costs_s']['clear_fail'])/a['cleared'],
        production_solver_changed=False, new_official_test_started=False,
        conclusion='The 400-second target calls for structural changes in discovery coverage or observation allocation; no achieved 400-second result is claimed.')
    output = here/'results/target400_assessment.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
