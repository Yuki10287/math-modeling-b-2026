"""Independent local research variants using the certified 22-station cell cover."""
import math
import numpy as np
from cell_cover import CellCover
from localization import Belief, optical_plan, choose_measure
from guarded_policy import score_at
from route_planning import open_route
from solver import accepted
from shared import core


VARIANTS = ('cells', 'context', 'interleave', 'opscan')


def solve_multi(api, variant='cells', trace=None, max_active_measures=6):
    if variant not in VARIANTS:
        raise ValueError('unknown cell-cover variant')
    if variant != 'cells':
        from context_policy import choose_context_measure, choose_context_optical
    trace = [] if trace is None else trace
    cover = CellCover()
    beliefs, local_steps = {}, {}
    cleared, discovered = set(), set()
    calls = 0
    context = variant != 'cells'
    interleave = variant in ('interleave', 'opscan')

    def unknown():
        if len(discovered) >= 16:
            return []
        return [c for c in range(1, 21) if c not in discovered and not cover.complete(c)]

    def measure(q, c, reason):
        nonlocal calls
        before = np.asarray(api.position).copy()
        calls += 1
        reply = api.measure(q, c)
        kind = accepted(reply, 'measure_result', ('no_signal', 'near', 'direction'))
        trace.append(dict(phase='actual_action', action='measure', reason=reason, channel=c,
            event=calls, position=np.asarray(q).tolist(), move_s=float(np.linalg.norm(q-before))/5, result=kind))
        if kind == 'no_signal':
            cover.observe_negative(c, q)
            if c in beliefs:
                beliefs[c].update(q, reply)
        elif c not in beliefs:
            beliefs[c] = Belief(q, reply)
            beliefs[c].negatives = [p.copy() for p in cover.negative[c]]
            local_steps[c] = 0
            discovered.add(c)
        else:
            beliefs[c].update(q, reply)
        if c in beliefs:
            b = beliefs[c]
            trace.append(dict(phase='belief', channel=c, polygon=b.P.tolist(),
                positives=[p.tolist() for p in b.positives], negatives=[p.tolist() for p in b.negatives]))

    def clear(q, c, reason):
        nonlocal calls
        before = np.asarray(api.position).copy()
        calls += 1
        kind = accepted(api.clear(q, c), 'clear_result', ('success', 'no_target_in_range'))
        trace.append(dict(phase='actual_action', action='clear', reason=reason, channel=c,
            event=calls, position=np.asarray(q).tolist(), move_s=float(np.linalg.norm(q-before))/5, result=kind))
        if kind == 'success':
            cleared.add(c)
            beliefs.pop(c, None)
            return True
        if c in beliefs:
            beliefs[c].failed_clears.append(np.asarray(q).copy())
        return False

    def continuations(exclude):
        points = [cover.stations[k] for k in cover.needed_stations(unknown())]
        points.extend(core.mec(b.P)[0] for c, b in beliefs.items() if c != exclude)
        return np.array(points, float).reshape(-1, 2)

    def service(c):
        belief = beliefs[c]
        for _ in range(max_active_measures+1):
            if belief.near is not None:
                return 'cleared' if clear(belief.near, c, 'near_clear') else 'failed'
            safe = core.nearest_certified_clear(belief.P, np.asarray(api.position))
            if safe is not None:
                return 'cleared' if clear(safe, c, 'certified_clear') else 'failed'
            remaining = continuations(c) if context else np.empty((0, 2))
            if context:
                plan = choose_context_optical(belief, np.asarray(api.position), remaining)
                action = (choose_context_measure(belief, np.asarray(api.position), c, api.channel, remaining)
                          if local_steps[c] < max_active_measures else None)
            else:
                plan = optical_plan(belief.P, np.asarray(api.position))
                action = (choose_measure(belief, np.asarray(api.position), c, api.channel)
                          if local_steps[c] < max_active_measures else None)
            if action is not None and action['score'] < plan['score']:
                trace.append(dict(phase='decision', channel=c, action='measure', predicted_s=action['score'],
                    optical_s=plan['score'], signal_mass=action['signal_mass'], scenarios=action['scenarios'],
                    continuation_points=len(remaining)))
                measure(action['q'], c, 'source_measure')
                local_steps[c] += 1
                if interleave:
                    return 'measured'
                continue
            trace.append(dict(phase='optical_plan', channel=c, polygon=belief.P.tolist(),
                **{k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in plan.items()}))
            # Finish the complete cover unless an actual success is returned.
            for q in plan['path']:
                if clear(q, c, 'optical_cover'):
                    return 'cleared'
            return 'failed'
        return 'failed'

    def share_at_stop(newly_found):
        q = np.asarray(api.position).copy()
        choices = []
        for c, b in list(beliefs.items()):
            if np.max(np.linalg.norm(b.P-q, axis=1)) <= 20-1e-6:
                if not clear(q, c, 'shared_clear'):
                    return False
                continue
            if c in newly_found or min(np.linalg.norm(q-p) for p in b.measured) < 80:
                continue
            value = score_at(b, q, q, c, api.channel, robust=False)
            if value is not None:
                gain = optical_plan(b.P, q)['score']-value['score']
                if gain > 0:
                    choices.append((gain, c))
        for gain, c in sorted(choices, reverse=True)[:2]:
            value = score_at(beliefs[c], q, q, c, api.channel, robust=False)
            if value is not None and value['score'] < optical_plan(beliefs[c].P, q)['score']:
                trace.append(dict(phase='shared_prediction', channel=c, gain_s=gain,
                                  signal_mass=value['signal_mass']))
                measure(q, c, 'shared_measure')
        return True

    def scan(q, reason='scan'):
        newly_found = set()
        for c in sorted(unknown(), key=lambda c:(c != api.channel, c)):
            if len(discovered) == 16:
                break
            if any(np.array_equal(q, p) for p in cover.negative[c]):
                continue
            measure(q, c, reason)
            if c in discovered:
                newly_found.add(c)
        return share_at_stop(newly_found)

    def scan_if_certificate_task_removed():
        channels = unknown()
        if not channels:
            return True
        q = np.asarray(api.position).copy()
        if any(np.array_equal(q, p) for p in cover.stations):
            return True
        before = set(cover.needed_stations(channels))
        after = set()
        for c in channels:
            gain = cover.preview_negative(c, q)
            # The cover exposes exactly the static witness sets used to guarantee completion.
            remaining_cells = np.flatnonzero(~(cover.covered[c] | gain))
            for t in remaining_cells:
                for k in cover.static_witnesses[t]:
                    if not any(np.array_equal(cover.stations[k], p) for p in cover.negative[c]):
                        after.add(int(k))
        removed = before-after
        if not removed:
            return True
        # Necessary task reduction is only a design scenario: all actual observations still cost time.
        trace.append(dict(phase='opportunity_prediction', removed_scan_stations=sorted(removed),
                          measured_channels=channels, negative_scenario_only=True))
        return scan(q, 'opportunity_scan')

    if not scan(np.asarray(api.position).copy()):
        return dict(complete=False, reason='shared_clear_conflict')
    # At most 22 fixed scans, 16*(6 measurements+one complete optical service), plus slack.
    for _ in range(len(cover.stations)+16*(max_active_measures+2)+2):
        absent = [c for c in range(1,21) if c not in discovered and cover.complete(c)]
        if len(cleared) == 16 or len(cleared)+len(absent) == 20:
            certificate = cover.geometry_certificate()
            certificate.update(
                basis='count_upper_bound' if len(cleared) == 16 else 'directional_cell_cover',
                cleared=sorted(cleared), absent=absent,
                channels={str(c):cover.certificate(c) for c in absent})
            return dict(complete=True, cleared=sorted(cleared), certificate=certificate)
        tasks, points = [], []
        for k in cover.needed_stations(unknown()):
            tasks.append(('scan', k))
            points.append(cover.stations[k])
        for c, b in beliefs.items():
            tasks.append(('source', c))
            points.append(core.mec(b.P)[0])
        if not tasks:
            break
        order = open_route(np.asarray(points), api.position)
        task, index = tasks[order[0]]
        trace.append(dict(phase='task_choice', task=task, index=index, known=len(beliefs),
                          unknown=len(unknown()), pending_stations=sum(t[0]=='scan' for t in tasks)))
        if task == 'scan':
            if not scan(cover.stations[index].copy()):
                return dict(complete=False, reason='shared_clear_conflict')
        else:
            status = service(index)
            if status == 'failed':
                return dict(complete=False, reason='local_cover_or_feedback_conflict', cleared=sorted(cleared))
            if variant == 'opscan' and not scan_if_certificate_task_removed():
                return dict(complete=False, reason='opportunity_clear_conflict')
    return dict(complete=False, reason='discovery_certificate_incomplete', cleared=sorted(cleared))
