"""Global travel/coverage opportunity correction to the frozen local cost.

No hidden simulator state is read. Prospective no-signal masks are copied;
they never become real evidence. The correction is a finite planning surrogate,
not an expected optimal value or a probabilistic absence claim.
"""
from __future__ import annotations
import copy
import math
import numpy as np

import geometry as core
from joint_planning import route_cost, shortest_visit
from local_policy import (choose_action as base_action, short_candidates,
    area_scenarios, _measure_score, best_optical_plan)
from v1_solver import rolling_route


def ordered(start, tasks):
    return rolling_route(np.asarray(start), tasks)[0] if tasks else []


class GlobalContext:
    def __init__(self, model, active):
        self.model, self.active = model, active
        self.center = core.mec(model.beliefs[active]['P'])[0]
        self.tasks = copy.deepcopy(next(r['route'] for r in reversed(model.trace) if r.get('phase') == 'plan'))
        self.others = [t for t in self.tasks if not (t['kind'] == 'source' and t['key'] == active)]
        self.reference = route_cost(self.center, ordered(self.center, self.others)) / 5
        self.channels = model.unknown()
        self.residual = (~model.coverage._excluded[[model.coverage._index(c) for c in self.channels]]).copy()
        self.masks = {t['key']: model.coverage._mask(t['position']).copy()
                      for t in self.tasks if t['kind'] == 'scan'}
        self.shared_prior = {}

    def waypoint_candidates(self, base):
        """Consider the next other tasks as stops while measuring the active source."""
        P, witness = self.model.beliefs[self.active]['P'], self.model.beliefs[self.active]['witness']
        tasks = sorted(self.others, key=lambda t: np.linalg.norm(np.asarray(t['position'])-base))[:3]
        qs = []
        for task in tasks:
            for fraction in (1., .5):
                q = fraction*np.asarray(task['position'])+(1-fraction)*base
                if (core.reception_certified(P, q, witness) and not self.model.recorded(self.active, q)
                        and np.linalg.norm(q-witness) > .1
                        and all(np.linalg.norm(q-old) > .1 for old in qs)):
                    qs.append(q)
        return qs

    def shared_credit(self, q):
        """Price existing opportunistic bearings, using the old tail approximation.

        The eligibility test mirrors the actual rule for currently known sources.
        It cannot predict sources newly discovered at this stop. Area samples and
        three error values are ranking scenarios, not an official posterior.
        Centre-start tail costs approximate intrinsic service separately from
        the travel tour, avoiding another charge for approaching the source.
        """
        possible = []
        for c, item in self.model.beliefs.items():
            P, witness = item['P'], item['witness']
            if (c == self.active or item['state'] or self.model.recorded(c, q) or item['near'] is not None
                    or np.linalg.norm(q-witness) < 80 or not core.reception_certified(P, q, witness)):
                continue
            center, radius = core.mec(P)
            if radius <= 20:
                continue
            samples = area_scenarios(P, 4)
            radii = []
            for g in samples:
                if np.linalg.norm(g-q) <= 5:
                    radii.append(5.)
                else:
                    angle = math.degrees(math.atan2(*(g-q)[::-1]))
                    radii.append(core.mec(core.wedge(P, q, angle))[1])
            gain = radius-float(np.mean(radii))
            if gain > 30 and np.mean(radii) < .7*radius:
                possible.append((gain, c, samples, center))
        credit, channels = 0., []
        for _, c, samples, center in sorted(possible, key=lambda x: (x[0], x[1]), reverse=True)[:2]:
            P = self.model.beliefs[c]['P']
            if c not in self.shared_prior:
                values = [core.continuation_cost(P, center, g) for g in samples]
                self.shared_prior[c] = .8*float(np.mean(values))+.2*float(np.max(values))
            after = []
            for g in samples:
                if np.linalg.norm(g-q) <= 5:
                    after.append(5.)
                    continue
                bearing = math.degrees(math.atan2(*(g-q)[::-1]))
                values = []
                for error in (-1., 0., 1.):
                    region = core.disk_clip(core.wedge(P, q, bearing+error), q)
                    if not len(region):
                        raise ValueError('empty shared-bearings prediction')
                    values.append(core.continuation_cost(region, core.mec(region)[0], g))
                after.append(float(np.mean(values)))
            remaining = .8*float(np.mean(after))+.2*float(np.max(after))
            credit += self.shared_prior[c]-remaining-6  # Pay for the real extra bearing.
            channels.append(c)
        return float(credit), channels

    def eligible(self, q):
        mask = self.model.coverage._mask(q)
        gains = np.count_nonzero(self.residual & mask, axis=1)
        counts = np.count_nonzero(self.residual, axis=1)
        return [i for i, c in enumerate(self.channels) if gains[i] and
                (gains[i] == counts[i] or gains[i] >= max(300, self.model.search_fraction * counts[i]))
                and not self.model.recorded(c, q)]

    def negative_route(self, q, tasks, residual, current_channel, *, prune):
        """Price a feasible all-negative continuation, per channel and cell."""
        route = ordered(q, tasks)
        required = np.any(residual, axis=0) if len(residual) else np.zeros(self.model.coverage.total_cells, bool)
        if prune:
            for task in list(reversed(route)):
                if task['kind'] != 'scan':
                    continue
                masks = [self.masks[t['key']] for t in route if t['kind'] == 'scan' and t is not task]
                covered = np.any(masks, axis=0) if masks else np.zeros(len(required), bool)
                if np.all(covered[required]):
                    route.remove(task)
            route = ordered(q, route)
        masks = [self.masks[t['key']] for t in route if t['kind'] == 'scan']
        covered = np.any(masks, axis=0) if masks else np.zeros(len(required), bool)
        assert np.all(covered[required]), 'hypothetical route lost residual coverage'
        remaining = residual.copy()
        measurement_s = 0.
        tuned = current_channel
        for task in route:
            if task['kind'] == 'source':
                # The source's local cost is counted by the local score.
                # Future channel selection remains an approximation.
                continue
            mask = self.masks[task['key']]
            needed = [i for i in range(len(self.channels)) if np.any(remaining[i] & mask)]
            for i in sorted(needed, key=lambda i: (self.channels[i] != tuned, self.channels[i])):
                c = self.channels[i]
                measurement_s += 5 + int(c != tuned)
                tuned = c
                remaining[i] &= ~mask
        assert not remaining.any(), 'hypothetical scans incomplete'
        return route_cost(q, route) / 5 + measurement_s, route

    def correction(self, action, variant):
        q = np.asarray(action['q'])
        final = action['kind'] == 'clear' and action.get('certified')
        tasks = self.others if final else self.tasks
        route = ordered(q, tasks)
        approach = 0. if final else float(np.linalg.norm(q - self.center)) / 5
        # Replace the isolated q-to-active-centre leg with a joint residual tour;
        # subtract the same centre-to-other-tasks reference for all alternatives.
        travel = route_cost(q, route) / 5 - approach - self.reference
        coverage, eligible = 0., []
        opportunity_runs = action['kind'] == 'measure' or final
        if variant in ('coverage', 'waypoints', 'shared') and self.channels and opportunity_runs:
            indices = self.eligible(q)
            eligible = [self.channels[i] for i in indices]
            if indices:
                tuned = self.active if action['kind'] == 'measure' else self.model.channel
                before, _ = self.negative_route(q, tasks, self.residual, tuned, prune=True)
                after = self.residual.copy()
                mask = self.model.coverage._mask(q)
                purchase = 0.
                for i in sorted(indices, key=lambda i: (self.channels[i] != tuned, self.channels[i])):
                    c = self.channels[i]
                    purchase += 5 + int(c != tuned)
                    tuned = c
                    after[i] &= ~mask
                future, _ = self.negative_route(q, tasks, after, tuned, prune=True)
                coverage = purchase + future - before
        shared, known_channels = self.shared_credit(q) if variant == 'shared' and opportunity_runs else (0., [])
        return dict(route_correction_s=float(travel), coverage_correction_s=float(coverage),
                    shared_credit_s=shared, hypothetical_shared_channels=known_channels,
                    hypothetical_scan_channels=eligible, correction_s=float(travel + coverage - shared))


def candidates(model, channel):
    """Retain the frozen local optimum and a small shortlist of its alternatives."""
    item, s = model.beliefs[channel], np.asarray(model.position)
    P, witness = item['P'], item['witness']
    if item['state'] is not None:
        return [base_action(P, s, witness, channel, current_channel=model.channel,
            state=item['state'], mode='hybrid', observed_positions=model.positions[channel], candidate_mode='short')]
    q = core.nearest_certified_clear(P, s)
    if q is not None:
        return [dict(kind='clear', q=q, state=None, certified=True,
            rationale='whole_belief_in_one_disk', estimated_s=float(np.linalg.norm(q-s))/5+5)]
    used = lambda q: model.recorded(channel, q) or np.linalg.norm(q - witness) <= .1
    qs = [q for q in short_candidates(P, s, witness) if not used(q)]
    if not used(s) and core.reception_certified(P, s, witness):
        qs.insert(0, s.copy())
    if not qs:
        raise ValueError('no unused reception-certified candidate')
    samples = area_scenarios(P)
    ranked = sorted([(_measure_score(P, s, q, samples) + int(model.channel != channel), i, q)
                     for i, q in enumerate(qs)], key=lambda x: (x[0], x[1]))
    choices = [dict(kind='measure', q=q, state=None, certified=False, rationale='area_risk_rollout',
        estimated_s=float(score)) for score, _, q in ranked[:4]]
    plan = best_optical_plan(P, s) if core.mec(P)[1] <= 180 else None
    if plan is not None:
        path = plan['path']
        choices.append(dict(kind='clear', q=path[0],
            state=dict(remaining=path[1:].tolist(), cover_size=len(path), exhausted=len(path)==1),
            certified=len(path)==1, optical_path=path.tolist(), estimated_s=plan['score'],
            worst_cover_s=plan['worst_s'], rationale='certified_optical_partition_cheaper'))
    # Match the frozen local selector: optical wins a tie.
    return sorted(choices, key=lambda a: (a['estimated_s'], a['kind'] != 'clear'))


def choose_global_action(model, channel, *, variant):
    if variant not in ('route', 'coverage', 'waypoints', 'shared'):
        raise ValueError('unknown global value variant')
    choices = candidates(model, channel)
    if model.beliefs[channel]['state'] is not None:
        return choices[0]  # Keep the already certified optical sequence.
    context = GlobalContext(model, channel)
    if variant in ('waypoints', 'shared') and not (choices[0]['kind'] == 'clear' and choices[0].get('certified')):
        P, s = model.beliefs[channel]['P'], np.asarray(model.position)
        samples = area_scenarios(P)
        for q in context.waypoint_candidates(np.asarray(choices[0]['q'])):
            if all(np.linalg.norm(q-a['q']) > .1 or a['kind'] != 'measure' for a in choices):
                choices.append(dict(kind='measure', q=q, state=None, certified=False,
                    rationale='shared_route_waypoint',
                    estimated_s=float(_measure_score(P, s, q, samples)+int(model.channel != channel))))
    if choices[0]['kind'] == 'clear' and choices[0].get('certified'):
        after = context.others[0]['position'] if context.others else None
        old = choices[0]
        q = shortest_visit(model.position, after, model.beliefs[channel]['P'], 20-1e-6, old['q'])
        if np.linalg.norm(q - old['q']) > .1:
            choices.append(dict(old, q=q, estimated_s=float(np.linalg.norm(q-model.position))/5+5))
    evaluated = []
    for index, action in enumerate(choices):
        terms = context.correction(action, variant)
        score = action['estimated_s'] + terms['correction_s']
        evaluated.append((score, index, dict(action, **terms, global_score_s=score)))
    best = min(evaluated, key=lambda x: (x[0], x[1]))
    action = best[2]
    action['rationale'] = 'global_value_' + variant
    action['changed_from_local'] = best[1] != 0
    model.trace.append(dict(phase='global_value_decision', channel=channel,
        chosen_index=best[1], variant=variant, alternatives=[dict(
            kind=a['kind'], position=np.asarray(a['q']).tolist(), local_score_s=a['estimated_s'],
            global_score_s=score, route_correction_s=a['route_correction_s'],
            coverage_correction_s=a['coverage_correction_s'],
            shared_credit_s=a['shared_credit_s'],
            hypothetical_shared_channels=a['hypothetical_shared_channels'],
            hypothetical_scan_channels=a['hypothetical_scan_channels']) for score, _, a in evaluated]))
    return action
