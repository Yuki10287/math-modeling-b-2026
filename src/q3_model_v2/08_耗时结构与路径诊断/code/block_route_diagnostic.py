"""Offline exact ordering of fixed logged blocks; never calls a simulator.

Every action position is preserved. channel_order also preserves each channel's
action order; removal_state only preserves its pre/post-removal status.
This hindsight optimum is not a deployable online strategy or a lower bound
for the unrestricted original problem.
"""
from functools import lru_cache
import math


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def action_cost(actions, position=(0., 0.), channel=1):
    parts = dict(movement=0., measure=0., switch=0., clear_fail=0., clear_success=0.)
    for event in actions:
        q = event['position']
        parts['movement'] += distance(position, q)/5.
        if event['action'] == 'measure':
            parts['measure'] += 5.
            parts['switch'] += int(channel != event['channel'])
            channel = event['channel']
        elif event['action'] == 'clear':
            result = event['clear_result']
            if result == 'success':
                parts['clear_success'] += 5.
            elif result == 'no_target_in_range':
                parts['clear_fail'] += 3.
            else:
                raise ValueError('Unsupported clear result')
        else:
            raise ValueError('Only accepted measure/clear actions are supported')
        position = q
    return dict(total_s=sum(parts.values()), costs_s=parts,
                end_position=list(position), end_channel=channel)


def block_data(blocks, precedence='channel_order'):
    if any(not block['actions'] for block in blocks):
        raise ValueError('Empty action block')
    predecessors = [0]*len(blocks)
    if precedence == 'channel_order':
        previous = {}
        for i, block in enumerate(blocks):
            for event in block['actions']:
                c = event['channel']
                if c in previous and previous[c] != i:
                    predecessors[i] |= 1 << previous[c]
                previous[c] = i
    elif precedence == 'removal_state':
        # Fixed-location Q3 readings commute until successful removal. Keep
        # every action on its original side of that channel's successful clear.
        success = {}
        for i, block in enumerate(blocks):
            for event in block['actions']:
                if event.get('clear_result') == 'success':
                    if event['channel'] in success:
                        raise ValueError('Repeated successful removal')
                    success[event['channel']] = i
        for i, block in enumerate(blocks):
            for event in block['actions']:
                j = success.get(event['channel'])
                if j is not None and i != j:
                    predecessors[max(i, j)] |= 1 << min(i, j)
    else:
        raise ValueError('Unknown precedence mode')
    summaries = []
    for block in blocks:
        events = block['actions']
        measurements = [e for e in events if e['action'] == 'measure']
        first = measurements[0]['channel'] if measurements else None
        last = measurements[-1]['channel'] if measurements else None
        cost = action_cost(events, events[0]['position'], first or 1)
        summaries.append(dict(entry=events[0]['position'], exit=events[-1]['position'],
                              internal_s=cost['total_s'], first=first, last=last))
    return predecessors, summaries


def solve_blocks(blocks, max_states=2_000_000, precedence='channel_order'):
    """Exact subset DP with precedence, movement and receiver retuning.

    All actions inside a block remain in original order. Clear does not change
    the measurement receiver channel. A state limit raises instead of silently
    returning a truncated/approximate result as optimal.
    """
    if not blocks:
        return dict(order=[], total_s=0., states=1, exact=True)
    predecessors, info = block_data(blocks, precedence)
    n = len(blocks)
    full = (1 << n)-1
    states = 0

    def choices(mask, last, tuned):
        position = (0., 0.) if last < 0 else info[last]['exit']
        for j, item in enumerate(info):
            if mask & (1 << j) or predecessors[j] & mask != predecessors[j]:
                continue
            switch = int(item['first'] is not None and tuned != item['first'])
            cost = distance(position, item['entry'])/5. + item['internal_s'] + switch
            yield j, cost, item['last'] if item['last'] is not None else tuned

    @lru_cache(maxsize=None)
    def future(mask, last, tuned):
        nonlocal states
        states += 1
        if states > max_states:
            raise RuntimeError(f'DP state budget exceeded: {max_states}')
        if mask == full:
            return 0.
        return min(cost+future(mask | (1 << j), j, after)
                   for j, cost, after in choices(mask, last, tuned))

    optimum = future(0, -1, 1)
    order = []
    mask, last, tuned = 0, -1, 1
    while mask != full:
        _, j, after = min((cost+future(mask | (1 << j), j, after), j, after)
                         for j, cost, after in choices(mask, last, tuned))
        order.append(j)
        mask, last, tuned = mask | (1 << j), j, after
    original = [event for block in blocks for event in block['actions']]
    reordered = [event for j in order for event in blocks[j]['actions']]
    def removal_states(events):
        removed, result = set(), []
        for event in events:
            result.append((event, event['channel'] in removed))
            if event.get('clear_result') == 'success':
                removed.add(event['channel'])
        return result
    before = {id(event): state for event, state in removal_states(original)}
    assert all(before[id(event)] == state for event, state in removal_states(reordered))
    same_channel_order = all([e for e in original if e['channel'] == channel] == [
        e for e in reordered if e['channel'] == channel]
        for channel in {e['channel'] for e in original})
    if precedence == 'channel_order':
        assert same_channel_order
    recomputed = action_cost(reordered)
    assert abs(recomputed['total_s']-optimum) < 1e-7
    old = action_cost(original)
    assert optimum <= old['total_s']+1e-7
    return dict(order=order, total_s=optimum, original_s=old['total_s'],
                saved_s=old['total_s']-optimum, states=states, exact=True,
                costs_s=recomputed['costs_s'], original_costs_s=old['costs_s'],
                predecessors=[[j for j in range(n) if mask & (1 << j)]
                              for mask in predecessors],
                precedence=precedence, per_channel_actions_identical=same_channel_order,
                all_actions_preserve_removal_state=True,
                scope='Fixed blocks and points, specified precedence; hindsight only.')
