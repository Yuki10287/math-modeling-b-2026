import itertools
import unittest
from block_route_diagnostic import action_cost, block_data, solve_blocks


def measure(x, y, c):
    return dict(action='measure', position=[x,y], channel=c, measure_result='no_signal')


def clear(x, y, c, success=True):
    return dict(action='clear', position=[x,y], channel=c,
                clear_result='success' if success else 'no_target_in_range')


class RouteTest(unittest.TestCase):
    def test_cost_clear_does_not_retune(self):
        result = action_cost([measure(0,0,2), clear(30,40,3), measure(30,40,2)])
        self.assertEqual(result['costs_s']['switch'], 1)
        self.assertEqual(result['total_s'], 26)

    def test_exact_against_brute_force_with_constraints(self):
        blocks = [dict(actions=a) for a in [
            [measure(0,0,2), measure(0,0,3)],
            [measure(400,0,2), clear(401,0,2)],
            [measure(0,500,4)], [clear(1,500,4)],
            [measure(50,0,3), clear(52,0,3,False), clear(55,0,3)],
        ]]
        pred, _ = block_data(blocks)
        feasible = []
        for order in itertools.permutations(range(len(blocks))):
            mask = 0
            for j in order:
                if pred[j] & mask != pred[j]:
                    break
                mask |= 1 << j
            else:
                feasible.append(action_cost([e for j in order for e in blocks[j]['actions']])['total_s'])
        result = solve_blocks(blocks)
        self.assertAlmostEqual(result['total_s'], min(feasible), places=9)
        self.assertGreater(result['saved_s'], 0.)

    def test_same_channel_cannot_reverse(self):
        blocks = [dict(actions=[measure(1000,0,2)]), dict(actions=[clear(1,0,2)])]
        self.assertEqual(solve_blocks(blocks)['order'], [0,1])

    def test_budget_rejects_partial_optimum(self):
        with self.assertRaisesRegex(RuntimeError, 'budget exceeded'):
            solve_blocks([dict(actions=[measure(0,0,2)])], max_states=1)

    def test_removal_relaxation_allows_commuting_reads_but_not_removal(self):
        blocks = [dict(actions=a) for a in [
            [measure(1000,0,2)], [measure(1,0,2)], [clear(1001,0,2)],
            [measure(2,0,2)],
        ]]
        pred, _ = block_data(blocks, 'removal_state')
        self.assertEqual(pred, [0,0,3,4])
        result = solve_blocks(blocks, precedence='removal_state')
        self.assertEqual(result['order'], [1,0,2,3])
        self.assertTrue(result['all_actions_preserve_removal_state'])
        self.assertFalse(result['per_channel_actions_identical'])

    def test_removal_mode_matches_full_permutation_search(self):
        blocks = [dict(actions=a) for a in [
            [measure(300,0,2), measure(300,0,3)],
            [measure(0,50,2)], [clear(305,0,2)],
            [clear(310,0,3)], [measure(0,10,4)]
        ]]
        pred, _ = block_data(blocks, 'removal_state')
        costs = []
        for order in itertools.permutations(range(5)):
            mask = 0
            for j in order:
                if pred[j] & mask != pred[j]:
                    break
                mask |= 1 << j
            else:
                costs.append(action_cost([e for j in order for e in blocks[j]['actions']])['total_s'])
        self.assertAlmostEqual(solve_blocks(blocks, precedence='removal_state')['total_s'], min(costs), places=9)


if __name__ == '__main__':
    unittest.main()
