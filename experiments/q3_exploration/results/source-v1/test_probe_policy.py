import copy
import unittest
from unittest.mock import patch
import numpy as np
from bootstrap import FeedbackModel,core
from environment import LocalArena
from local_policy import choose_action
from validate_model import PublicAPI,independent_time_audit,observation_audit,trace_truth_audit
from probe_policy import radius_scenarios,negative_region,choose_probe
from exploration_solver import solve_multi


class ProbeTests(unittest.TestCase):
    def model(self):
        arena=LocalArena([dict(channel=1,position=[1200.,0.],radius=1100.)],field='constant')
        model=FeedbackModel(PublicAPI(arena),[],range_cuts=True,search_fraction=.3)
        model.measure([500.,0.],1)
        model.measure([0.,0.],20)
        return arena,model

    def test_joint_radius_bounds_and_negative_truth_containment(self):
        arena,model=self.model()
        self.assertFalse(core.reception_certified(model.beliefs[1]['P'],np.zeros(2),model.beliefs[1]['witness']))
        self.assertEqual(model.measure([0.,0.],1)['measure_result'],'no_signal')
        self.assertIsNone(model.beliefs[1]['conflict'])
        self.assertTrue(radius_scenarios(model,1))
        for g,lo,hi in radius_scenarios(model,1):
            self.assertLessEqual(np.linalg.norm(g),1800)
            self.assertGreaterEqual(lo,1000)
            self.assertLessEqual(hi,1500)
            self.assertLessEqual(np.linalg.norm(g-[500,0]),lo+1e-7)
            self.assertLess(hi,np.linalg.norm(g))
        trace_truth_audit(arena,model.trace)

    def test_scoring_does_not_change_real_evidence(self):
        arena,model=self.model();item=model.beliefs[1]
        action=choose_action(item['P'],model.position,item['witness'],1,
            current_channel=model.channel,observed_positions=model.positions[1],candidate_mode='short')
        P=item['P'].copy();excluded=model.coverage._excluded.copy();events=copy.deepcopy(arena.events)
        before=[x.copy() for x in model.positions[1]]
        choose_probe(model,1,action,[np.array([-500.,300.])])
        np.testing.assert_array_equal(P,item['P'])
        np.testing.assert_array_equal(excluded,model.coverage._excluded)
        np.testing.assert_array_equal(before,model.positions[1])
        self.assertEqual(events,arena.events)

    def test_committed_optical_cover_is_not_interrupted(self):
        _,model=self.model()
        model.beliefs[1]['state']={'remaining':[[1100,0]],'cover_size':2,'exhausted':False}
        action=dict(kind='clear',q=np.array([1090.,0.]),state={})
        self.assertIs(choose_probe(model,1,action),action)

    def test_actual_silent_probe_returns_to_baseline_and_completes(self):
        sources=[dict(channel=c,position=[1600.,0.],radius=1000.) for c in range(1,11)]
        arena=LocalArena(sources,field='extreme');trace=[];forced=False
        def select(model,c,fallback,hints):
            nonlocal forced
            if not forced and fallback['kind']=='measure':
                forced=True
                return dict(kind='measure',q=np.array([0.,50.]),state=None,certified=False,
                    uncertain_reception=True,rationale='test_forced_legal_silent_probe')
            return fallback
        with patch('exploration_solver.choose_probe',select):
            result=solve_multi(PublicAPI(arena),trace=trace,variant='probe')
        self.assertTrue(forced)
        self.assertTrue(result['complete'])
        self.assertEqual(result['probe_outcomes']['no_signal'],1)
        self.assertEqual(result['fallback_count'],0)
        self.assertTrue(arena.evaluation()['all_cleared'])
        independent_time_audit(arena);observation_audit(arena);trace_truth_audit(arena,trace)


if __name__=='__main__':unittest.main()
