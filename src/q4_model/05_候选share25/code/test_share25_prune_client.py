"""Small production prune-client protocol and failure tests on owned loopback.

The separate HTTP validation driver covers full source layouts and real pruning.
These tests isolate dispatch, accepted-feedback provenance, error cleanup,
idempotent retries, source fingerprints, and exclusive log creation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from validate_share25_prune_http import Q4,PROJECT,fixture,LocalArena,write
sys.path.insert(0,str(Q4))
import q4_share25_prune_client as client
import q4_official_client as original
import ordered_optical_prune

OUT=None


class PruneClientTests(unittest.TestCase):
    def setUp(self):
        self.directory=OUT/self._testMethodName
        self.directory.mkdir()
        self.log=self.directory/'client.jsonl'

    def api(self,server,log=None):
        self.assertEqual(server.server.server_address[0],'127.0.0.1')
        self.assertNotIn(server.server.server_address[1],(0,2026))
        api=client.Q4Client('local-test-team',server.url,log or self.log,
                            http_timeout_s=.5,max_attempts=2,exit_reserve_s=.05)
        self.addCleanup(api.close_log)
        return api

    def records(self,path=None):
        return [json.loads(line) for line in (path or self.log).read_text(encoding='utf-8').splitlines()]

    def save_scenario(self,scenario,name='fixture.json'):
        write(self.directory/name,dict(events=scenario.arena.events,
            requests=[dict(path=p,body=b.decode('utf-8')) for p,b,_ in scenario.requests],
            executions=dict(scenario.executions),active=scenario.active,local_only=True))

    def test_real_runtime_hashes_and_unchanged_transport(self):
        self.assertTrue(issubclass(client.Q4Client,original.Q4Client))
        for name in ('ClientError','solver_api','check_completion','transport'):
            self.assertIs(getattr(client,name),getattr(original,name))
        expected={'negative_region.py','ordered_optical_prune.py','negative_region_proof.py',
                  'prune_proof.py','q4_share25_prune_client.py','official_client.py','geometry.py'}
        hashes=client.source_hashes()
        self.assertTrue(expected <= {Path(k).name for k in hashes})
        self.assertNotIn('q4_model/task_sharing_solver.py',hashes)
        for name,digest in hashes.items():
            self.assertEqual(digest,hashlib.sha256((PROJECT/'src'/name).read_bytes()).hexdigest())
        write(self.directory/'runtime-hashes.json',hashes)

    def test_default_dispatch_incomplete_is_not_certified(self):
        scenario=fixture.ArenaScenario(LocalArena([]))
        with fixture.LocalServer(scenario) as server:
            api=self.api(server)
            with patch.object(ordered_optical_prune,'solve_multi',return_value=dict(complete=False,reason='dispatch_only')) as actual:
                summary=client.run_session(api)
            self.assertEqual(actual.call_args.kwargs['variant'],'share25')
            public=actual.call_args.args[0]
            for name in ('accepted_actions','negative_points','time_s','detected_channels','source_count','cleared_channels'):
                self.assertFalse(hasattr(public,name))
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['prune_audit']['passed'])
            self.assertIsNone(summary['optical_prune_skipped_before_success'])
            self.assertTrue(summary['exit_accepted'])
            self.assertTrue(api._log_stream.closed)
        records=self.records()
        self.assertEqual(records[0]['variant'],'share25_prune')
        self.assertEqual(records[-1]['variant'],'share25_prune')
        self.save_scenario(scenario)

    def test_same_request_retry_executes_once_and_clear_keeps_channel(self):
        scenario=fixture.ArenaScenario(LocalArena([dict(channel=2,position=[0.,0.],radius=1000.,orientation=None)]))
        scenario.drop_first_measure=True
        with fixture.LocalServer(scenario) as server:
            api=self.api(server)
            api.enter()
            self.assertEqual(api.measure([0.,0.],2)['measure_result'],'near')
            self.assertEqual(api.channel,2)
            self.assertEqual(api.clear([0.,0.],1)['clear_result'],'no_target_in_range')
            self.assertEqual(api.channel,2)
            self.assertEqual(api.clear([0.,0.],2)['clear_result'],'success')
            self.assertEqual(api.channel,2)
            self.assertEqual(len(api.accepted_actions),3)
            self.assertEqual(api.accepted_actions[0]['measure_result'],'near')
            self.assertEqual(api.accepted_actions[1]['clear_result'],'no_target_in_range')
            self.assertEqual(api.time_s,14.)
            api.exit()
        measures=[r for r in scenario.requests if r[0]=='/measure']
        self.assertEqual(len(measures),2)
        self.assertEqual(measures[0][1],measures[1][1])
        self.assertEqual(scenario.executions['/measure'],1)
        self.save_scenario(scenario)

    def test_old_log_or_trace_is_not_overwritten(self):
        scenario=fixture.ArenaScenario(LocalArena([]))
        with fixture.LocalServer(scenario) as server:
            self.log.write_text('OLD_LOG',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                self.api(server)
            self.assertEqual(self.log.read_text(encoding='utf-8'),'OLD_LOG')
            another=self.directory/'old-trace.jsonl'
            trace=Path(str(another)+'.beliefs.json')
            trace.write_text('OLD_TRACE',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                self.api(server,another)
            self.assertFalse(another.exists())
            self.assertEqual(trace.read_text(encoding='utf-8'),'OLD_TRACE')
            self.assertEqual(scenario.requests,[])

    def test_original_solver_errors_and_partial_trace_survive_cleanup(self):
        for i,(error,failed_exit) in enumerate(((ValueError('geometry failure'),False),
                (ValueError('geometry failure plus exit failure'),True),(KeyboardInterrupt('interrupt fixture'),False))):
            with self.subTest(i=i):
                scenario=fixture.ArenaScenario(LocalArena([]))
                if failed_exit:
                    scenario.close_paths.add('/exit')
                log=self.directory/f'error-{i}.jsonl'
                with fixture.LocalServer(scenario) as server:
                    api=self.api(server,log)
                    def failing(public,**options):
                        options['trace'].append(dict(phase='saved_before_error',index=i))
                        raise error
                    with self.assertRaises(type(error)) as caught:
                        client.run_session(api,failing)
                    self.assertIs(caught.exception,error)
                    self.assertTrue(api._log_stream.closed)
                self.assertEqual(json.loads(api.trace_path.read_text(encoding='utf-8')),[dict(phase='saved_before_error',index=i)])
                self.assertFalse(any(r.get('type')=='session_summary' for r in self.records(log)))
                self.assertTrue(any(r.get('type')=='client_error' for r in self.records(log)))
                self.assertEqual(len({r['request_id'] for p,_,r in scenario.requests if p=='/exit'}),1)
                if failed_exit:
                    self.assertTrue(error.__notes__)
                self.save_scenario(scenario,f'error-{i}-fixture.json')

    def test_uncertain_enter_or_measure_does_not_send_new_exit(self):
        for path in ('/enter','/measure'):
            with self.subTest(path=path):
                scenario=fixture.ArenaScenario(LocalArena([]))
                scenario.close_paths.add(path)
                log=self.directory/(path[1:]+'.jsonl')
                with fixture.LocalServer(scenario) as server:
                    api=self.api(server,log)
                    def one_measure(public,**options):
                        self.assertNotEqual(path,'/enter')
                        options['trace'].append(dict(phase='before_uncertain_measure'))
                        public.measure([0.,0.],1)
                    with self.assertRaises(client.transport.TransportFailure):
                        client.run_session(api,one_measure)
                    self.assertTrue(api._log_stream.closed)
                    self.assertEqual(api.accepted_actions,[])
                self.assertNotIn('/exit',[p for p,_,_ in scenario.requests])
                self.assertFalse(any(r.get('type')=='session_summary' for r in self.records(log)))
                self.save_scenario(scenario,path[1:]+'-fixture.json')

    def test_rejected_or_malformed_measure_not_added_to_accepted_ledger(self):
        for mode,expected in (('reject',client.transport.ActionRejected),('malformed',client.transport.ResponseProtocolError)):
            with self.subTest(mode=mode):
                scenario=fixture.ArenaScenario(LocalArena([]))
                log=self.directory/(mode+'.jsonl')
                with fixture.LocalServer(scenario) as server:
                    api=self.api(server,log)
                    def bad_measure(public,**options):
                        if mode=='reject':
                            scenario.reject_next=True
                        else:
                            scenario.malformed_next=True
                        public.measure([0.,0.],1)
                    with self.assertRaises(expected):
                        client.run_session(api,bad_measure)
                    self.assertEqual(api.accepted_actions,[])
                    self.assertTrue(api._log_stream.closed)
                self.assertNotIn('/exit',[p for p,_,_ in scenario.requests])
                self.save_scenario(scenario,mode+'-fixture.json')

    def test_complete_but_missing_actual_trace_is_rejected(self):
        # This tiny fixture is a provenance challenge, not a contest score.
        sources=[dict(channel=c,position=[c/10.,0.],radius=1000.,orientation=None) for c in range(1,17)]
        scenario=fixture.ArenaScenario(LocalArena(sources))
        with fixture.LocalServer(scenario) as server:
            api=self.api(server)
            def forged_trace(public,**options):
                for c in range(1,17):
                    self.assertEqual(public.measure([0.,0.],c)['measure_result'],'near')
                    self.assertEqual(public.clear([0.,0.],c)['clear_result'],'success')
                # Clearing is real, but the claimed empty trace omits every
                # accepted action and must fail the complete-proof audit.
                return dict(complete=True,cleared=list(range(1,17)),certificate=dict(
                    basis='count_upper_bound',cleared=list(range(1,17))))
            with self.assertRaises(client.ClientError):
                client.run_session(api,forged_trace)
            self.assertEqual(len(api.accepted_actions),32)
            self.assertTrue(api._log_stream.closed)
        self.assertEqual(scenario.executions['/exit'],1)
        self.assertFalse(any(r.get('type')=='session_summary' for r in self.records()))
        self.save_scenario(scenario)

    def test_hash_drift_is_not_reported_successfully(self):
        scenario=fixture.ArenaScenario(LocalArena([]))
        original_hash=client.source_hashes()
        changed=dict(original_hash,synthetic_changed_dependency='fixture_only')
        with fixture.LocalServer(scenario) as server:
            api=self.api(server)
            with patch.object(client,'source_hashes',side_effect=[original_hash,changed]):
                with self.assertRaises(client.ClientError):
                    client.run_session(api,lambda public,**kw:dict(complete=False,reason='hash_fixture'))
            self.assertTrue(api._log_stream.closed)
        self.assertFalse(any(r.get('type')=='session_summary' for r in self.records()))
        self.assertEqual(scenario.executions['/exit'],1)
        self.save_scenario(scenario)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    OUT=args.out.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    snapshot=client.source_hashes()
    test_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PruneClientTests))
    stable=snapshot==client.source_hashes() and test_hash==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(passed=result.wasSuccessful() and stable,tests_run=result.testsRun,source_snapshot_stable=stable,
        source_sha256=snapshot,test_sha256=test_hash,local_only=True,official_contacted=False,
        failures=[dict(test=str(t),detail=d) for t,d in result.failures],
        errors=[dict(test=str(t),detail=d) for t,d in result.errors])
    write(OUT/'summary.json',report)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
