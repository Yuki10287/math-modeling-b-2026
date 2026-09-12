"""Check migration/runtime boundaries with a tiny isolated local fixture."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import study_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.group = self.root/'src/q/00_study'
        (self.group/'code').mkdir(parents=True)
        (self.group/'results').mkdir()
        (self.group/'code/model.py').write_bytes(b'VALUE = 42\r\n')
        (self.group/'results/frozen.json').write_bytes(b'{"value": 42}\r\n')
        (self.group/'results/README.md').write_text('batch navigation')
        self.manifest = dict(groups={'demo':dict(path='src/q/00_study', logical_root='src/q')}, files=[
            dict(logical='src/q/model.py', path='src/q/00_study/code/model.py', study='demo'),
            dict(logical='src/q/results/frozen.json',path='src/q/00_study/results/frozen.json',study='demo')])
        self.index = self.root/'src/studies.json'
        self.index.write_text(json.dumps(self.manifest),encoding='utf-8')
        self.patch_root = patch.object(runtime,'ROOT',self.root)
        self.patch_index = patch.object(runtime,'INDEX',self.index)
        self.patch_root.start();self.patch_index.start()
        self.addCleanup(self.patch_root.stop);self.addCleanup(self.patch_index.stop)

    def test_source_bytes_and_logical_import_location(self):
        assembled = runtime.assemble()
        self.assertEqual((assembled/'src/q/model.py').read_bytes(),b'VALUE = 42\r\n')
        self.assertEqual((assembled/'src/q/results/frozen.json').read_bytes(),b'{"value": 42}\r\n')
        self.assertEqual(runtime.resolve('src/q/model.py'),self.group/'code/model.py')

    def test_discover_new_code_and_ignore_navigation(self):
        (self.group/'code/new_model.py').write_text('VALUE=43')
        names={record['logical'] for record in runtime.index()['files']}
        self.assertEqual(names,{'src/q/model.py','src/q/new_model.py','src/q/results/frozen.json'})

    def test_duplicate_module_is_rejected(self):
        other = self.root/'src/q/01_other/code'
        other.mkdir(parents=True)
        (other/'model.py').write_text('VALUE=0')
        self.manifest['groups']['other']=dict(path='src/q/01_other',logical_root='src/q')
        self.index.write_text(json.dumps(self.manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            runtime.index()

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            runtime.safe_path(self.root,'../outside.py')

    def test_results_are_new_batch_and_original_evidence_unchanged(self):
        assembled=runtime.assemble();before=runtime.artifact_snapshot(assembled)
        (assembled/'src/q/results/frozen.json').write_text('{"value": 99}')
        (assembled/'src/q/results/new.json').write_text('{"new": true}')
        (assembled/'src/q/model.py').write_text('VALUE=0')
        copied=runtime.collect_outputs(assembled,'demo',before)
        self.assertEqual(len(copied),2)
        self.assertTrue(all('/results/local_runs/' in path for path in copied))
        self.assertEqual((self.group/'results/frozen.json').read_bytes(),b'{"value": 42}\r\n')
        self.assertEqual((self.group/'code/model.py').read_bytes(),b'VALUE = 42\r\n')
        unchanged=runtime.assemble()
        self.assertEqual(runtime.collect_outputs(unchanged,'demo',runtime.artifact_snapshot(unchanged)),[])

    def test_concurrent_original_edit_is_detected(self):
        assembled=runtime.assemble();before=runtime.artifact_snapshot(assembled)
        (assembled/'src/q/results/frozen.json').write_text('{"value": 99}')
        (self.group/'results/frozen.json').write_text('{"value": 88}')
        with self.assertRaisesRegex(RuntimeError,'changed during execution'):
            runtime.collect_outputs(assembled,'demo',before)


if __name__=='__main__':
    unittest.main()
