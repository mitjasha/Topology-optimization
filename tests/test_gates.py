import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from topopt.gates import verify_benchmark


class Gates(unittest.TestCase):
    def test_strict_benchmark_cli_rejects_numerical_failure(self):
        from topopt.benchmark import main
        with patch('topopt.benchmark.run',return_value=[{'quality_gate_pass':False}]) as run:
            with self.assertRaises(SystemExit) as error:
                main(['--strict','--output','unused-test-output'])
            self.assertEqual(error.exception.code,2)
            self.assertTrue(run.called)
    def test_cli_fails_on_missing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            proc=subprocess.run([sys.executable,'-m','topopt.gates',folder],capture_output=True)
            self.assertEqual(proc.returncode,2)

    def test_numerical_failure_and_nan_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            for name in ('dataset.npz','ridge_model.npz','report.html'):(path/name).touch()
            meta={'split':{'train':[0],'validation':[1],'test':[2]},'cases':[{}, {}, {}],'group_ids':[0,1,2]}
            (path/'manifest.json').write_text(json.dumps(meta))
            row={'case_id':2,'hybrid_compliance_ratio':1.0,'volume_error':0.0,'reference_compliance':10,
                 'hybrid_equilibrium_error':0.0,'quality_gate_pass':True,'reference_converged':True,'hybrid_converged':True}
            def save():
                with (path/'test_results.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
            save();self.assertEqual(verify_benchmark(path),1)
            for value in (1.2,float('nan')):
                row['hybrid_compliance_ratio']=value;save()
                with self.assertRaises(ValueError):verify_benchmark(path)
            row['hybrid_compliance_ratio']=1.0;row['quality_gate_pass']=False;save()
            with self.assertRaises(ValueError):verify_benchmark(path)
