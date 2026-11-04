"""Safety gates are checked without calling Slurm or touching real ledgers."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import controller
from common import atomic


class SchedulerSafety(unittest.TestCase):
    def test_pending_reservation_and_terminal_actual_cost(self):
        ledger={'allocations':[{'job_id':'1','reserved_seconds':7200},{'job_id':'2','reserved_seconds':7200}]}
        with patch.object(controller,'capture',return_value='1|PENDING|0|\n2|COMPLETED|125|'):
            controller.account(ledger)
        self.assertEqual(controller.charged(ledger),7325)
        self.assertNotIn('actual_seconds',ledger['allocations'][0])
    def test_old_mlp_blocks_all_new_gpu_submissions(self):
        with tempfile.TemporaryDirectory() as d:
            art=Path(d);(art/'claims').mkdir();(art/'workers').mkdir()
            atomic(art/'cpu_preflight.json',{'status':'passed','source_sha256':'abc'})
            atomic(art/'budget.json',{'allocations':[]})
            with patch.object(controller,'ART',art),patch.object(controller,'validate',return_value={'source_sha256':'abc'}),\
                 patch.object(controller,'queue',return_value={}),patch.object(controller,'prior_complete',return_value=(False,14)),\
                 patch.object(controller,'watch') as watch,patch.object(controller,'capture') as capture:
                controller.tick()
                watch.assert_called_once_with(delay=True);capture.assert_not_called()
            self.assertEqual(controller.read(art/'status.json')['status'],'waiting_existing_mlp')
    def test_wrong_source_rejected_before_scheduler_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            art=Path(d);atomic(art/'cpu_preflight.json',{'status':'passed','source_sha256':'old'})
            with patch.object(controller,'ART',art),patch.object(controller,'validate',return_value={'source_sha256':'new'}),\
                 patch.object(controller,'capture') as capture:
                with self.assertRaises(RuntimeError):controller.tick()
                capture.assert_not_called()
    def test_only_terminal_owned_claims_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            art=Path(d);(art/'claims/active').mkdir(parents=True);(art/'claims/ended').mkdir()
            atomic(art/'claims/active/owner.json',{'job_id':'1'})
            atomic(art/'claims/ended/owner.json',{'job_id':'2'})
            ledger={'allocations':[{'job_id':'1','scheduler_state':'RUNNING'},{'job_id':'2','scheduler_state':'TIMEOUT'}]}
            with patch.object(controller,'ART',art):controller.recover_claims(ledger,{'1':{}})
            self.assertTrue((art/'claims/active').exists());self.assertFalse((art/'claims/ended').exists())
            self.assertTrue((art/'recovered_claims/ended-2.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
