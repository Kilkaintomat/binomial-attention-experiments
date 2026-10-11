import argparse
import os
from transport_reasoning.reachability.campaign import run_queue
from transport_reasoning.reachability.config import root,write_json

parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,required=True);parser.add_argument('--remaining-seconds',type=float,required=True);args=parser.parse_args()
status=run_queue(args.seconds,args.remaining_seconds)
write_json(root()/('queue/slot_result_'+os.environ['SLURM_JOB_ID']+'.json'),{'status':status})
print({'status':status},flush=True)
