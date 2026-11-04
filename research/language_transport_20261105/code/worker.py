import argparse
import gc
import os
import signal
import time
import traceback

from common import ART,SIDE,atomic,validate,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,required=True);p.add_argument('--role',required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    import torch
    import engine
    from plan import next_stage,completed_training,complete
    from gpu_check import check
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    signal.signal(signal.SIGTERM,engine.stopping);signal.signal(signal.SIGUSR1,engine.stopping)
    plan=validate();job=os.environ['SLURM_JOB_ID'];deadline=time.monotonic()+args.seconds-60
    state=ART/'workers'/(job+'.json')
    atomic(state,dict(status='running',role=args.role,started_epoch=time.time()))
    try:
        if args.role=='preflight':check(deadline)
        else:
            while not engine.STOP and time.monotonic()<deadline-180:
                stage,items=next_stage()
                if stage not in ['pilot','scratch','pretrained']:break
                worked=False
                for s in items:
                    done=completed_training(s) if stage=='pilot' else complete(s)
                    if done:continue
                    claim=ART/'claims'/s['id'].replace('/','__')
                    try:claim.mkdir(parents=False)
                    except FileExistsError:continue
                    atomic(claim/'owner.json',dict(job_id=job,run=s['id'],epoch=time.time()))
                    try:
                        atomic(state,dict(status='running',role=args.role,stage=stage,run_id=s['id'],epoch=time.time()))
                        if not s.get('frozen'):
                            trained=engine.train(s,deadline)
                            if not trained:return
                        if stage!='pilot':
                            if not engine.evaluate(s,deadline):return
                        worked=True
                    finally:
                        (claim/'owner.json').unlink();claim.rmdir()
                    if engine.STOP or time.monotonic()>deadline-180:break
                if not worked:break
    except BaseException as exc:
        atomic(state,dict(status='failed',role=args.role,error=repr(exc),traceback=traceback.format_exc(),epoch=time.time()))
        raise
    finally:
        previous=read(state)
        if previous.get('status')!='failed':atomic(state,dict(previous,status='finished_slice',ended_epoch=time.time()))


if __name__=='__main__':main()
