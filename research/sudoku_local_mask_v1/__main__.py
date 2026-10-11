import argparse
import json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['checks','overfit','profile','prepare','train','evaluate','report','submit','gate','account','slot','allow','run'])
    p.add_argument('--label',choices=['F4-onehop','F4-binomial','R32-onehop'])
    p.add_argument('--support',choices=['graph','grid4'],default='graph')
    p.add_argument('--config');p.add_argument('--resume',action='store_true')
    p.add_argument('--seconds',type=int,default=6900);p.add_argument('--remaining-seconds',type=float)
    p.add_argument('--gpu-hours',type=float);p.add_argument('--wall-hours',type=float);p.add_argument('--job')
    args=p.parse_args()
    if args.command=='checks':
        from .checks import run
        result=run()
    elif args.command=='overfit':
        from .measure import overfit
        result=overfit(args.label,args.support)
    elif args.command=='profile':
        from .measure import profile
        result=profile()
    elif args.command in ['submit','gate','account','slot','allow']:
        from . import scheduler
        result=getattr(scheduler,args.command)(args)
    elif args.command=='prepare':
        from .campaign import prepare
        result=prepare()
    elif args.command=='report':
        from .reporting import report
        result=report()
    elif args.command=='run':
        import os
        from transport_reasoning.config import write_json
        from .config import root
        from .campaign import run_slot
        result={'status':run_slot(args.seconds,args.remaining_seconds)}
        write_json(root()/('queue/slot_result_'+os.environ['SLURM_JOB_ID']+'.json'),result)
    else:
        from .config import load
        from . import training
        c=load(args.config)
        result=training.train(c,resume=c.run_dir()/'checkpoint_last.pt' if args.resume else None) if args.command=='train' else training.evaluate(c)
    if result is not None: print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
