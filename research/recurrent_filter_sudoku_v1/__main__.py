import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','checks','overfit','profile','train','evaluate','report','submit','gate','account','slot','allow','run'])
    parser.add_argument('--config'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--label', choices=['R16','B16','R32','B32'])
    parser.add_argument('--seconds', type=int, default=6900)
    parser.add_argument('--remaining-seconds', type=float)
    parser.add_argument('--gpu-hours', type=float); parser.add_argument('--wall-hours', type=float)
    parser.add_argument('--job'); args = parser.parse_args()
    if args.command in ['submit','gate','account','slot','allow']:
        from . import scheduler
        result = getattr(scheduler, args.command)(args)
    elif args.command == 'run':
        import os
        from .campaign import run_slot
        from .config import root
        from transport_reasoning.config import write_json
        result = {'status': run_slot(args.seconds, args.remaining_seconds)}
        write_json(root() / ('queue/slot_result_' + os.environ['SLURM_JOB_ID'] + '.json'), result)
    elif args.command == 'prepare':
        from .campaign import prepare
        result = prepare()
    elif args.command == 'checks':
        from .campaign import check_all
        result = check_all()
    elif args.command in ['overfit','profile']:
        from . import measure
        result = measure.overfit(args.label) if args.command == 'overfit' else measure.profile()
    elif args.command == 'report':
        from .reporting import report
        result = report()
    else:
        from .config import load
        from . import training
        c = load(args.config)
        result = training.train(c, resume=c.run_dir() / 'checkpoint_last.pt' if args.resume else None) if args.command == 'train' else training.evaluate(c)
    if result is not None: print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
