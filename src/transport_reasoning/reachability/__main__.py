"""CLI for the isolated reachability experiment."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare-data','audit-data','preflight','build-grid','profile','run-queue','evaluate','report','baselines'])
    parser.add_argument('--family',choices=['er','dag']);parser.add_argument('--n',type=int)
    parser.add_argument('--small',action='store_true');parser.add_argument('--stress',action='store_true')
    parser.add_argument('--config');parser.add_argument('--seed',type=int);parser.add_argument('--stage')
    parser.add_argument('--seconds',type=int,default=1500);parser.add_argument('--remaining-seconds',type=float,default=86400)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.command=='prepare-data':
        from .data import generate_size,generate_stress
        if args.dry_run:result={'family':args.family,'n':args.n,'small':args.small,'stress':args.stress}
        elif args.stress:result=generate_stress(args.small)
        elif args.family and args.n:result=generate_size(args.family,args.n,args.small)
        else:parser.error('prepare-data requires --family and --n, or --stress')
    elif args.command=='audit-data':
        from .data import audit_data
        result=audit_data(args.small)
    elif args.command=='build-grid':
        from .campaign import build_grid
        result=build_grid()
    elif args.command in ['preflight','profile']:
        from . import checks
        result=getattr(checks,args.command)(args)
    elif args.command=='run-queue':
        from .campaign import run_queue
        result=run_queue(args.seconds,args.remaining_seconds,args.dry_run)
    elif args.command=='evaluate':
        from .training import evaluate_config
        from .config import Config,read
        if not args.config:parser.error('evaluate requires --config')
        result=evaluate_config(Config(**read(args.config)).validate())
    elif args.command=='baselines':
        from .baseline import baselines
        result=baselines()
    else:
        from .reporting import report
        result=report()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
