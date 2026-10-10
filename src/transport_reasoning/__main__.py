"""Command-line interface with validated configs and explicit task boundaries."""
import argparse
import json
import os
import sys
from pathlib import Path
from transport_reasoning.config import ROOT,load,write_json


def check_artifacts():
    # Optional site-specific forbidden names are supplied externally, never embedded in source.
    words=[s.casefold() for s in os.environ.get("TRANSPORT_FORBIDDEN_TERMS","").split(",") if s]
    failures=[]; checked=0
    for root in (Path("src"),Path("tests"),Path("configs"),Path("scripts")):
        for p in root.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts: continue
            if p.suffix in (".pdf",".xlsx",".pt",".npz") or p.stat().st_size>5_000_000:
                failures.append({"file":str(p),"reason":"external_or_large_artifact"}); continue
            text=p.read_text(errors="replace").casefold(); checked+=1
            if any(word in text or word in str(p).casefold() for word in words):
                failures.append({"file":str(p),"reason":"naming_rule"})
    for p in [*Path(".").glob("*.md"),*ROOT.glob("*.md"),*ROOT.glob("*.json")]:
        checked+=1
        if any(word in p.read_text().casefold() for word in words): failures.append({"file":str(p),"reason":"naming_rule"})
    if failures: raise ValueError(json.dumps(failures))
    return {"files_checked":checked,"failures":0,"external_term_list_provided":bool(words),
            "scope":"source/config/report scan; manual review also required"}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m transport_reasoning")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Runtime and allocation inventory")
    for cmd in ("prepare-data","train","evaluate","run-queue"):
        p=sub.add_parser(cmd); p.add_argument("--config",required=True)
        if cmd=="train": p.add_argument("--resume"); p.add_argument("--stop-after",type=int)
        if cmd=="evaluate":
            p.add_argument("--checkpoint",required=True); p.add_argument("--split",choices=("train","validation","test"),default="validation")
        if cmd=="run-queue": p.add_argument("--dry-run",action="store_true")
    p=sub.add_parser("summarize"); p.add_argument("--runs-dir",default=str(ROOT/"runs"))
    sub.add_parser("check-artifacts")
    sub.add_parser("audit-mlp",help="Small float64 checks, not task evidence")
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            from transport_reasoning.training import environment
            result=environment(); write_json(ROOT/"environment.json",result)
        elif args.command=="prepare-data":
            from transport_reasoning.data import prepare
            result=prepare(load(args.config))
        elif args.command=="train":
            from transport_reasoning.training import train
            result=train(load(args.config),args.resume,args.stop_after)
        elif args.command=="evaluate":
            from transport_reasoning.training import evaluate_checkpoint
            result=evaluate_checkpoint(load(args.config),args.checkpoint,args.split)
        elif args.command=="run-queue":
            from transport_reasoning.orchestration import queue
            result=queue(args.config,args.dry_run)
        elif args.command=="summarize":
            from transport_reasoning.reporting import summarize
            result=summarize(args.runs_dir)
        elif args.command=="audit-mlp":
            from transport_reasoning.models.mlp_audit import audit
            result=audit(); write_json(ROOT/"mlp_algebra_audit.json",result)
        else: result=check_artifacts()
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as e:
        # Avoid machine paths in user-facing errors. Full error types are always retained.
        print(json.dumps({"status":"failed","error_type":type(e).__name__,"message":str(e).replace(str(Path.cwd()),".").replace(str(Path.home()),"~")}),file=sys.stderr)
        raise SystemExit(1)


if __name__=="__main__": main()
