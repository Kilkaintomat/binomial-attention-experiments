"""Single-process bounded queues with allocation accounting and durable registry."""
import fcntl
import json
import os
import time
from pathlib import Path
from transport_reasoning.config import ROOT,load,write_json
from transport_reasoning.training import train


def queue(path,dry_run=False):
    specification=json.loads(Path(path).read_text())
    configs=[load(p) for p in specification["configs"]]
    estimate=sum(specification.get("estimated_seconds",{}).get(c.hash(),c.max_seconds) for c in configs)
    if dry_run:
        return {"runs":[{"config_hash":c.hash(),"task":c.task,"method":c.method,"seed":c.seed,"steps":c.steps} for c in configs],
                "estimated_seconds":estimate,"ceiling_gpu_hours":specification.get("gpu_hours",24),
                "note":"A missing measured estimate is replaced by the per-run maximum, not zero."}
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"campaign.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=any(c.device=="cuda" for c in configs)
        if gpu and not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("A GPU allocation is required")
        budget_path=ROOT/"budget.json"
        budget=json.loads(budget_path.read_text()) if budget_path.exists() else {"ceiling_gpu_hours":24,"allocations":[]}
        limit=min(24,float(specification.get("gpu_hours",24)))*3600
        charged=sum(a.get("actual_seconds",a["reserved_seconds"]) for a in budget["allocations"])
        left=limit-charged if gpu else specification.get("cpu_seconds",3600)
        if left<=120: return {"status":"budget_exhausted"}
        started=time.perf_counter(); deadline=started+left
        allocation={"job_id":os.environ.get("SLURM_JOB_ID"),"reserved_seconds":left,"gpu_count":1,"status":"active"}
        if gpu:
            budget["allocations"].append(allocation)
            write_json(budget_path,budget)
        records=[]
        try:
            for c in configs:
                if time.perf_counter()>=deadline-120: break
                checkpoint=c.run_dir()/"checkpoint_last.pt"
                try:
                    result=train(c,resume=checkpoint if checkpoint.exists() else None,deadline=deadline)
                    item={"config_hash":c.hash(),"status":result["status"],"task":c.task,"method":c.method,"seed":c.seed}
                except Exception as e:
                    item={"config_hash":c.hash(),"status":"failed","error_type":type(e).__name__,"task":c.task,"method":c.method,"seed":c.seed}
                with (ROOT/"run_registry.jsonl").open("a") as f: f.write(json.dumps(item)+"\n")
                records.append(item)
                write_json(ROOT/"queue_state.json",records)
        finally:
            elapsed=time.perf_counter()-started
            if gpu:
                allocation.update(actual_seconds=elapsed,status="finished")
                budget["used_gpu_seconds"]=charged+elapsed
                write_json(budget_path,budget)
        return {"runs":records,"elapsed_seconds":elapsed}
