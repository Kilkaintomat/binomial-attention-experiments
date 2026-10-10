"""Sequential first campaign; no test access before a frozen final protocol."""
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.config import ROOT,METHODS,Config,load,write_json
from transport_reasoning.data import TaskData,prepare
from transport_reasoning.training import train,evaluate_checkpoint,environment
from transport_reasoning.orchestration.profile import profile
from transport_reasoning.reporting import summarize

ROOT.mkdir(parents=True,exist_ok=True)
lock=(ROOT/"campaign.lock").open("a+")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if not (ROOT/"ready_for_gpu.json").exists(): raise RuntimeError("CPU checks, datasets and pointer bridge have not passed")
if (ROOT/"frozen_protocol.json").exists():
    raise RuntimeError("Final protocol already frozen. Resume the saved frozen_queue.json; do not rerun tuning or overwrite the protocol.")
if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(): raise RuntimeError("One allocated GPU is required")
if torch.cuda.device_count()!=1: raise RuntimeError("Expected exactly one scheduler-visible GPU")
start=time.perf_counter(); utc_start=time.time(); job=os.environ["SLURM_JOB_ID"]
budget_path=ROOT/"budget.json"
budget=json.loads(budget_path.read_text()) if budget_path.exists() else {"ceiling_gpu_hours":24,"allocations":[]}
# Interrupted allocations retain their full reservation, preventing accidental budget reuse.
charged=sum(a.get("actual_seconds",a["reserved_seconds"]) for a in budget["allocations"])
available=min(23.5*3600,24*3600-charged)
if available<600: raise RuntimeError("Insufficient remaining campaign allocation budget")
allocation={"job_id":job,"gpu_count":1,"started_utc_epoch":utc_start,"reserved_seconds":available,"status":"active"}
budget["allocations"].append(allocation); budget["ceiling_gpu_hours"]=24
deadline=start+available-120
write_json(budget_path,budget)
registry=ROOT/"run_registry.jsonl"
config_dir=ROOT/"configs"; config_dir.mkdir(exist_ok=True)
stage="initializing"; outcomes=[]


def update_status(message):
    allocation["observed_seconds"]=time.perf_counter()-start
    budget["observed_gpu_seconds"]=charged+allocation["observed_seconds"]
    write_json(budget_path,budget)
    (ROOT/"STATUS.md").write_text(f"# Status\n\nStage: {stage}.\n\n{message}\n\nAllocated GPU seconds observed: {budget['observed_gpu_seconds']:.1f} / 86400.\n")
    print(json.dumps({"stage":stage,"message":message,"gpu_seconds":budget["observed_gpu_seconds"]}),flush=True)


def remaining(): return deadline-time.perf_counter()


def run(c,stop_after=None):
    if remaining()<180: return None
    c.validate(); write_json(config_dir/(c.hash()+".json"),c.dict())
    checkpoint=c.run_dir()/"checkpoint_last.pt"
    try:
        result=train(c,resume=checkpoint if checkpoint.exists() else None,stop_after=stop_after,deadline=deadline)
        event={"stage":stage,"task":c.task,"method":c.method,"seed":c.seed,"config_hash":c.hash(),"status":result["status"],"step":result["step"],"inner_beta":c.inner_beta}
    except Exception as e:
        event={"stage":stage,"task":c.task,"method":c.method,"seed":c.seed,"config_hash":c.hash(),"status":"failed","error_type":type(e).__name__}
        write_json(c.run_dir()/"failure.json",event)
        result=None
    with registry.open("a") as f: f.write(json.dumps(event)+"\n")
    outcomes.append(event); update_status(json.dumps(event)); return result


def altered(base,**kwargs):
    d=base.dict(); d.update(kwargs); return Config(**d).validate()


def run_campaign():
    global stage
    write_json(ROOT/"gpu_environment.json",environment())
    bases={task:load(f"configs/pilot_{task}.json") for task in ("maze","sudoku")}
    profiles={}; stage="profiling"
    profile_path=ROOT/"profiles.json"
    if profile_path.exists(): profiles=json.loads(profile_path.read_text())
    for task,base in bases.items():
        for method in METHODS:
            key=task+"/"+method
            if key not in profiles:
                c=altered(base,method=method,stage="profile")
                profiles[key]=profile(c)
                write_json(profile_path,profiles)
                update_status(f"Profile complete: {key}")
    costs={k:v["seconds_per_step"] for k,v in profiles.items()}
    proposed=sum(costs.values())*3*2000
    write_json(ROOT/"pilot_cost_estimate.json",{"training_seconds_upper_before_halving":proposed,
               "profile_seconds":time.perf_counter()-start,"remaining_seconds":remaining(),"GPU_count":1,
               "reserve_for_MLP_seconds":max(1800,12*500*(costs["maze/binomial"]+costs["sudoku/binomial"])*2)})
    stage="tiny_overfit"
    overfit={}
    for task,base in bases.items():
        for method in METHODS:
            c=altered(base,method=method,stage="overfit",overfit_count=64,steps=1000,eval_every=250,
                      lr=.001,max_seconds=max(900,1500*costs[task+"/"+method]))
            result=run(c)
            overfit[task+"/"+method]=None if result is None else result["best_validation_solved_rate"]
    write_json(ROOT/"overfit_audit.json",{"scores":overfit,"criterion":.95,"subset_size":64,
              "note":"Failures are retained. A failed memorization check limits the interpretation of subsequent task results."})
    # Common pilot protocol: all methods see the same LR budget and at least 500 steps.
    stage="lr_pilot"
    selected={}; pilot_results={}
    for task,base in bases.items():
        selected[task]={}
        for method in METHODS:
            candidates=[]
            for lr in (1e-4,3e-4,1e-3):
                c=altered(base,method=method,lr=lr,steps=2000,max_seconds=max(1800,2500*costs[task+"/"+method]))
                result=run(c,stop_after=500)
                if result:
                    metric=result.get("last_validation") or {}
                    candidates.append(((metric.get("solved_rate",-1),-metric.get("loss",1e9)),c))
            if not candidates: continue
            best=sorted(candidates,key=lambda item:item[0],reverse=True)[0][1]
            result=run(best)
            selected[task][method]=best.dict()
            pilot_results[task+"/"+method]=result
            write_json(ROOT/"selected_pilot_configs.json",selected)
    summarize()
    stage="horizon_pilot"
    # Shared H choice uses mean validation solved fraction, then mean loss across the three filters.
    families=("binomial","exponential_truncated","finite_geometric")
    chosen_h={}; horizon_scores={}
    for task,base in bases.items():
        scores=[]
        for hops in (4,8,16):
            methods=[]
            for method in families:
                if method not in selected[task]: continue
                c=altered(Config(**selected[task][method]),stage="horizon_pilot",hops=hops,steps=1000,
                          max_seconds=max(1800,1500*costs[task+"/"+method]*(hops/8)))
                result=run(c)
                if result and result.get("last_validation"):
                    metric=result["last_validation"]; methods.append((metric["solved_rate"],metric["loss"]))
            if len(methods)==3:
                scores.append((float(np.mean([p[0] for p in methods])),-float(np.mean([p[1] for p in methods])),hops))
        chosen_h[task]=max(scores)[2] if scores else 8
        horizon_scores[task]=scores
    write_json(ROOT/"horizon_selection.json",{"chosen_H":chosen_h,"scores":horizon_scores,"criterion":"mean validation solved, then mean negative loss; ties larger H"})
    # Reserve MLP pilot before planning final runs, so late stages cannot consume its budget.
    mlp_configs=[]
    for task,base in bases.items():
        for seed in (101,103):
            for method in families:
                for beta in (0.,.1):
                    if method in selected[task]:
                        mlp_configs.append(altered(Config(**selected[task][method]),stage="mlp_pilot",seed=seed,
                            hops=chosen_h[task],inner_beta=beta,inner_width=base.width,steps=500,
                            max_seconds=max(1800,1500*costs[task+"/"+method])))
    mlp_reserve=max(1800,sum(1.8*costs[c.task+"/"+c.method]*c.steps*(1+4*c.inner_beta)*(c.hops/8) for c in mlp_configs))
    stage="freeze_final_protocol"
    usable=max(0,remaining()-mlp_reserve-1800)
    denom=5*sum(costs[task+"/"+method]*max(1,chosen_h[task]/8) for task in bases for method in METHODS)
    steps=min(30000,int(usable/(1.6*denom)//250*250)) if denom else 0
    # Data volumes stay at the documented pilot-sized maze for this compute-bounded first campaign.
    frozen={"created_before_final_test":True,"seed_list":[11,23,37,53,71],"methods":list(METHODS),
            "common_H":chosen_h,"updates_per_run":steps,"effective_batch":64,"LR_selection":selected,
            "data_manifests":{t:json.loads((b.data_dir()/"manifest.json").read_text()) for t,b in bases.items()},
            "checkpoint_rule":"maximum validation solved rate; ties minimum validation loss",
            "thresholds":{"maze":[.9],"sudoku":[.5,.9]},"threshold_confirmations":2,
            "practical_effect":{"solved_rate":.02,"training_time_fraction":.1},"equivalence_claim_authorized":False,
            "pilot_seeds_excluded":[101,103],"architecture":{"L":4,"width":128,"heads":4,"recurrent_T":32,"gain":1},
            "amendments":["Maze remains 8x8 with 10k train/1k validation/1k IID test after runtime profiling; mixed 8/12 training is deferred.",
                          "Sudoku preserves all official RRN splits; generated 4x4 is smoke only.",
                          "Shared coefficients per block across heads, including learned polynomial.",
                          "Recurrent T=32 is a declared equal-radius control, not validation-tuned T or parameter-matched width.",
                          "Final update cap reduced from 30k only by measured total cost before final seeds and test."]}
    write_json(ROOT/"frozen_protocol.json",frozen)
    final_configs=[]
    if steps>=1000 and all(len(selected[t])==6 for t in bases):
        for task in ("maze","sudoku"):
            for seed in (11,23,37,53,71):
                for method in METHODS:
                    final_configs.append(altered(Config(**selected[task][method]),stage="final",seed=seed,
                        hops=chosen_h[task],steps=steps,eval_every=250,
                        max_seconds=max(1800,3*steps*costs[task+"/"+method]*max(1,chosen_h[task]/8))))
    write_json(ROOT/"planned_final_queue.json",[c.dict() for c in final_configs])
    write_json(ROOT/"planned_mlp_queue.json",[c.dict() for c in mlp_configs])
    for c in final_configs+mlp_configs:
        write_json(config_dir/(c.hash()+".json"),c.dict())
    write_json(ROOT/"frozen_queue.json",{"configs":[str(config_dir/(c.hash()+".json")) for c in final_configs+mlp_configs],
               "gpu_hours":24,"estimated_seconds":{c.hash():costs[c.task+"/"+c.method]*c.steps*2 for c in final_configs+mlp_configs}})
    stage="final_matched_comparison"
    completed_final=[]
    for c in final_configs:
        if remaining()<mlp_reserve+300: break
        result=run(c)
        if result and result["status"]=="completed":
            evaluate_checkpoint(c,c.run_dir()/"checkpoint_best.pt",split="test")
            completed_final.append(c.hash())
            summarize()
    stage="mlp_pilot"
    for c in mlp_configs:
        if remaining()<180: break
        run(c)
    stage="reporting"
    summarize()
    remaining_configs=[c.dict() for c in final_configs if c.hash() not in completed_final]
    write_json(ROOT/"remaining_queue.json",{"unfinished_final_configs":remaining_configs,
               "deferred":["mixed-size maze training","maze 16x16 and Wilson OOD","dense attention controls",
                           "recurrent T tuning and parameter matching","trained-checkpoint frozen-FFN audits",
                           "sparsification and approximate evaluation GPU timings","softmax-memory replacement"],
               "reason":"First campaign prioritizes correctness, both task pilots, six primary controls, five seeds and the short MLP pilot.",
               "final_runs_expected":60,"final_runs_completed":len(completed_final)})
    return {"final_runs_completed":len(completed_final),"final_runs_expected":60,"final_steps":steps,
            "status":"completed_primary_runs" if len(completed_final)==60 else "preliminary_partial_coverage"}


try:
    outcome=run_campaign()
    write_json(ROOT/"campaign_result.json",outcome)
    update_status(json.dumps(outcome))
except Exception as e:
    write_json(ROOT/"campaign_failure.json",{"stage":stage,"error_type":type(e).__name__,
               "message":str(e).replace(str(Path.cwd()),".").replace(str(Path.home()),"~")})
    update_status("Campaign stopped with error type "+type(e).__name__)
    raise
finally:
    allocation["actual_seconds"]=time.perf_counter()-start
    allocation["status"]="finished"
    budget["used_gpu_seconds"]=sum(a.get("actual_seconds",a["reserved_seconds"]) for a in budget["allocations"])
    write_json(budget_path,budget)
