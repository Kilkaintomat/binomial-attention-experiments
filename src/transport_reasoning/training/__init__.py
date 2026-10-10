"""Full-state resume, paired data streams, and auditable timing."""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.config import ROOT, write_json
from transport_reasoning.data import TaskData,prepare
from transport_reasoning.models import Encoder
from transport_reasoning.evaluation import evaluate_model,loss_for,sync


def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path("src").rglob("*.py")):
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    return {"python":platform.python_version(),"platform":platform.system(),
            "packages":{n:importlib.metadata.version(n) for n in ("torch","numpy","matplotlib","pytest")},
            "cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
            "gpu_model":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "commit":os.environ.get("TRANSPORT_COMMIT") or None,"source_sha256":source_hash(),
            "scheduler_job_id":os.environ.get("SLURM_JOB_ID"),"dtype":"float32",
            "backend":"dense_masked_generic_action","coefficient_sharing":"one vector per block shared across heads",
            "tf32":False,"threads":torch.get_num_threads()}


def setup(c):
    torch.set_num_threads(c.threads)
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    if c.device=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("GPU runs require a scheduler allocation")
        torch.cuda.manual_seed_all(c.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return torch.device(c.device)


def atomic_checkpoint(path,state):
    temporary=path.with_suffix(".tmp")
    torch.save(state,temporary); temporary.replace(path)


def restore_rng(state,generator):
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    generator.bit_generator.state=state["data_rng"]
    if state["cuda_rng"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda_rng"])


def train(c,resume=None,stop_after=None,deadline=None):
    c.validate(); device=setup(c); manifest=prepare(c)
    directory=c.run_dir(); directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/"result.json"
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old["status"]=="completed": return old
    if (directory/"checkpoint_last.pt").exists() and resume is None:
        raise ValueError("Run exists; use explicit --resume")
    env=environment(); write_json(directory/"config.json",c.dict()); write_json(directory/"environment.json",env)
    model=Encoder(c).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    warmup=max(1,round(c.steps*c.warmup_fraction))
    def schedule(step):
        return (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,c.steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    generator=np.random.default_rng(c.seed+987123)
    step=0; best=(-1.,-float("inf")); training_seconds=0.; evaluation_seconds=0.; checkpoint_seconds=0.; elapsed_previous=0.
    if resume:
        state=torch.load(resume,map_location="cpu",weights_only=False)
        if state["config"]!=c.dict() or state["data_hash"]!=manifest["data_hash"]:
            raise ValueError("Resume config or data hash differs")
        if state["source_sha256"]!=env["source_sha256"]:
            raise ValueError("Resume source differs; create an explicitly amended run")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        for value in optimizer.state.values():
            for key,tensor in value.items():
                if torch.is_tensor(tensor): value[key]=tensor.to(device)
        step=state["step"]; best=tuple(state["best"])
        training_seconds=state["training_seconds"]; evaluation_seconds=state["evaluation_seconds"]
        checkpoint_seconds=state["checkpoint_seconds"]; elapsed_previous=state["elapsed_seconds"]
        restore_rng(state,generator)
        # Discard metrics beyond the last atomic checkpoint after an interruption.
        metrics=directory/"metrics.jsonl"
        if metrics.exists():
            kept=[line for line in metrics.read_text().splitlines() if json.loads(line)["step"]<=step]
            metrics.write_text("\n".join(kept)+("\n" if kept else ""))
    train_data=TaskData(c,"train"); validation=TaskData(c,"validation")
    if c.overfit_count:
        if c.stage!="overfit": raise ValueError("A tiny training subset is reserved for the overfit stage")
        train_data.subset(c.overfit_count)
        validation=TaskData(c,"train").subset(c.overfit_count)
    start=time.perf_counter(); last_metric=None; status="running"
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    def save(is_best=False):
        nonlocal checkpoint_seconds
        tick=time.perf_counter()
        state={"config":c.dict(),"data_hash":manifest["data_hash"],"source_sha256":env["source_sha256"],
               "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "step":step,"best":list(best),"training_seconds":training_seconds,"evaluation_seconds":evaluation_seconds,
               "checkpoint_seconds":checkpoint_seconds,"elapsed_seconds":elapsed_previous+time.perf_counter()-start,
               "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else None,
               "data_rng":generator.bit_generator.state,"filters":model.filter_summary()}
        atomic_checkpoint(directory/"checkpoint_last.pt",state)
        if is_best: atomic_checkpoint(directory/"checkpoint_best.pt",state)
        checkpoint_seconds+=time.perf_counter()-tick
    target=min(c.steps,stop_after) if stop_after else c.steps
    while step<target:
        now=time.perf_counter()
        if now-start>=c.max_seconds or (deadline is not None and now>=deadline-90):
            status="budget_paused"; save(); break
        sync(device); tick=time.perf_counter()
        indices=generator.integers(0,len(train_data),size=c.batch_size)
        batch=train_data.batch(indices,device); model.train(); optimizer.zero_grad(set_to_none=True)
        logits=model(batch); loss=loss_for(logits,batch,c,manifest.get("pos_weight",1.))
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.clip,error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); step+=1
        sync(device); training_seconds+=time.perf_counter()-tick
        if step==1 or step%c.eval_every==0 or step==target:
            train_memory=torch.cuda.max_memory_allocated() if device.type=="cuda" else None
            metric=evaluate_model(model,validation,c,device)
            evaluation_seconds+=metric["evaluation_seconds"]
            metric.update(step=step,examples=step*c.batch_size,training_seconds=training_seconds,
                          train_loss=float(loss.item()),gradient_norm=float(norm),lr=optimizer.param_groups[0]["lr"],
                          filters=model.filter_summary(),train_peak_memory_bytes=train_memory)
            rank=(metric["solved_rate"],-metric["loss"]); improved=rank>best
            if improved: best=rank
            with open(directory/"metrics.jsonl","a") as f: f.write(json.dumps(metric,allow_nan=False)+"\n")
            with open(directory/"stdout.log","a") as f: f.write(json.dumps({k:metric[k] for k in ("step","train_loss","solved_rate","training_seconds")})+"\n")
            save(improved); last_metric=metric
    if step>=c.steps: status="completed"
    elif status=="running": status="paused"
    peak=torch.cuda.max_memory_allocated() if device.type=="cuda" else None
    result={"status":status,"config":c.dict(),"config_hash":c.hash(),"data_hash":manifest["data_hash"],
            "commit":env["commit"],"source_sha256":env["source_sha256"],"step":step,"examples":step*c.batch_size,
            "parameter_count":model.parameter_count(),"training_seconds":training_seconds,
            "evaluation_seconds":evaluation_seconds,"checkpoint_seconds":checkpoint_seconds,
            "elapsed_seconds":elapsed_previous+time.perf_counter()-start,"train_peak_memory_bytes":peak,
            "gpu_seconds":elapsed_previous+time.perf_counter()-start if device.type=="cuda" else 0,
            "best_checkpoint_rule":"maximum validation solved rate; ties minimum validation loss; earliest exact tie",
            "best_validation_solved_rate":best[0],"filters":model.filter_summary(),
            "failure_reason":None,"last_validation":last_metric}
    write_json(result_path,result)
    return result


def evaluate_checkpoint(c,path,split="validation"):
    device=setup(c); prepare(c)
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state["config"]!=c.dict(): raise ValueError("Checkpoint config differs")
    data=TaskData(c,split)
    if state["data_hash"]!=data.manifest["data_hash"]: raise ValueError("Checkpoint data differs")
    model=Encoder(c).to(device); model.load_state_dict(state["model"])
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    result=evaluate_model(model,data,c,device,Path(path).parent/f"predictions_{split}.npz")
    result.update(split=split,checkpoint_step=state["step"],data_hash=state["data_hash"],
                  checkpoint_filename=Path(path).name)
    write_json(Path(path).parent/f"evaluation_{split}.json",result)
    return result
