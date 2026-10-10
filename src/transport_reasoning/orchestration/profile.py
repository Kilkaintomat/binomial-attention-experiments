"""Training profiles include data preparation and synchronized GPU work."""
import time
import numpy as np
import torch
from transport_reasoning.training import setup
from transport_reasoning.data import TaskData
from transport_reasoning.models import Encoder
from transport_reasoning.evaluation import loss_for,sync


def profile(c,warmup=20,steps=120):
    device=setup(c); data=TaskData(c,"train"); model=Encoder(c).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    rng=np.random.default_rng(492)
    samples=[]
    for i in range(warmup+steps):
        if i==warmup and device.type=="cuda": torch.cuda.reset_peak_memory_stats()
        sync(device); start=time.perf_counter()
        batch=data.batch(rng.integers(0,len(data),size=c.batch_size),device)
        optimizer.zero_grad(set_to_none=True)
        loss=loss_for(model(batch),batch,c,data.manifest.get("pos_weight",1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),c.clip,error_if_nonfinite=True); optimizer.step()
        sync(device)
        if i>=warmup: samples.append(time.perf_counter()-start)
    radius=c.recurrent_steps if c.method=="recurrent" else c.layers*(1 if c.method=="onehop" else c.hops)
    return {"task":c.task,"method":c.method,"config":c.dict(),"warmup_steps":warmup,"measured_steps":steps,
            "seconds_per_step":float(np.mean(samples)),"median_seconds_per_step":float(np.median(samples)),
            "p95_seconds_per_step":float(np.quantile(samples,.95)),"samples":samples,
            "examples_per_second":c.batch_size/float(np.mean(samples)),"parameters":model.parameter_count(),
            "peak_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
            "gpu_model":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
            "maximum_graph_radius":radius if c.support=="graph" else None,
            "training_paths_longer_than_radius_fraction":float(((data.data["y"].sum(1)-1)>radius).mean()) if c.task=="maze" and c.support=="graph" else None}
