"""Scheduler GPU gate: finite updates, tiny overfit and measured cost."""
import gc
import os
import time

import torch
from attention import install
from common import SIDE,ART,ROOT,atomic,rows,source_hash
from engine import load,batch_loss,validation,encode


def check(deadline):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    started=time.time();results=[]
    for task in ['vt','phop_conditional']:
        pool=rows(SIDE/'data'/task/'tiny.jsonl')
        for method in ['baseline','binomial']:
            spec=dict(model='scratch',method=method,seed=901,layers=2,r=8,loops=2)
            model,tok=load(spec);model.train()
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
            history=[];passed=False
            for step in range(1,2001):
                if time.monotonic()>deadline-180:raise TimeoutError('Tiny overfit exceeded allocated diagnostic budget')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=batch_loss(model,tok,pool,'cuda')
                if not torch.isfinite(loss):raise FloatingPointError('Tiny overfit non-finite')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%100==0:
                    metric=validation(model,tok,pool,n=8)
                    history.append(dict(step=step,loss=float(loss),**metric))
                    print('TINY',task,method,history[-1],flush=True)
                    if metric['accuracy']>=.875:passed=True;break
            results.append(dict(task=task,method=method,passed=passed,history=history))
            atomic(ART/'tiny_overfit.json',results)
            del model,opt;gc.collect();torch.cuda.empty_cache()
            if not passed:raise RuntimeError('Baseline/binomial tiny-set learnability gate failed; no main jobs permitted')
    benchmarks=[]
    for method in ['baseline','binomial','geometric','looped']:
        spec=dict(model='scratch',method=method,seed=902,layers=2,r=8,loops=2)
        model,tok=load(spec);model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        pool=rows(SIDE/'data/vt/train.jsonl')
        batch=[x for x in pool if x['context_words']==512][:8]
        durations=[];torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=batch_loss(model,tok,batch,'cuda')
            loss.backward();opt.step();torch.cuda.synchronize()
            if i>=2:durations.append(time.perf_counter()-t)
        benchmarks.append(dict(model='scratch',method=method,microbatch=8,context_words=512,
            seconds_per_microbatch=sum(durations)/len(durations),peak_bytes=torch.cuda.max_memory_allocated()))
        del model,opt;gc.collect();torch.cuda.empty_cache()
    # Verify the actual cached checkpoint/tokenizer on GPU and r=1 integration.
    spec=dict(model='qwen',method='baseline',seed=903,layers=2,r=1,frozen=True)
    model,tok=load(spec);model.eval()
    pool=rows(SIDE/'data/vt/tiny.jsonl');ids,_,_=encode(tok,pool[0]);ids=torch.tensor([ids],device='cuda')
    with torch.no_grad():
        baseline=model(ids,use_cache=False,logits_to_keep=1).logits.float()
        for m in model.modules():
            if hasattr(m,'lt_method'):m.lt_method='binomial';m.lt_r=1
        changed=model(ids,use_cache=False,logits_to_keep=1).logits.float()
    torch.testing.assert_close(baseline,changed,atol=0,rtol=0)
    del model;gc.collect();torch.cuda.empty_cache()
    for method,layers in [('baseline',2),('binomial',2),('binomial',4)]:
        spec=dict(model='qwen',method=method,seed=904,layers=layers,r=8)
        model,tok=load(spec);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        row=next(x for x in rows(SIDE/'data/vt/train.jsonl') if x['context_words']==512)
        times=[];torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss,_,_=batch_loss(model,tok,[row],'cuda')
            if not torch.isfinite(loss):raise FloatingPointError('Qwen benchmark loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Qwen benchmark gradients')
            opt.step();torch.cuda.synchronize()
            if i:times.append(time.perf_counter()-t)
        benchmarks.append(dict(model='qwen',method=method,layers=layers,microbatch=1,input_tokens=1536,
            seconds_per_microbatch=sum(times)/len(times),peak_bytes=torch.cuda.max_memory_allocated()))
        del model,opt;gc.collect();torch.cuda.empty_cache()
    atomic(ART/'benchmarks.json',benchmarks)
    atomic(ART/'gpu_preflight.json',dict(status='passed',source_sha256=source_hash(),seconds=time.time()-started,
        hardware=torch.cuda.get_device_name(),tiny_tests=len(results),benchmark_conditions=len(benchmarks),
        cache=False,job_id=os.environ['SLURM_JOB_ID']))

