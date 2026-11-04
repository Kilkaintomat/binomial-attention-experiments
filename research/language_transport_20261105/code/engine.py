"""Answer-only training and cache-free, unconstrained greedy evaluation."""
import gc
import math
import os
import random
import re
import signal
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

from attention import install, change_r, root, operator_row, set_loops
from common import SIDE, ROOT, atomic, read, rows, append, freeze, digest, source_hash
from data import WordTokenizer

STOP = False


def stopping(*args):
    global STOP
    STOP = True


def load(spec, device='cuda'):
    torch.manual_seed(spec['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(spec['seed'])
    if spec['model']=='scratch':
        tok=WordTokenizer()
        config=Qwen3Config(vocab_size=len(tok),hidden_size=256,intermediate_size=1024,
            num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=4,head_dim=32,
            max_position_embeddings=4096,attention_dropout=0.,tie_word_embeddings=True,
            pad_token_id=0,bos_token_id=1,eos_token_id=2,use_cache=False)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).to(device)
        selected=[2,3]
    else:
        modeldir=ROOT/'models/Qwen3-1.7B'
        tok=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(str(modeldir),local_files_only=True,
            torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,
            attn_implementation='eager').to(device)
        for p in model.parameters():p.requires_grad_(False)
        if not spec.get('frozen'):
            model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,
                target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
        selected=[26,27] if spec['layers']==2 else [24,25,26,27]
    install(model,spec['method'],selected,spec.get('r',8),spec.get('loops',2))
    model.config.use_cache=False
    return model,tok


def encode(tok,row):
    if isinstance(tok,WordTokenizer):
        ids=[tok.bos_token_id]+tok.encode(row['prompt'])
        answer=tok.encode(row['answer'])+[tok.eos_token_id]
        diag={k:[x+1 for x in row['diagnostics'].get(k,[])] for k in ['source_word_positions','route_word_positions']}
    else:
        def render(text):
            return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
        prompt=render(row['prompt'])
        ids=tok.encode(prompt,add_special_tokens=False)
        # The equal-context VT slice is exact in each tokenizer, not merely words.
        # Prefix filler has no assignment syntax or answer information.
        if row['task']=='vt':
            target={128:512,256:768,512:1536,1024:3072}[row['context_words']]
            nfill=max(0,target-len(ids))
            for _ in range(5):
                prompt=render('quiet '*nfill+row['prompt'])
                ids=tok.encode(prompt,add_special_tokens=False)
                if len(ids)==target:break
                nfill+=target-len(ids)
                if nfill<0:raise ValueError('LM context target too short')
            if len(ids)!=target:raise ValueError('Unable to match exact LM input length')
        answer=tok.encode(row['answer'],add_special_tokens=False)+[tok.eos_token_id]
        base=prompt.index(row['prompt']); spans=list(re.finditer(r'\S+',row['prompt']))
        offsets=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        diag={}
        for k in ['source_word_positions','route_word_positions']:
            result=[]
            for w in row['diagnostics'].get(k,[]):
                span=spans[w];start,end=base+span.start(),base+span.end()
                ix=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
                if k=='route_word_positions':ix=ix[-1:]
                result.extend(ix)
            diag[k]=result
    return ids,answer,diag


def batch_loss(model,tok,batch,device):
    encoded=[encode(tok,x) for x in batch]
    lengths=[len(a)+len(b) for a,b,_ in encoded];n=max(lengths)
    x=torch.full((len(batch),n),tok.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(x);labels=torch.full_like(x,-100)
    for i,(a,b,_) in enumerate(encoded):
        x[i,:len(a)+len(b)]=torch.tensor(a+b,device=device)
        mask[i,:len(a)+len(b)]=1
        labels[i,len(a):len(a)+len(b)]=torch.tensor(b,device=device)
    if isinstance(tok,WordTokenizer):
        logits=model(input_ids=x,attention_mask=mask,use_cache=False).logits[:,:-1].float()
        target=labels[:,1:]
    else:
        assert len(batch)==1
        logits=model(input_ids=x,attention_mask=mask,use_cache=False,logits_to_keep=len(encoded[0][1])+1).logits[:,:-1].float()
        target=torch.tensor([encoded[0][1]],device=device)
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape(target.shape)
    loss=(losses.sum(-1)/(target!=-100).sum(-1)).mean()
    return loss,sum(lengths),sum(len(b) for _,b,_ in encoded)


def save_checkpoint(path,model,optimizer,step,stats):
    payload={'weights':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
             'optimizer':optimizer.state_dict(),'step':step,'stats':stats,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def restore(path,model,optimizer=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    named=dict(model.named_parameters())
    with torch.no_grad():
        for n,p in ck['weights'].items():named[n].copy_(p)
    if optimizer is not None:
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


@torch.no_grad()
def prediction(model,tok,row,device='cuda',capture=False):
    ids,answer,diag=encode(tok,row)
    mods=[(i,m.self_attn) for i,m in enumerate(root(model).model.layers) if hasattr(m.self_attn,'lt_capture')]
    for _,m in mods:m.lt_capture=False
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    input_ids=torch.tensor([ids],device=device)
    sync(device);start=time.perf_counter()
    out=model(input_ids=input_ids,use_cache=False,logits_to_keep=1).logits
    sync(device);prefill=time.perf_counter()-start
    generated=[]
    for _ in range(8):
        nxt=int(out[0,-1].argmax());generated.append(nxt)
        if nxt==tok.eos_token_id:break
        x=torch.tensor([ids+generated],device=device)
        out=model(input_ids=x,use_cache=False,logits_to_keep=1).logits
    sync(device);seconds=time.perf_counter()-start
    inference_peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
    diagnostic=[]
    if capture:
        # A separate forward keeps instrumentation out of inference timings.
        for _,m in mods:m.lt_capture=True
        model(input_ids=input_ids,use_cache=False,logits_to_keep=1)
        for i,m in mods:
            m.lt_capture=False
            a=m.lt_attention[0];q=len(ids)-1
            mass=operator_row(a,q,m.lt_method,m.lt_r)
            targets=diag['source_word_positions'];path=[q]+diag['route_word_positions']
            edges=[a[:,u,v].mean().item() for u,v in zip(path,path[1:])]
            diagnostic.append({'layer':i,'query_token_position':q,'source_token_positions':targets,
                'operator_mass_all_heads_mean':float(mass[:,targets].sum(-1).mean()) if targets else 0.,
                'operator_mass_each_head':mass[:,targets].sum(-1).tolist() if targets else [0.]*a.shape[0],
                'annotated_edge_mass_mean':sum(edges)/len(edges) if edges else None,
                'route_anchor_edges':len(path)-1,'tokenized_source_extra_edge':1 if targets and row['task']=='vt' else 0,
                'interpretation':'measured learned routing; nominal annotation is not an absorbing chain'})
            del m.lt_attention
    # Whole answer-string probability, excluding EOS. This teacher-forced
    # diagnostic is outside the timed free-generation interval.
    gold=answer[:-1]
    x=torch.tensor([ids+gold],device=device)
    logits=model(input_ids=x,use_cache=False,logits_to_keep=len(gold)+1).logits[0,:-1].float()
    logp=float(logits.log_softmax(-1).gather(-1,torch.tensor(gold,device=device)[:,None]).sum())
    text=tok.decode(generated,skip_special_tokens=True).strip()
    return {'id':row['id'],'task':row['task'],'hops':row['hops'],'distribution':row['distribution'],
            'context_words':row['context_words'],'sequence_length':row.get('sequence_length'),
            'input_tokens':len(ids),'chains':row['chains'],
            'answer':row['answer'],'prediction':text,'correct':text==row['answer'],
            'answer_sequence_log_probability_without_eos':logp,'answer_sequence_probability_without_eos':math.exp(logp),
            'prefill_seconds':prefill,'answer_seconds':seconds,'cache':False,
            'peak_allocated_bytes':inference_peak,
            'eos':bool(generated[-1]==tok.eos_token_id),'diagnostics':diagnostic,
            'pointer':{k:row['diagnostics'][k] for k in ['missing','self_loop','effective_transitions','nontrivial'] if k in row['diagnostics']}}


def validation(model,tok,pool,device='cuda',n=64):
    was=model.training;model.eval()
    rec=[prediction(model,tok,x,device) for x in pool[:n]]
    model.train(was)
    return {'accuracy':sum(x['correct'] for x in rec)/len(rec),'n':len(rec),
            'by_hops':{str(h):sum(x['correct'] for x in rec if x['hops']==h)/sum(x['hops']==h for x in rec)
                       for h in sorted({x['hops'] for x in rec})}}


def train(spec,deadline,device='cuda'):
    run=SIDE/'runs'/spec['id'];run.mkdir(parents=True,exist_ok=True)
    data_manifest=read(SIDE/'data/manifest.json')
    freeze(run/'manifest.json',dict(spec,source_sha256=source_hash(),data_sha256=digest(SIDE/'data/manifest.json')))
    if (run/'training.json').exists() and read(run/'training.json').get('status')=='completed':return True
    model,tok=load(spec,device);model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=.01)
    ckpath=run/'checkpoint_last.pt'
    stats={'trainable_parameters':sum(p.numel() for p in params),'total_parameters':sum(p.numel() for p in model.parameters()),
           'train_tokens':0,'supervised_tokens':0,'examples':0,'training_seconds':0.,'precision':'BF16 autocast' if device=='cuda' else 'FP32',
           'hardware':torch.cuda.get_device_name() if device=='cuda' else 'CPU','updates':0}
    step=0
    if ckpath.exists():
        ck=restore(ckpath,model,opt);step=ck['step'];stats=ck['stats']
    pool=rows(SIDE/'data'/spec['task']/(spec.get('train_split','train')+'.jsonl'))
    val=rows(SIDE/'data'/spec['task']/'validation.jsonl')
    by_hops={h:[r for r in pool if r['hops']==h] for h in sorted({x['hops'] for x in pool})}
    status='paused'
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<spec['steps'] and not STOP and time.monotonic()<deadline-120:
        rng=random.Random(spec['seed']*100000000+step)
        h=rng.choice(list(by_hops));batch=rng.choices(by_hops[h],k=spec['batch'])
        micro=spec['micro'];opt.zero_grad(set_to_none=True)
        warm=min(1.,(step+1)/max(1,spec['steps']//20))
        lr=spec['lr']*warm*(.1+.9*.5*(1+math.cos(math.pi*step/spec['steps'])))
        for group in opt.param_groups:group['lr']=lr
        sync(device);t=time.perf_counter();loss_sum=0.
        for k in range(0,len(batch),micro):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                loss,nt,ns=batch_loss(model,tok,batch[k:k+micro],device)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            (loss*(len(batch[k:k+micro])/len(batch))).backward()
            loss_sum+=float(loss.detach())*len(batch[k:k+micro])/len(batch)
            stats['train_tokens']+=nt;stats['supervised_tokens']+=ns
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
        opt.step();sync(device);elapsed=time.perf_counter()-t;step+=1
        stats['examples']+=len(batch);stats['training_seconds']+=elapsed;stats['updates']=step
        stats['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else None
        append(run/'training_log.jsonl',dict(step=step,loss=loss_sum,lr=lr,seconds=elapsed,
            cumulative_train_seconds=stats['training_seconds'],gradient_norm=float(norm),hops=h))
        if step%25==0:print(spec['id'],'STEP',step,'LOSS',loss_sum,flush=True)
        if step%100==0 or step==spec['steps']:
            save_checkpoint(ckpath,model,opt,step,stats)
            atomic(run/'training.json',dict(stats,status='running',step=step,target_steps=spec['steps']))
        if step in spec.get('validation_steps',[]) or step==spec['steps']:
            result=validation(model,tok,val,device,spec.get('validation_n',64))
            append(run/'validation.jsonl',dict(step=step,training_seconds=stats['training_seconds'],**result))
    if step==spec['steps']:
        status='completed'
        vp=run/'validation.jsonl'
        if not vp.exists() or not any(x['step']==step for x in rows(vp)):
            result=validation(model,tok,val,device,spec.get('validation_n',64))
            append(vp,dict(step=step,training_seconds=stats['training_seconds'],**result))
        if spec['phase'].startswith('pilot') and spec['task']=='vt' and spec['method']=='baseline':
            # A disjoint short-depth pilot-only set describes the half-accuracy
            # region. It never changes the fixed main test or its distribution.
            model.eval();probe=rows(SIDE/'data/vt/pilot.jsonl')
            records=[prediction(model,tok,x,device) for x in probe]
            groups={}
            for x in records:groups.setdefault((x['hops'],x['context_words']),[]).append(x)
            cells=[dict(hops=h,context_words=n,n=len(g),accuracy=sum(x['correct'] for x in g)/len(g)) for (h,n),g in sorted(groups.items())]
            atomic(run/'pilot_probe.json',dict(scope='disjoint pilot-only depths <=4',cells=cells,
                closest_to_half=min(cells,key=lambda x:abs(x['accuracy']-.5)),main_test_changed=False))
    save_checkpoint(ckpath,model,opt,step,stats)
    atomic(run/'training.json',dict(stats,status=status,step=step,target_steps=spec['steps']))
    del model,opt;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return status=='completed'


def evaluate(spec,deadline,device='cuda'):
    run=SIDE/'runs'/spec['id'];run.mkdir(parents=True,exist_ok=True)
    freeze(run/'manifest.json',dict(spec,source_sha256=source_hash(),data_sha256=digest(SIDE/'data/manifest.json')))
    model,tok=load(spec,device)
    checkpoint=run/'checkpoint_last.pt'
    if not spec.get('frozen'):restore(checkpoint,model)
    checkpoint_hash=digest(checkpoint) if checkpoint.exists() else read(SIDE/'artifacts/environment.json')['model']['revision']
    model.eval()
    test=rows(SIDE/'data'/spec['task']/'test.jsonl')
    conditions=[(spec['method'],spec['loops'] if spec['method']=='looped' else spec.get('r',8),'main',test)]
    if spec['method'] in ['binomial','looped']:
        # Fixed predefined 8 items per depth/distribution, same for every seed.
        selected=[];counts={}
        for row in test:
            key=(row['hops'],row['distribution'])
            counts[key]=counts.get(key,0)+1
            if counts[key]<=8:selected.append(row)
        if spec['method']=='binomial':conditions += [('binomial',r,'r_sweep',selected) for r in [1,2,4,8,16,32,64,80,96]]
        else:conditions += [('looped',n,'loop_sweep',selected) for n in [1,2,4,8,16]]
        if spec['task'].startswith('phop') and spec['method']=='binomial':
            conditions += [('power',None,'exact_p_diagnostic',selected)]
    outpath=run/'predictions.jsonl';records=rows(outpath) if outpath.exists() else []
    done={(x['slice'],x['inference_setting'],x['id']) for x in records}
    expected=sum(len(xs) for _,_,_,xs in conditions)
    if len(done)!=len(records):raise ValueError('Duplicate evaluation keys')
    for method,r,slice_name,examples in conditions:
        for row in examples:
            reval=row['hops'] if r is None else r
            key=(slice_name,reval,row['id'])
            if key in done:continue
            if STOP or time.monotonic()>deadline-120:
                atomic(run/'result.json',{'status':'paused','completed':len(records),'expected':expected});return False
            if spec['method']=='looped':set_loops(model,reval)
            else:change_r(model,reval,method)
            rec=prediction(model,tok,row,device,capture=slice_name=='r_sweep')
            rec.update(slice=slice_name,r_train=spec.get('r',8),r_eval=None if method=='looped' else reval,
                       loops_eval=reval if method=='looped' else 1,inference_setting=reval,method_eval=method,
                       seed=spec['seed'],checkpoint_sha256=checkpoint_hash)
            append(outpath,rec);records.append(rec);done.add(key)
            if len(records)%32==0:print(spec['id'],'EVAL',len(records),'/',expected,flush=True)
    main=[x for x in records if x['slice']=='main']
    atomic(run/'result.json',{'status':'completed','completed':len(records),'expected':expected,
        'main_accuracy':sum(x['correct'] for x in main)/len(main),'main_n':len(main),
        'checkpoint_sha256':digest(checkpoint) if checkpoint.exists() else 'frozen_pretrained_revision',
        'source_sha256':source_hash(),'cache':False})
    del model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return True
