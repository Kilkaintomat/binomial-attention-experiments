"""Independent legality validators and prediction metrics, with no repair."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F


def maze_valid(a,chosen,start,goal):
    chosen=np.asarray(chosen,dtype=bool)
    if not chosen[start] or not chosen[goal]: return False
    vertices=np.flatnonzero(chosen)
    induced=a[np.ix_(vertices,vertices)].copy(); np.fill_diagonal(induced,False)
    degrees=induced.sum(1)
    expected=np.where((vertices==start)|(vertices==goal),1,2)
    if not np.array_equal(degrees,expected): return False
    reached={start}; stack=[start]
    while stack:
        v=stack.pop()
        for w in np.flatnonzero(a[v]&chosen):
            w=int(w)
            if w not in reached: reached.add(w); stack.append(w)
    return len(reached)==len(vertices)


def sudoku_valid(pred,puzzle):
    size=math.isqrt(len(pred)); k=math.isqrt(size)
    if np.any(np.asarray(pred)[puzzle!=0]!=puzzle[puzzle!=0]): return False
    grid=np.asarray(pred).reshape(size,size); target=np.arange(1,size+1)
    if not all(np.array_equal(np.sort(v),target) for v in [*grid,*grid.T]): return False
    return all(np.array_equal(np.sort(grid[r:r+k,c:c+k].ravel()),target)
               for r in range(0,size,k) for c in range(0,size,k))


def loss_for(logits,batch,config,pos_weight=1.):
    if config.task=="maze":
        losses=F.binary_cross_entropy_with_logits(logits.squeeze(-1),batch["y"].float(),
                    pos_weight=torch.as_tensor(pos_weight,device=logits.device),reduction="none")
        return losses.mean(1).mean()
    losses=F.cross_entropy(logits.transpose(1,2),batch["y"]-1,reduction="none")
    blank=batch["puzzle"]==0
    if not bool(blank.any(1).all()): raise ValueError("A Sudoku has no blank cells")
    return ((losses*blank).sum(1)/blank.sum(1)).mean()


def sync(device):
    if torch.device(device).type=="cuda": torch.cuda.synchronize()


@torch.no_grad()
def evaluate_model(model,data,config,device,save_predictions=None):
    model.eval(); solved=[]; exact=[]; predictions=[]; loss_sum=0.; tp=fp=fn=correct=blank_count=0
    breaks=branches=conflicts=0; times=[]; lengths=[]; buckets={}
    sync(device); started=time.perf_counter()
    for begin in range(0,len(data),config.batch_size):
        ids=np.arange(begin,min(begin+config.batch_size,len(data)))
        batch=data.batch(ids,device)
        sync(device); tick=time.perf_counter(); logits=model(batch); sync(device)
        times.append((time.perf_counter()-tick,len(ids)))
        loss_sum+=loss_for(logits,batch,config,data.manifest.get("pos_weight",1)).item()*len(ids)
        y=batch["y"].cpu().numpy()
        if config.task=="maze":
            pred=(logits.squeeze(-1)>=0).cpu().numpy()
            for j,index in enumerate(ids):
                a=data.data["a"][index]; s=int(data.data["start"][index]); g=int(data.data["goal"][index])
                valid=maze_valid(a,pred[j],s,g); same=bool(np.array_equal(pred[j],y[j]))
                if valid!=same: raise AssertionError("Independent maze validator disagrees with tree oracle")
                solved.append(valid); exact.append(same)
                length=int(y[j].sum()-1); lengths.append(length)
                bucket="long" if length>data.manifest["long_path_boundary"] else "short"
                buckets.setdefault(bucket,[]).append(valid)
                induced=a[np.ix_(pred[j],pred[j])].copy(); np.fill_diagonal(induced,False)
                branches+=bool((induced.sum(1)>2).any())
                breaks+=not valid and not bool((induced.sum(1)>2).any())
            tp+=int((pred&(y==1)).sum()); fp+=int((pred&(y==0)).sum()); fn+=int((~pred&(y==1)).sum())
        else:
            puzzle=batch["puzzle"].cpu().numpy()
            pred=logits.argmax(-1).cpu().numpy()+1
            pred=np.where(puzzle!=0,puzzle,pred)
            blank=puzzle==0
            correct+=int(((pred==y)&blank).sum()); blank_count+=int(blank.sum())
            for p,q,target in zip(pred,puzzle,y):
                valid=sudoku_valid(p,q); solved.append(valid); exact.append(bool(np.array_equal(p,target)))
                label=f"clues_{int(np.count_nonzero(q))//10*10}s"
                buckets.setdefault(label,[]).append(valid)
                grid=p.reshape(config.size,config.size)
                conflicts+=sum(config.size-len(set(row)) for row in [*grid,*grid.T])
        predictions.append(pred)
    sync(device)
    result={"loss":loss_sum/len(data),"solved_rate":float(np.mean(solved)),"exact_match":float(np.mean(exact)),
            "count":len(data),"evaluation_seconds":time.perf_counter()-started,
            "inference_seconds_per_example":sum(t for t,n in times)/len(data),
            "inference_peak_memory_bytes":torch.cuda.max_memory_allocated() if torch.device(device).type=="cuda" else None,
            "buckets":{k:{"count":len(v),"solved_rate":float(np.mean(v))} for k,v in buckets.items()}}
    if config.task=="maze": result.update(vertex_f1=2*tp/max(2*tp+fp+fn,1),vertex_iou=tp/max(tp+fp+fn,1),
                                         broken_fraction=breaks/len(data),branch_fraction=branches/len(data))
    else: result.update(blank_accuracy=correct/max(blank_count,1),row_column_conflicts_mean=conflicts/len(data))
    if save_predictions is not None:
        np.savez_compressed(save_predictions,ids=np.array(data.ids),predictions=np.concatenate(predictions),
                            solved=np.array(solved),exact=np.array(exact),path_lengths=np.array(lengths))
    return result
