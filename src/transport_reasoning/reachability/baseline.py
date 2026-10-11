"""Classical reachability controls and a small degree-only logistic model."""
import time
import numpy as np
from .config import allocation,read,root,write_json,source_hash
from .data import Dataset
from .metrics import binary_metrics


def features(data):
    xs=[];ys=[];edges=[];weak=[]
    for meta,a,count,q,prefix in data.groups:
        pairs=np.asarray(a['pairs'][:count]);s=pairs[:,:,0];t=pairs[:,:,1];row=np.arange(count)[:,None]
        indegree=np.asarray(a['indegree'][:count]);outdegree=np.asarray(a['outdegree'][:count]);m=outdegree.sum(1)
        x=np.stack([indegree[row,s],outdegree[row,s],indegree[row,t],outdegree[row,t],np.full_like(s,meta['n']),np.broadcast_to(m[:,None],s.shape)],axis=-1).reshape(-1,6)
        xs.append(np.log1p(x.astype(float)));ys.append((a['distance'][:count]>=0).reshape(-1));edges.append(a['B'][:count][row,s,t].reshape(-1));weak.append(a['weak_same'][:count].reshape(-1))
    return np.concatenate(xs),np.concatenate(ys).astype(np.int8),np.concatenate(edges),np.concatenate(weak)


def fit_logistic(x,y,regularization=.001):
    mean=x.mean(0);std=x.std(0);std=np.where(std>1e-12,std,1.)
    X=np.c_[np.ones(len(x)),(x-mean)/std];w=np.zeros(X.shape[1]);penalty=np.eye(len(w))*regularization;penalty[0,0]=0
    for iteration in range(50):
        z=np.clip(X@w,-40,40);p=1/(1+np.exp(-z));gradient=X.T@(p-y)/len(y)+penalty@w
        hessian=(X.T*(p*(1-p)))@X/len(y)+penalty+np.eye(len(w))*1e-9
        delta=np.linalg.solve(hessian,gradient);w-=delta
        if np.max(np.abs(delta))<1e-7:break
    return {'weights':w.tolist(),'mean':mean.tolist(),'std':std.tolist(),'iterations':iteration+1,'l2':regularization}


def baselines():
    allocation();target=root()/'baselines';target.mkdir(exist_ok=True)
    specification={'features':['log1p_in_s','log1p_out_s','log1p_in_t','log1p_out_t','log1p_n','log1p_m'],
                   'normalization':'train-only mean/std','optimizer':'Newton, up to 50 iterations, tolerance1e-7',
                   'l2':.001,'selection':'no hyperparameter selection; fixed before baseline test evaluation',
                   'hard_predicate_bce':'undefined; no invented probability calibration'}
    p=target/'protocol.json'
    if p.exists() and read(p)!=specification:raise RuntimeError('Baseline design changed')
    write_json(p,specification);rows=[]
    for family in ['er','dag']:
        started=time.perf_counter();x,y,_,_=features(Dataset(family,'train'));model=fit_logistic(x,y)
        model.update(family=family,source_hash=source_hash(),fit_seconds=time.perf_counter()-started);write_json(target/(family+'_logistic.json'),model)
        for split,n in [('validation',None),('test_ID',None),('test_near',None),('test_far',32),('test_far',64),('test_far',128)]:
            data=Dataset(family,split,n=n);x,y,edge,weak=features(data)
            logits=np.c_[np.ones(len(x)),(x-np.array(model['mean']))/np.array(model['std'])]@np.array(model['weights'])
            for method,z in [('degree_logistic',logits),('always_negative',-np.ones(len(y))),('always_positive',np.ones(len(y))),
                             ('direct_edge',np.where(edge,1.,-1.)),('weak_connectivity',np.where(weak,1.,-1.)),('directed_BFS_oracle',np.where(y,1.,-1.))]:
                metric=binary_metrics(y,z)
                if method!='degree_logistic':metric['bce']=None
                if method=='directed_BFS_oracle':assert metric['accuracy']==1.
                rows.append({'family':family,'split':split,'n':n,'method':method,**metric})
    write_json(target/'results.json',{'source_hash':source_hash(),'rows':rows,'BFS_implementation':'labels were computed by BFS and independently audited; no fresh BFS speed claim'})
    return {'status':'completed','rows':len(rows),'new_DL_trainings':0}
