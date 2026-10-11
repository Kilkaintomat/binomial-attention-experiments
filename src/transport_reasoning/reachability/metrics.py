"""Binary metrics; undefined single-class metrics remain missing."""
import numpy as np


def binary_metrics(labels,logits):
    y=np.asarray(labels,dtype=np.int8);z=np.asarray(logits,dtype=np.float64)
    if len(y)==0:return {'count':0,'balanced_accuracy':None,'recall':None,'auroc':None,'bce':None}
    if not np.isfinite(z).all():raise FloatingPointError('Non-finite prediction')
    pred=z>=0;positive=y==1;negative=~positive
    tp=int((pred&positive).sum());tn=int((~pred&negative).sum());fp=int((pred&negative).sum());fn=int((~pred&positive).sum())
    p=tp+fn;n=tn+fp;recall=tp/p if p else None;specificity=tn/n if n else None
    auc=ap=None
    if p and n:
        order=np.argsort(z,kind='stable');zs=z[order];ys=y[order]
        starts=np.r_[0,np.flatnonzero(np.diff(zs))+1];ends=np.r_[starts[1:],len(zs)]
        ranks=np.empty(len(zs),np.float64)
        for lo,hi in zip(starts,ends):ranks[lo:hi]=(lo+1+hi)/2
        auc=float((ranks[ys==1].sum()-p*(p+1)/2)/(p*n))
        order=np.argsort(-z,kind='stable');zs=z[order];ys=y[order]
        end=np.r_[np.flatnonzero(np.diff(zs)),len(zs)-1];cum=np.cumsum(ys)[end]
        ap=float(np.sum(np.diff(np.r_[0,cum])/p*cum/(end+1)))
    return {'count':len(y),'positive_count':p,'negative_count':n,'positive_fraction':p/len(y),
            'balanced_accuracy':(recall+specificity)/2 if p and n else None,'accuracy':(tp+tn)/len(y),
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':recall,'specificity':specificity,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'auroc':auc,'average_precision':ap,
            'bce':float(np.mean(np.logaddexp(0,z)-y*z)),'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def slices(labels,logits,metadata):
    y=np.asarray(labels);z=np.asarray(logits);out=[]
    for name,low,high in [('1',1,1),('2',2,2),('3-4',3,4),('5-8',5,8),('9-16',9,16),('17-32',17,32),('>32',33,10000)]:
        ids=np.array([low<=r['distance']<=high for r in metadata]);row=binary_metrics(y[ids],z[ids]);row.update(slice='positive_distance_'+name,metric='recall');out.append(row)
    for same in [False,True]:
        for reverse in [False,True]:
            ids=np.array([r['distance']<0 and r['weak_same']==same and r['reverse']==reverse for r in metadata])
            row=binary_metrics(y[ids],z[ids]);row.update(slice='negative_weak_'+str(same)+'_reverse_'+str(reverse),metric='specificity');out.append(row)
    return out


def clustered_ba_interval(labels,logits,graph_ids,repeats=2000):
    y=np.asarray(labels);pred=np.asarray(logits)>=0;unique,inverse=np.unique(graph_ids,return_inverse=True)
    bins=np.zeros((len(unique),4),np.int64)
    category=np.where(y==1,np.where(pred,0,3),np.where(pred,2,1))
    np.add.at(bins,(inverse,category),1);rng=np.random.default_rng(20261011);values=[]
    for _ in range(repeats):
        tp,tn,fp,fn=bins[rng.integers(len(bins),size=len(bins))].sum(0)
        if tp+fn and tn+fp:values.append(.5*(tp/(tp+fn)+tn/(tn+fp)))
    return {'unit':'graph/template','clusters':len(unique),'resamples':repeats,
            'ba_lower95':float(np.quantile(values,.025)) if values else None,
            'ba_upper95':float(np.quantile(values,.975)) if values else None}
