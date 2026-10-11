import hashlib
import json
import numpy as np
from transport_reasoning.data import TaskData as LegacyTaskData, sudoku_graph


def allowed_mask(support, size=9):
    if support == 'graph': return sudoku_graph(size)
    if support != 'grid4': raise ValueError('Unknown communication support')
    row,col = np.divmod(np.arange(size*size),size)
    return np.abs(row[:,None]-row[None,:]) + np.abs(col[:,None]-col[None,:]) <= 1


def mask_hash(mask):
    a=np.asarray(mask,dtype=np.bool_)
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(json.dumps(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def diameter(mask):
    n=len(mask); distance=np.where(mask,1,n+1); np.fill_diagonal(distance,0)
    for k in range(n): distance=np.minimum(distance,distance[:,k,None]+distance[None,k,:])
    if (distance>n).any(): raise ValueError('Disconnected mask')
    return int(distance.max())


class TaskData(LegacyTaskData):
    """Same verified puzzle/target cache and 38 features; only communication changes."""
    def __init__(self, config, split):
        config.validate(); super().__init__(config,split)
        self.graph=allowed_mask(config.support,config.size)
        self.mask_sha256=mask_hash(self.graph)
