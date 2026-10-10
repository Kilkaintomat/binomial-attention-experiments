"""Deterministic task data. Solvers generate labels and never enter model inputs."""
import hashlib
import json
import math
import os
import csv
from collections import deque
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.config import digest, write_json


def neighbors(v, size):
    r, c = divmod(v, size)
    return [rr*size+cc for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
            if 0 <= rr < size and 0 <= cc < size]


def tree_maze(size, rng, algorithm="dfs"):
    n = size*size
    a = np.eye(n, dtype=bool)
    if algorithm == "dfs":
        stack = [int(rng.integers(n))]
        seen = {stack[0]}
        while stack:
            choices = [w for w in neighbors(stack[-1], size) if w not in seen]
            if not choices: stack.pop(); continue
            w = int(rng.choice(choices)); v = stack[-1]
            a[v,w] = a[w,v] = True
            seen.add(w); stack.append(w)
    elif algorithm == "wilson":
        seen = {int(rng.integers(n))}
        while len(seen) < n:
            start = next(v for v in range(n) if v not in seen)
            path = [start]; indices = {start:0}
            while path[-1] not in seen:
                nxt = int(rng.choice(neighbors(path[-1], size)))
                if nxt in indices:
                    path = path[:indices[nxt]+1]
                    indices = {v:i for i,v in enumerate(path)}
                else: indices[nxt] = len(path); path.append(nxt)
            for v,w in zip(path[:-1],path[1:]):
                a[v,w] = a[w,v] = True; seen.add(v)
    else: raise ValueError("Unknown maze generator")
    return a


def topology_id(a, size):
    grid = np.arange(size*size).reshape(size,size)
    forms = []
    for mirror in (grid, np.fliplr(grid)):
        for k in range(4):
            order = np.rot90(mirror,k).ravel()
            forms.append(np.packbits(a[np.ix_(order,order)]).tobytes())
    return hashlib.sha256(min(forms)).hexdigest()


def path_label(a, start, goal):
    parents = {start:None}; queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == goal: break
        for w in np.flatnonzero(a[v]):
            w = int(w)
            if w not in parents: parents[w] = v; queue.append(w)
    if goal not in parents: raise ValueError("Disconnected graph")
    y = np.zeros(len(a),np.int64); v = goal
    while v is not None: y[v] = 1; v = parents[v]
    return y


def sudoku_graph(size):
    k = math.isqrt(size)
    r, c = np.divmod(np.arange(size*size),size)
    block = (r//k)*k+c//k
    return (r[:,None]==r)|(c[:,None]==c)|(block[:,None]==block)


def sudoku_solutions(grid, limit=2, rng=None):
    """Independent MRV backtracking oracle; stop after `limit` solutions."""
    size = math.isqrt(len(grid)); k = math.isqrt(size)
    board = np.asarray(grid,dtype=np.int64).copy()
    rows=[set() for _ in range(size)]; cols=[set() for _ in range(size)]; blocks=[set() for _ in range(size)]
    for v,d in enumerate(board):
        if d:
            r,c=divmod(v,size); b=r//k*k+c//k
            if d in rows[r] or d in cols[c] or d in blocks[b]: return []
            rows[r].add(int(d)); cols[c].add(int(d)); blocks[b].add(int(d))
    answers=[]; digits=set(range(1,size+1))
    def search():
        if len(answers)>=limit: return
        best=None; options=None
        for v in np.flatnonzero(board==0):
            r,c=divmod(int(v),size); b=r//k*k+c//k
            choices=digits-rows[r]-cols[c]-blocks[b]
            if not choices: return
            if options is None or len(choices)<len(options):
                best=(int(v),r,c,b); options=sorted(choices)
                if len(choices)==1: break
        if best is None: answers.append(board.copy()); return
        if rng is not None: rng.shuffle(options)
        v,r,c,b=best
        for d in options:
            board[v]=d; rows[r].add(d); cols[c].add(d); blocks[b].add(d)
            search()
            board[v]=0; rows[r].remove(d); cols[c].remove(d); blocks[b].remove(d)
            if len(answers)>=limit: return
    search()
    return answers


def solved_grid_id(grid):
    """Digit relabeling canonical form, not full Sudoku symmetry canonicalization."""
    mapping={}; normalized=[]
    for d in grid:
        mapping.setdefault(int(d),len(mapping)+1); normalized.append(mapping[int(d)])
    return digest(normalized)


def make_sudoku(size,rng,clue_fraction):
    solved=sudoku_solutions(np.zeros(size*size,dtype=int),limit=1,rng=rng)[0]
    puzzle=solved.copy()
    target=max(size,round(size*size*clue_fraction))
    for v in rng.permutation(size*size):
        if np.count_nonzero(puzzle)<=target: break
        previous=puzzle[v]; puzzle[v]=0
        if len(sudoku_solutions(puzzle,limit=2))!=1: puzzle[v]=previous
    return puzzle,solved


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def prepare(config):
    directory=config.data_dir(); manifest_path=directory/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for name,expected in manifest["files"].items():
            if file_hash(directory/name)!=expected: raise ValueError("Dataset integrity check failed")
        return manifest
    directory.mkdir(parents=True,exist_ok=True)
    if config.data_source=="rrn": return prepare_rrn(config,directory)
    rng=np.random.default_rng(config.data_seed)
    seen=set(); splits={}; stats={}; files={}
    base_split={}
    for split,count in (("train",config.train_count),("validation",config.val_count),("test",config.test_count)):
        records=[]; ids=[]; attempts=0
        while len(records)<count:
            attempts+=1
            if attempts>count*500: raise RuntimeError("Cannot obtain enough distinct data groups")
            if config.task=="maze":
                a=tree_maze(config.size,rng,config.generator)
                key=topology_id(a,config.size)
                if key in seen: continue
                s,g=map(int,rng.choice(config.size**2,2,replace=False))
                y=path_label(a,s,g)
                records.append((a,y,s,g)); seen.add(key)
            else:
                puzzle,y=make_sudoku(config.size,rng,config.clue_fraction)
                base=solved_grid_id(y)
                key=digest(puzzle.tolist())
                assigned=("train","validation","test")[int(base[:16],16)%3]
                if assigned!=split: continue
                if key in seen or (base in base_split and base_split[base]!=split): continue
                # No symmetry augmentation is performed. All repeated base grids stay in one split.
                base_split[base]=split
                records.append((puzzle,y)); seen.add(key)
            ids.append(key)
        arrays={"ids":np.array(ids)}
        if config.task=="maze":
            arrays.update(a=np.stack([r[0] for r in records]),y=np.stack([r[1] for r in records]),
                          start=np.array([r[2] for r in records]),goal=np.array([r[3] for r in records]))
            lengths=arrays["y"].sum(1)-1
            stats[split]={"path_length_mean":float(lengths.mean()),"path_length_max":int(lengths.max()),
                          "path_length_quantiles":np.quantile(lengths,[.25,.5,.75,.9,.95]).tolist()}
        else:
            arrays.update(puzzle=np.stack([r[0] for r in records]),y=np.stack([r[1] for r in records]))
            stats[split]={"clue_count_mean":float((arrays["puzzle"]!=0).sum(1).mean())}
        path=directory/f"{split}.npz"
        np.savez_compressed(path,**arrays)
        files[path.name]=file_hash(path); splits[split]=count
    manifest={"spec":config.data_spec(),"source":"independent generator v1",
              "splits":splits,"files":files,"statistics":stats,
              "no_augmentation":True,"unique_sudoku_solutions":config.task=="sudoku",
              "maze_split":"D4 canonical topology" if config.task=="maze" else None,
              "sudoku_grouping":"base solved grid, digit normalized; no full symmetry certificate" if config.task=="sudoku" else None,
              "labels_available_to_model":False}
    if config.task=="maze":
        with np.load(directory/"train.npz") as d:
            positives=int(d["y"].sum()); total=d["y"].size
        manifest["pos_weight"]=(total-positives)/positives
        manifest["long_path_boundary"]=stats["train"]["path_length_quantiles"][3]
    manifest["data_hash"]=digest(manifest)
    write_json(manifest_path,manifest)
    return manifest


def prepare_rrn(config,directory):
    external=Path(os.environ.get("TRANSPORT_RRN_DIR","/inputs/rrn"))
    files={}; splits={}; statistics={}; seen=set(); repeated=0
    for split,filename in (("train","train.csv"),("validation","valid.csv"),("test","test.csv")):
        path=external/filename
        if not path.exists(): raise FileNotFoundError("Official RRN CSV is absent; run the data download step")
        puzzles=[]; solutions=[]; ids=[]
        with path.open() as f:
            for row in csv.reader(f):
                q,a=row
                if len(q)!=81 or len(a)!=81: raise ValueError("Unexpected RRN row format")
                puzzle=np.fromiter((int(d) for d in q),dtype=np.int8,count=81)
                answer=np.fromiter((int(d) for d in a),dtype=np.int8,count=81)
                if np.any((puzzle!=0)&(puzzle!=answer)) or np.any(answer<1) or np.any(answer>9):
                    raise ValueError("RRN clue/solution mismatch")
                key=hashlib.sha256(q.encode()).hexdigest()
                if key in seen: repeated+=1
                puzzles.append(puzzle); solutions.append(answer); ids.append(key)
        seen.update(ids)
        out=directory/f"{split}.npz"
        np.savez_compressed(out,puzzle=np.stack(puzzles),y=np.stack(solutions),ids=np.array(ids))
        splits[split]=len(ids); files[out.name]=file_hash(out)
        statistics[split]={"source_sha256":file_hash(path),"clue_count_mean":float((np.stack(puzzles)!=0).sum(1).mean())}
    manifest={"spec":config.data_spec(),"source":"Recurrent Relational Networks official sudoku-hard data",
              "source_url":"https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/master/tasks/sudoku/data.py",
              "download_url":"https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1",
              "license":"No explicit dataset license found in the inspected source; external private research input, not redistributed",
              "official_splits_preserved":True,"splits":splits,"files":files,"statistics":statistics,
              "cross_split_exact_duplicate_rows":repeated,"full_symmetry_audit":False,"no_augmentation":True,
              "labels_available_to_model":False,"unique_solution":"as supplied; sampled solver audit saved separately"}
    if repeated: raise ValueError("Official splits contain exact duplicates; requires a recorded protocol amendment")
    manifest["data_hash"]=digest(manifest)
    write_json(directory/"manifest.json",manifest)
    return manifest


class TaskData:
    def __init__(self,config,split):
        self.config=config
        self.data=dict(np.load(config.data_dir()/f"{split}.npz",allow_pickle=False))
        self.ids=self.data["ids"].tolist()
        self.graph=sudoku_graph(config.size) if config.task=="sudoku" else None
        self.manifest=json.loads((config.data_dir()/"manifest.json").read_text())

    def __len__(self): return len(self.ids)

    def subset(self,count):
        self.data={k:v[:count] for k,v in self.data.items()}
        self.ids=self.ids[:count]
        return self

    def batch(self,indices,device):
        c=self.config; n=c.size*c.size
        rr,cc=np.divmod(np.arange(n),c.size)
        if c.task=="maze":
            a=self.data["a"][indices]; x=np.zeros((len(indices),n,8),np.float32)
            for v in range(n):
                r,col=divmod(v,c.size)
                for channel,(dr,dc) in enumerate(((-1,0),(1,0),(0,-1),(0,1))):
                    nr,nc=r+dr,col+dc
                    if 0<=nr<c.size and 0<=nc<c.size: x[:,v,channel]=a[:,v,nr*c.size+nc]
            x[:,:,4]=rr; x[:,:,5]=cc
            s=self.data["start"][indices]; g=self.data["goal"][indices]
            x[np.arange(len(indices)),s,6]=1; x[np.arange(len(indices)),g,7]=1
            inputs=np.zeros((len(indices),n),np.int64)
        else:
            inputs=self.data["puzzle"][indices]
            k=math.isqrt(c.size); block=rr//k*k+cc//k
            onehot=np.eye(c.size+1,dtype=np.float32)[inputs]
            positions=np.concatenate([np.eye(c.size)[rr],np.eye(c.size)[cc],np.eye(c.size)[block]],axis=1)
            x=np.concatenate([onehot,(inputs!=0)[:,:,None],np.broadcast_to(positions,(len(indices),n,3*c.size))],axis=2).astype(np.float32)
            a=np.broadcast_to(self.graph,(len(indices),n,n)).copy()
        if c.support=="dense": a=np.ones_like(a)
        return {"x":torch.as_tensor(x,device=device),"mask":torch.as_tensor(a,device=device),
                "y":torch.as_tensor(self.data["y"][indices],device=device,dtype=torch.long),
                "puzzle":torch.as_tensor(inputs,device=device,dtype=torch.long)}
