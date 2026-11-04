"""Two task families, independent splits and solvers. No route enters inputs."""
import hashlib
import json
import random
import re
from pathlib import Path

VALUES = ['red','blue','green','gold','pink','gray','cyan','lime',
          'black','white','brown','purple','orange','silver','violet','teal']
NAMES = ['v%03d' % i for i in range(256)]
ALPHABET = ['A','B','C','D']
SPLIT_SEEDS = {'train': 710001, 'validation': 720001, 'test': 730001,
               'pilot': 740001, 'tiny': 750001}
TASKS = ['vt','phop_conditional','phop_fixed16','phop_fixed32']


def phop_solver(sequence, p):
    """Literal 1-based Definition A.1, with find(0)=0 and undefined v_0.
    find_1(v,i) = max({0} union {j: 2 <= j <= i, v[j-1] = v[i]}).
    In zero-based Python a match at k<i maps to k+1 (the following token).
    """
    i, path = len(sequence)-1, [len(sequence)-1]
    for _ in range(p):
        if i < 0:
            path.append(-1); continue
        found = [k+1 for k in range(i) if sequence[k] == sequence[i]]
        i = max(found) if found else -1
        path.append(i)
    return ('BOTTOM' if i < 0 else sequence[i]), path


def phop_fast(sequence, p):
    # Independent pointer construction by last occurrence, then integer jumping.
    last, pointers = {}, []
    for i, x in enumerate(sequence):
        pointers.append(last[x]+1 if x in last else -1)
        last[x] = i
    i = len(sequence)-1
    for _ in range(p):
        i = -1 if i < 0 else pointers[i]
    return 'BOTTOM' if i < 0 else sequence[i]


def vt_solver(text):
    env = {}
    for name, source in re.findall(r'(v\d{3}) (?:=|copies) (v\d{3}|[a-z]+) ;', text):
        if name in env:
            raise ValueError('Redefinition')
        env[name] = env[source] if source.startswith('v') and source not in VALUES else source
    query = re.search(r'Question value (v\d{3}) \?', text).group(1)
    return env[query]


def variable_tracking(rng, depth, length=512, chains=4):
    # Only target-chain depth varies. Independent distractor depths are 1..4.
    depths = [depth] + [rng.randint(1,4) for _ in range(chains-1)]
    names = iter(rng.sample(NAMES, sum(d+1 for d in depths)))
    values = rng.sample(VALUES, chains)
    lists, target_names = [], None
    for c, d in enumerate(depths):
        ns = [next(names) for _ in range(d+1)]
        if c == 0: target_names = ns
        op = rng.choice(['=', 'copies'])
        rows = [(ns[0]+' = '+values[c]+' ;', c, 0)]
        rows += [(ns[j]+' '+op+' '+ns[j-1]+' ;', c, j) for j in range(1,d+1)]
        lists.append(rows)
    records = []
    while any(lists):
        c = rng.choice([i for i,l in enumerate(lists) if l])
        records.append(lists[c].pop(0))
    header = 'Track variable assignments . Return only the value .'
    query = 'Question value '+target_names[-1]+' ? Answer :'
    used = len(header.split()) + len(query.split()) + sum(len(t.split()) for t,_,_ in records)
    if used > length: raise ValueError('Context too short for records')
    # Random filler placement before, between, and after records, independent of target.
    slots = [0]*(len(records)+1)
    for _ in range(length-used): slots[rng.randrange(len(slots))] += 1
    words = header.split(); route = {}; source_span = None
    for j, (text,c,k) in enumerate(records):
        words.extend(rng.choices(['quiet','cloud','stone','river'],k=slots[j]))
        start = len(words)
        if c == 0:
            route[k] = start + len(text.split()) - 1
            if k == 0: source_span = start+2
        words.extend(text.split())
    words.extend(rng.choices(['quiet','cloud','stone','river'],k=slots[-1]))
    words.extend(query.split())
    prompt = ' '.join(words)
    assert len(words) == length
    answer = values[0]
    assert vt_solver(prompt) == answer
    return dict(prompt=prompt, answer=answer, hops=depth, context_words=length,
                chains=chains, distribution='vt_adaptation',
                diagnostics=dict(source_word_positions=[source_span],
                    route_word_positions=[route[k] for k in range(depth,-1,-1)],
                    nominal_semantic_hops=depth, lexical_query_extra_hop=1))


def induction(rng, p, n=256, distribution='iid', conditional=True):
    seq = []
    for _ in range(n):
        choices = [x for x in ALPHABET if not seq or x != seq[-1]] if distribution == 'no_adjacent' else ALPHABET
        seq.append(rng.choice(choices))
    answer, path = phop_solver(seq, p)
    assert answer == phop_fast(seq, p)
    # Unary p uses a token observed at all training depths, with no unseen p ID.
    prefix = 'Follow previous equal then next . Return BOTTOM if absent . '
    if conditional: prefix += 'Steps : '+' '.join(['|']*p)+' . '
    else: prefix += 'Fixed task . '
    prompt = prefix+'Sequence : '+' '.join(seq)+' . Answer :'
    start = len((prefix+'Sequence :').split())
    changes = sum(a >= 0 and b >= 0 and a != b for a,b in zip(path,path[1:]))
    return dict(prompt=prompt,answer=answer,hops=p,context_words=len(prompt.split()),
                sequence_length=n,chains=0,distribution=distribution,
                diagnostics=dict(pointer_path=path,missing=path[-1]<0,
                    self_loop=any(a==b and a>=0 for a,b in zip(path,path[1:])),
                    effective_transitions=changes,nontrivial=changes==p,
                    source_word_positions=[] if path[-1]<0 else [start+path[-1]],
                    route_word_positions=[start+x for x in path if x>=0]))


def example(task, split, index, hops=None, length=None, chains=4, distribution=None):
    seed = SPLIT_SEEDS[split] + TASKS.index(task)*10000000 + index*1009
    rng = random.Random(seed)
    if task == 'vt':
        row = variable_tracking(rng, hops or (index%4+1), length or rng.choice([256,512]), chains)
    else:
        p = int(task.removeprefix('phop_fixed')) if task.startswith('phop_fixed') else (hops or index%4+1)
        row = induction(rng,p,length or 256,distribution or ('iid' if task.startswith('phop_fixed') or (index//4)%2==0 else 'no_adjacent'),
                        conditional=not task.startswith('phop_fixed'))
    row['id'] = hashlib.sha256((task+'\0'+row['prompt']).encode()).hexdigest()
    row['task'], row['split'] = task, split
    return row


class WordTokenizer:
    """Known finite alphabet; test uses new combinations, not unseen name tokens."""
    def __init__(self):
        literal = ('Track variable assignments . Return only the value Question ? Answer : = copies ; '
                   'quiet cloud stone river Follow previous equal then next BOTTOM if absent Steps | Sequence Fixed task').split()
        self.words = ['<pad>','<bos>','<eos>'] + sorted(set(literal+NAMES+VALUES+ALPHABET))
        self.ids = {x:i for i,x in enumerate(self.words)}
        self.pad_token_id,self.bos_token_id,self.eos_token_id=0,1,2
    def encode(self, text):
        return [self.ids[x] for x in text.split()]
    def decode(self, ids, **kwargs):
        return ' '.join(self.words[i] for i in ids if i>2)
    def __len__(self): return len(self.words)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')


def make_data(folder, train_n=16384):
    folder = Path(folder)
    if (folder/'manifest.json').exists(): return json.loads((folder/'manifest.json').read_text())
    manifest = {'split_seeds':SPLIT_SEEDS, 'files':{}, 'test_is_not_used_for_selection':True}
    tokenizer = WordTokenizer()
    for task in TASKS:
        seen = set(); train_tokens = set()
        for split,count in [('train',train_n),('validation',256),('pilot',128),('tiny',8),('test',0)]:
            if split != 'test':
                rows = [example(task,split,i, hops=1 if split=='tiny' else None,
                                length=128 if split=='tiny' else None, chains=2 if split=='tiny' else 4) for i in range(count)]
            elif task == 'vt':
                rows=[]
                for h in [1,2,3,4,8,16,32]:
                    for n in [512,1024]:
                        rows += [example(task,split,len(rows)+i,h,n) for i in range(64)]
                for h in [4,16]:
                    for c in [2,8]:
                        rows += [example(task,split,len(rows)+i,h,1024,c) for i in range(32)]
            elif task == 'phop_conditional':
                rows=[]
                for h in [1,2,3,4,8,16,32]:
                    for dist in ['iid','no_adjacent']:
                        rows += [example(task,split,len(rows)+i,h,256,distribution=dist) for i in range(64)]
                for h in [4,16]:
                    for dist in ['iid','no_adjacent']:
                        rows += [example(task,split,len(rows)+i,h,512,distribution=dist) for i in range(32)]
            else: rows=[example(task,split,i) for i in range(512)]
            for row in rows:
                if row['id'] in seen: raise ValueError('Duplicate across or within splits')
                seen.add(row['id'])
                ids=tokenizer.encode(row['prompt'])
                if split=='train':train_tokens.update(ids)
                elif not set(ids).issubset(train_tokens):raise ValueError('Untrained scratch input symbol')
            path=folder/task/(split+'.jsonl');write_rows(path,rows)
            summary={'n':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'answers':{a:sum(x['answer']==a for x in rows) for a in sorted({x['answer'] for x in rows})}}
            if task!='vt':
                summary.update({k:sum(bool(x['diagnostics'][k]) for x in rows) for k in ['missing','self_loop','nontrivial']})
            manifest['files'][str(path.relative_to(folder))]=summary
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest
