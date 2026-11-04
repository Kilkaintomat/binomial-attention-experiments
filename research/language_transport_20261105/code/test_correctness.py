import copy
import math
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from attention import action,coefficients,install,operator_row
from data import phop_solver,phop_fast,example,variable_tracking,vt_solver,WordTokenizer


class FilterTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(21);torch.set_num_threads(2)
    def test_dense_reference(self):
        a=torch.randn(5,5,dtype=torch.float64).tril().masked_fill(torch.ones(5,5).triu(1).bool(),-torch.inf).softmax(-1)
        v=torch.randn(5,3,dtype=torch.float64)
        for r in [1,2,4,8,16,32,64,80,96]:
            expected=sum(float(math.comb(r,s))*torch.linalg.matrix_power(a,s) for s in range(1,r+1))/float(2**r-1)
            torch.testing.assert_close(action(a,v,'binomial',r),expected@v,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(action(a,v,'binomial',r),sum(c*torch.linalg.matrix_power(a,s)@v for s,c in enumerate(coefficients('binomial',r),1)),atol=1e-12,rtol=1e-12)
    def test_gradients(self):
        a=torch.randn(3,3,dtype=torch.double,requires_grad=True)
        v=torch.randn(3,2,dtype=torch.double,requires_grad=True)
        for method in ['binomial','geometric','power']:
            self.assertTrue(torch.autograd.gradcheck(lambda x,y:action(x.softmax(-1),y,method,4),(a,v)))
    def test_oracle_tail_and_rows(self):
        for ell in [1,2,4,8,16,32]:
            n=ell+1;a=torch.zeros(n,n,dtype=torch.float64);a[0,0]=1
            for i in range(1,n):a[i,i-1]=1
            for r in [1,2,4,8,16,32,64,80,96]:
                y=action(a,torch.eye(n,dtype=torch.float64),'binomial',r)
                expected=sum(math.comb(r,s) for s in range(ell,r+1))/(2**r-1)
                self.assertAlmostEqual(float(y[-1,0]),expected,places=13)
                torch.testing.assert_close(y.sum(-1),torch.ones(n,dtype=torch.float64),atol=2e-14,rtol=2e-14)
                torch.testing.assert_close(operator_row(a,n-1,'binomial',r),y[-1],atol=2e-14,rtol=2e-14)
    def test_geometric_mean(self):
        for r in [2,4,8,16,32,96]:
            c=coefficients('geometric',r)
            self.assertAlmostEqual(float(c@torch.arange(1,r+1,dtype=c.dtype)),r/(2*(1-2.**-r)),places=11)
    def test_dtype_stability(self):
        a=torch.rand(12,12).tril();a/=a.sum(-1,keepdim=True);v=torch.randn(12,5)
        for dtype in [torch.float32,torch.bfloat16]:
            for r in [1,8,96]:
                x=action(a.to(dtype),v.to(dtype),'binomial',r)
                self.assertTrue(torch.isfinite(x).all())


class DataTests(unittest.TestCase):
    def test_manual_pointer(self):
        self.assertEqual(phop_solver(list('ABAC'),1),('BOTTOM',[3,-1]))
        self.assertEqual(phop_solver(list('ABAB'),1),('A',[3,2]))
        self.assertEqual(phop_solver(list('ABAB'),2),('B',[3,2,1]))
        self.assertEqual(phop_solver(list('ABAB'),3),('BOTTOM',[3,2,1,-1]))
        self.assertEqual(phop_solver(list('ABBA'),2),('BOTTOM',[3,1,-1]))
        self.assertEqual(phop_solver(list('ABB'),32)[0],'B')
        self.assertEqual(phop_solver(list('ABB'),2)[1],[2,2,2])
    def test_independent_solvers(self):
        rng=random.Random(1)
        for _ in range(1000):
            seq=rng.choices('ABCD',k=rng.randrange(1,100));p=rng.randrange(1,40)
            self.assertEqual(phop_solver(seq,p)[0],phop_fast(seq,p))
    def test_vt_order_lengths_and_answers(self):
        tok=WordTokenizer()
        for depth in [1,4,8,16,32]:
            for chains in [2,4,8]:
                x=variable_tracking(random.Random(depth*100+chains),depth,512,chains)
                self.assertEqual(vt_solver(x['prompt']),x['answer'])
                self.assertEqual(len(tok.encode(x['prompt'])),512)
                self.assertEqual(len(x['diagnostics']['route_word_positions']),depth+1)
                self.assertEqual(x['prompt'].split()[x['diagnostics']['source_word_positions'][0]],x['answer'])
    def test_conditioning_and_split_independence(self):
        a=example('phop_conditional','test',5,8)
        b=example('phop_conditional','test',5,16)
        self.assertNotEqual(a['prompt'],b['prompt'])
        self.assertEqual(a['prompt'].count('|'),8)
        self.assertEqual(b['prompt'].count('|'),16)
        for task in ['vt','phop_conditional','phop_fixed16','phop_fixed32']:
            sets=[{example(task,s,i)['id'] for i in range(64)} for s in ['train','validation','test']]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


class IntegrationTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(13)
        c=Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
                      attention_dropout=0.,use_cache=False,pad_token_id=0)
        c._attn_implementation='eager'
        return Qwen3ForCausalLM(c).eval()
    def test_r1_matches_native_gqa_rope_padding(self):
        native=self.model();custom=copy.deepcopy(native);install(custom,'binomial',[0,1],1)
        ids=torch.tensor([[1,2,3,4,0,0],[1,5,6,7,8,9]])
        mask=ids.ne(0).long()
        with torch.no_grad():
            x=native(ids,attention_mask=mask,use_cache=False).logits
            y=custom(ids,attention_mask=mask,use_cache=False).logits
        torch.testing.assert_close(x[mask.bool()],y[mask.bool()],atol=3e-7,rtol=2e-5)
    def test_causality_and_padding_invariance(self):
        model=self.model();install(model,'binomial',[0,1],8)
        x=torch.tensor([[1,2,3,4,5,6]]);y=x.clone();y[:,4:]=9
        with torch.no_grad():
            a=model(x,use_cache=False).logits;b=model(y,use_cache=False).logits
            c=model(torch.cat([x,torch.zeros(1,2,dtype=torch.long)],1),attention_mask=torch.tensor([[1]*6+[0]*2]),use_cache=False).logits
        torch.testing.assert_close(a[:,:4],b[:,:4],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a,c[:,:6],atol=1e-7,rtol=1e-6)
    def test_qk_gradients_and_no_extra_parameters(self):
        model=self.model();count=sum(p.numel() for p in model.parameters());install(model,'binomial',[0,1],8)
        self.assertEqual(count,sum(p.numel() for p in model.parameters()))
        model(torch.tensor([[1,2,3,4]]),use_cache=False).logits[0,-1,5].backward()
        for layer in model.model.layers:
            for p in [layer.self_attn.q_proj.weight,layer.self_attn.k_proj.weight]:
                self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
    def test_loop_shared_weights_and_real_recomputation(self):
        model=self.model();count=sum(p.numel() for p in model.parameters());install(model,'looped',[],8,2)
        self.assertIs(model.model.layers[0].mlp,model.model.layers[2].mlp)
        self.assertIs(model.model.layers[0].self_attn.q_proj,model.model.layers[2].self_attn.q_proj)
        self.assertEqual(count,sum(p.numel() for p in model.parameters()))
        calls=[];h=model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda *args:calls.append(1))
        model(torch.tensor([[1,2,3,4]]),use_cache=False);h.remove();self.assertEqual(len(calls),2)


class EngineTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(55);tok=WordTokenizer()
        c=Qwen3Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=2048,
            attention_dropout=0.,use_cache=False,pad_token_id=0)
        c._attn_implementation='eager';m=Qwen3ForCausalLM(c)
        install(m,'binomial',[0,1],4)
        return m,tok
    def test_answer_only_loss_padding_matches_independent_examples(self):
        from engine import batch_loss
        model,tok=self.model();model.eval()
        a=example('vt','tiny',0,1,128,2);b=example('vt','tiny',1,1,256,2)
        batched,_,_=batch_loss(model,tok,[a,b],'cpu')
        one,_,_=batch_loss(model,tok,[a],'cpu');two,_,_=batch_loss(model,tok,[b],'cpu')
        torch.testing.assert_close(batched,(one+two)/2,atol=1e-6,rtol=1e-6)
    def test_pause_resume_train_evaluate_and_repeated_evaluation(self):
        import engine
        from common import atomic,read,rows
        from data import write_rows
        from plan import spec
        with tempfile.TemporaryDirectory() as d:
            side=Path(d);(side/'data/vt').mkdir(parents=True)
            pool=[example('vt','tiny',i,1,128,2) for i in range(2)]
            for split in ['train','validation','test']:write_rows(side/'data/vt'/(split+'.jsonl'),pool)
            atomic(side/'data/manifest.json',{'test_fixture':True})
            s=spec('scratch','vt','binomial',11,phase='smoke',steps=2)
            s.update(batch=2,micro=1,validation_n=2,validation_steps=[],r=4)
            with patch.object(engine,'SIDE',side),patch.object(engine,'load',side_effect=lambda *a,**k:self.model()):
                self.assertFalse(engine.train(s,time.monotonic(),'cpu'))
                self.assertEqual(read(side/'runs'/s['id']/'training.json')['step'],0)
                self.assertTrue(engine.train(s,time.monotonic()+300,'cpu'))
                self.assertEqual(read(side/'runs'/s['id']/'training.json')['step'],2)
                self.assertTrue(engine.evaluate(s,time.monotonic()+300,'cpu'))
                result=read(side/'runs'/s['id']/'result.json')
                self.assertEqual(result['completed'],20)
                before=rows(side/'runs'/s['id']/'predictions.jsonl')
                self.assertTrue(engine.evaluate(s,time.monotonic()+300,'cpu'))
                self.assertEqual(before,rows(side/'runs'/s['id']/'predictions.jsonl'))


if __name__=='__main__':unittest.main(verbosity=2)
