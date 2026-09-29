"""Sharded, CPU-resident state for the unmodified upstream Muon/AdamW update rules.

Model parameters and averaged gradients stay replicated on every rank. Whole Muon matrices are assigned to one
rank; AdamW's element-wise state is split into flat chunks. The upstream optimizer classes (moe/optim.py of
Mapika/decider) perform every update with FP32 master weights; the FP32 state rests on the CPU between
updates, and the updated parameters are then synchronized exactly across ranks.
"""
import importlib.util
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

UPSTREAM = Path(os.environ.get('WEBJEV_UPSTREAM', Path(__file__).resolve().parents[1] / 'third_party' / 'decider'))
spec = importlib.util.spec_from_file_location('upstream_moe_optim', UPSTREAM/'moe/optim.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class ShardedRecipeOptim:
    def __init__(self, named_parameters, lr=1e-5, adam_chunk_elements=16*1024*1024):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.named = list(named_parameters)
        self.lr = lr
        self.buckets = []
        self.optimizers = []
        groups = {}
        for name, p in self.named:
            kind = 'muon' if recipe.is_muon_param(name, p) else 'adamw'
            groups.setdefault((kind, p.dtype), []).append((name, p))
        for (kind, dtype), params in groups.items():
            count = sum(p.numel() for _, p in params)
            size = math.ceil(count/self.world)*self.world if kind == 'adamw' else count
            flat = torch.zeros(size, dtype=dtype, device=params[0][1].device)
            entries = []
            offset = 0
            with torch.no_grad():
                for name, p in params:
                    shape = tuple(p.shape)
                    view = flat[offset:offset+p.numel()].view(shape)
                    view.copy_(p)
                    p.data = view
                    entries.append(dict(name=name, param=p, start=offset, end=offset+p.numel(), shape=shape))
                    offset += p.numel()
            b = dict(key=f'{kind}:{dtype}', kind=kind, dtype=str(dtype), flat=flat,
                     size=size, count=count, entries=entries, grad=None)
            self.buckets.append(b)
            if kind == 'muon':
                boundaries = [0] + [e['end'] for e in entries]
                cuts = [0]
                for rank in range(1, self.world):
                    target=count*rank/self.world
                    cuts.append(min(range(cuts[-1],len(boundaries)),key=lambda j:abs(boundaries[j]-target)))
                cuts.append(len(entries))
                b['ranges']=[(boundaries[cuts[r]],boundaries[cuts[r+1]]) for r in range(self.world)]
                for e in entries[cuts[self.rank]:cuts[self.rank+1]]:
                    opt = recipe.MasterMuon([e['param']],lr=lr)
                    self.optimizers.append(dict(key=b['key']+':'+e['name'], kind=kind,
                                                bucket=b, param=e['param'], opt=opt,
                                                start=e['start'],end=e['end'],name=e['name']))
            else:
                width=size//self.world
                b['width']=width
                lo,hi=self.rank*width,(self.rank+1)*width
                for start in range(lo,hi,adam_chunk_elements):
                    end=min(hi,start+adam_chunk_elements)
                    param=flat[start:end].detach()
                    opt=recipe.MasterAdamW([param],lr=lr)
                    self.optimizers.append(dict(key=f'{b["key"]}:{start}:{end}',kind=kind,
                                                bucket=b,param=param,opt=opt,start=start,end=end))
        torch.cuda.empty_cache()

    def allocate_gradients(self):
        for b in self.buckets:
            b['grad']=torch.zeros_like(b['flat'])
            for e in b['entries']:
                e['param'].grad=b['grad'][e['start']:e['end']].view(e['shape'])

    def zero_grad(self):
        for b in self.buckets:
            if b['grad'] is None:raise RuntimeError('allocate_gradients must run first')
            b['grad'].zero_()

    def average_gradients(self):
        if self.world==1:return
        handles=[dist.all_reduce(b['grad'],op=dist.ReduceOp.AVG,async_op=True) for b in self.buckets]
        for handle in handles:handle.wait()

    @torch.no_grad()
    def step(self,lr):
        self.lr=lr
        for entry in self.optimizers:
            p,opt=entry['param'],entry['opt']
            if entry['kind']=='adamw':p.grad=entry['bucket']['grad'][entry['start']:entry['end']]
            opt.param_groups[0]['lr']=lr
            state=opt.state[p]
            for key,value in list(state.items()):
                if torch.is_tensor(value):state[key]=value.to(device=p.device)
            opt.step()
            for key,value in list(state.items()):
                if torch.is_tensor(value):state[key]=value.to(device='cpu')
            if entry['kind']=='adamw':p.grad=None
        if self.world>1:
            for b in self.buckets:
                if b['kind']=='muon':
                    for owner,(start,end) in enumerate(b['ranges']):
                        if end>start:dist.broadcast(b['flat'][start:end],src=owner)
                else:
                    width=b['width'];local=b['flat'][self.rank*width:(self.rank+1)*width].clone()
                    dist.all_gather_into_tensor(b['flat'],local)
                    del local

    def state_dict(self):
        return dict(rank=self.rank,world=self.world,lr=self.lr,
                    optimizer_states={e['key']:e['opt'].state_dict() for e in self.optimizers},
                    layout=self.layout())

    def load_state_dict(self,state):
        if state['rank']!=self.rank or state['world']!=self.world:
            raise ValueError('resume requires the same optimizer-state shard assignment')
        if state['layout']!=self.layout():raise ValueError('optimizer parameter layout changed')
        self.lr=state['lr']
        for e in self.optimizers:
            # torch.optim.load_state_dict would cast FP32 masters to BF16 because
            # the live parameter is BF16. Restore these tensors directly instead.
            saved=state['optimizer_states'][e['key']]
            if len(saved['param_groups'])!=1:raise ValueError('unexpected optimizer groups')
            group=saved['param_groups'][0]
            if len(group['params'])!=1:raise ValueError('unexpected optimizer parameter count')
            saved_state=saved['state'].get(group['params'][0],{})
            e['opt'].state[e['param']]={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in saved_state.items()}
            for key,value in group.items():
                if key!='params':e['opt'].param_groups[0][key]=value

    def layout(self):
        return [dict(key=b['key'],count=b['count'],size=b['size'],
                     entries=[dict(name=e['name'],start=e['start'],end=e['end'],shape=list(e['shape'])) for e in b['entries']])
                for b in self.buckets]

    def summary(self):
        return dict(world=self.world,rank=self.rank,
                    groups={kind:sum(b['count'] for b in self.buckets if b['kind']==kind) for kind in ['muon','adamw']},
                    muon_matrices=sum(len(b['entries']) for b in self.buckets if b['kind']=='muon'),
                    local_update_elements=sum(e['param'].numel() for e in self.optimizers),
                    master_state_device='cpu',update_formulas='upstream moe/optim.py, unmodified')
