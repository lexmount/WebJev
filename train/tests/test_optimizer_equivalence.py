"""The sharded optimizer must reproduce the upstream MasterMuon/MasterAdamW bit for bit, including resume.

Needs CUDA and at least two GPUs:
    torchrun --standalone --nproc_per_node 2 tests/test_optimizer_equivalence.py
"""
import io
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sharded_recipe_optim import ShardedRecipeOptim, recipe  # noqa: E402

dist.init_process_group('nccl')
rank=dist.get_rank();torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.manual_seed(123)
specs=[('model.layers.0.self_attn.q_proj.weight',(128,64),torch.bfloat16),
       ('model.layers.1.mlp.shared_expert.up_proj.weight',(96,128),torch.bfloat16),
       ('model.embed_tokens.weight',(137,64),torch.bfloat16),
       ('model.layers.1.mlp.gate.weight',(128,64),torch.bfloat16),
       ('model.layers.0.input_layernorm.weight',(73,),torch.float32)]
named=[(n,torch.nn.Parameter(torch.randn(s,device='cuda',dtype=d)*0.1)) for n,s,d in specs]
ref=[(n,torch.nn.Parameter(p.detach().clone())) for n,p in named]
ref_by_name=dict(ref)
opts=[recipe.MasterMuon([p for n,p in ref if recipe.is_muon_param(n,p)],lr=0.01),
      recipe.MasterAdamW([p for n,p in ref if not recipe.is_muon_param(n,p)],lr=0.01)]
ref_state={n:(opts[0] if recipe.is_muon_param(n,p) else opts[1],p) for n,p in ref}
sharded=ShardedRecipeOptim(named,lr=0.01,adam_chunk_elements=777)
sharded.allocate_gradients()
max_errors=[]
for step in range(1,6):
    sharded.zero_grad()
    torch.manual_seed(700+step+rank)
    for n,p in named:p.grad.copy_(torch.randn_like(p))
    sharded.average_gradients()
    for n,p in named:ref_by_name[n].grad=p.grad.clone()
    torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
    torch.nn.utils.clip_grad_norm_([p for _,p in ref],1.)
    sharded.step(0.01)
    for opt in opts:opt.step()
    for n,p in named:
        torch.testing.assert_close(p,ref_by_name[n],rtol=0,atol=0)
    for e in sharded.optimizers:
        state=e['opt'].state[e['param']]
        if e['kind']=='muon':
            opt,p=ref_state[e['name']]
            for key in ['master','momentum']:
                torch.testing.assert_close(state[key],opt.state[p][key].cpu(),rtol=0,atol=0)
        else:
            for key in ['master','exp_avg','exp_avg_sq']:
                values=[]
                for ent in e['bucket']['entries']:
                    opt,p=ref_state[ent['name']];values.append(opt.state[p][key].cpu().reshape(-1))
                expected=torch.cat(values)
                if expected.numel()<e['bucket']['size']:
                    expected=torch.cat([expected,torch.zeros(e['bucket']['size']-expected.numel())])
                torch.testing.assert_close(state[key],expected[e['start']:e['end']],rtol=0,atol=0)
    if step==3:
        buffer=io.BytesIO();torch.save(sharded.state_dict(),buffer);buffer.seek(0)
        state=torch.load(buffer,map_location='cpu',weights_only=False)
        sharded.load_state_dict(state)
        assert all(v.dtype==torch.float32 for e in sharded.optimizers for k,v in e['opt'].state[e['param']].items() if torch.is_tensor(v))
    max_errors.append(0.0)
dist.barrier()
if rank==0:
    report=dict(passed=True,world=dist.get_world_size(),steps=5,parameter_max_abs_errors=max_errors,
                fp32_master_and_moment_max_abs_error=0.0,resume_state_verified=True,
                reference='upstream MasterMuon and MasterAdamW',
                torch=torch.__version__)
    print(json.dumps(report),flush=True)
    if os.environ.get('WEBJEV_TEST_REPORT'):
        Path(os.environ['WEBJEV_TEST_REPORT']).write_text(json.dumps(report,indent=2)+'\n')
dist.destroy_process_group()
