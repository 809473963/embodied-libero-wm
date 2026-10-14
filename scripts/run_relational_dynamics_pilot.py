#!/usr/bin/env python3
"""Frozen six-fit offline pilot. No simulator, visual model or controller entry."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from src.probes.relational_dynamics_pilot import (sha,build_dataset,normalization,Dynamics,score,pair_score,BINARY_NAMES)
OUT=ROOT/'results/relational_dynamics_pilot'

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing pilot retained; no automatic reruns')
    protocol=ROOT/'configs/relational_dynamics_pilot_protocol.json';cfg=json.loads(protocol.read_text())
    for p,h in cfg['frozen_files'].items():assert sha(ROOT/p)==h,p
    runtime=dict(status='RUNNING',phase='dataset',pid=os.getpid(),started_epoch=time.time(),completed_fits=0,maximum_fits=6,protocol_sha256=sha(protocol),simulation=False)
    write(OUT/'runtime.json',runtime)
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device(cfg['device']);data,meta,schema=build_dataset(ROOT)
        fit=data['fit'];dev=~fit;weights=data['weights'];stats=normalization(data['x'],data['actions'],data['delta'],fit,weights)
        np.savez_compressed(OUT/'normalization.npz',**stats)
        np.savez_compressed(OUT/'dataset.npz',**data)
        write(OUT/'samples.json',meta);write(OUT/'feature_schema.json',schema)
        x=(data['x']-stats['x_mean'])/stats['x_std'];act=(data['actions']-stats['action_mean'])/stats['action_std']
        x=torch.tensor(x,dtype=torch.float32,device=device);act=torch.tensor(act,dtype=torch.float32,device=device)
        y=torch.tensor(data['delta']/stats['delta_scale'],dtype=torch.float32,device=device)
        binary=torch.tensor(np.maximum(data['binary'],0),dtype=torch.float32,device=device)
        valid=torch.tensor(data['binary']>=0,dtype=torch.float32,device=device)
        known=data['binary'][fit]>=0;positive=data['binary'][fit]==1
        fw=weights[fit,None];pos=(fw*positive).sum(0);neg=(fw*known*~positive).sum(0)
        posweight=np.clip(neg/np.maximum(pos,1e-12),1,20)
        pw=torch.tensor(posweight,dtype=torch.float32,device=device)
        wt=torch.tensor(weights/weights[fit].mean(),dtype=torch.float32,device=device)
        write(OUT/'dataset_summary.json',dict(samples=len(meta),fit_samples=int(fit.sum()),development_samples=int(dev.sum()),
            fit_roots=30,development_roots=10,feature_dimension=int(x.shape[-1]),history=11,
            positive_weights=dict(zip(BINARY_NAMES,posweight.tolist())),
            prefix_deduplicated=True,all_root_candidate_histories_identical=True,normalization_fit_only=True))
        baseline_prob=np.where(data['current']>=0,data['current'],np.nan).astype(float)
        zero=np.zeros_like(data['delta']);predictions={};evaluations={}
        for name,delta,prob in [('unchanged',zero,baseline_prob)]:
            evaluations[name]={s:score(delta,prob,data,meta,m) for s,m in [('fit',fit),('development',dev)]}
            evaluations[name]['development_pairs']=pair_score(delta,data,meta,dev)
        ids=np.flatnonzero(fit)
        for seed in cfg['seeds']:
            for family in ('no_action','action'):
                torch.manual_seed(seed);np.random.seed(seed);model=Dynamics(x.shape[-1]).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                rng=np.random.default_rng(seed);losses=[];family_act=torch.zeros_like(act) if family=='no_action' else act
                name=f'{family}_seed{seed}'
                for epoch in range(cfg['epochs']):
                    model.train();total=0.;nb=0
                    for start in range(0,len(ids),cfg['batch_size']):
                        # Order is fixed per epoch and paired across families.
                        if start==0:order=rng.permutation(ids)
                        ii=torch.tensor(order[start:start+cfg['batch_size']],device=device)
                        out=model(x[ii],family_act[ii]);continuous=F.smooth_l1_loss(out[:,:12],y[ii],reduction='none').mean(1)
                        bce=F.binary_cross_entropy_with_logits(out[:,12:],binary[ii],pos_weight=pw,reduction='none')
                        aux=(bce*valid[ii]).sum(1)/valid[ii].sum(1).clamp_min(1)
                        loss=((continuous+cfg['binary_loss_weight']*aux)*wt[ii]).mean()
                        assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                        total+=float(loss.detach());nb+=1
                    losses.append(total/nb)
                    runtime.update(phase='training',model=name,epoch=epoch+1,epochs=cfg['epochs'],updated_epoch=time.time());write(OUT/'runtime.json',runtime)
                    if (epoch+1)%20==0:print(json.dumps(dict(model=name,epoch=epoch+1,loss=losses[-1])),flush=True)
                model.eval()
                def predict(actions):
                    output=[]
                    with torch.no_grad():
                        for start in range(0,len(meta),256):output.append(model(x[start:start+256],actions[start:start+256]).cpu())
                    output=torch.cat(output)
                    return output[:,:12].numpy()*stats['delta_scale'],torch.sigmoid(output[:,12:]).numpy()
                delta,prob=predict(family_act)
                assert np.isfinite(delta).all() and np.isfinite(prob).all()
                predictions[name]=(delta,prob)
                np.savez_compressed(OUT/f'{name}_predictions.npz',delta=delta,probability=prob)
                torch.save(dict(state_dict=model.cpu().state_dict(),feature_dimension=int(x.shape[-1]),protocol_sha256=sha(protocol)),OUT/f'{name}.pt')
                evaluations[name]={s:score(delta,prob,data,meta,m) for s,m in [('fit',fit),('development',dev)]}
                evaluations[name]['development_pairs']=pair_score(delta,data,meta,dev)
                evaluations[name]['training_loss']=losses
                if family=='action':
                    model.to(device);permutation=np.arange(len(meta))
                    for rid in {m['root'] for m in meta}:
                        ix=sorted([i for i,m in enumerate(meta) if m['root']==rid and m['phase']=='root'],key=lambda i:meta[i]['candidate'])
                        permutation[ix]=np.roll(ix,-1)
                    sd,sp=predict(act[torch.tensor(permutation,device=device)])
                    np.savez_compressed(OUT/f'{name}_root_action_permuted.npz',delta=sd,probability=sp)
                    evaluations[name]['permuted_development_pairs']=pair_score(sd,data,meta,dev)
                runtime['completed_fits']+=1;write(OUT/'runtime.json',runtime)
        for family in ('no_action','action'):
            delta=np.mean([predictions[f'{family}_seed{s}'][0] for s in cfg['seeds']],axis=0)
            prob=np.mean([predictions[f'{family}_seed{s}'][1] for s in cfg['seeds']],axis=0)
            np.savez_compressed(OUT/f'{family}_ensemble_predictions.npz',delta=delta,probability=prob)
            evaluations[family+'_ensemble']={s:score(delta,prob,data,meta,m) for s,m in [('fit',fit),('development',dev)]}
            evaluations[family+'_ensemble']['development_pairs']=pair_score(delta,data,meta,dev)
        seed_checks=[]
        for s in cfg['seeds']:
            a=evaluations[f'action_seed{s}'];n=evaluations[f'no_action_seed{s}'];b=evaluations['unchanged']
            continuous=all(a['development'][metric]<min(n['development'][metric],b['development'][metric]) for metric in ('root_macro_target_plate_mm','root_macro_target_gripper_mm'))
            ap=a['development_pairs']['material_mean_error_mm'];bp=b['development_pairs']['material_mean_error_mm'];npair=n['development_pairs']['material_mean_error_mm']
            pair=ap is not None and ap<.9*min(bp,npair)
            seed_checks.append(dict(seed=s,continuous_both_better=continuous,material_pair_improvement_10pct=pair,joint_pass=continuous and pair))
        verdict='ACTION_SIGNAL_PILOT_SUPPORTED' if sum(r['joint_pass'] for r in seed_checks)>=2 else 'NO_CONSISTENT_ACTION_CONDITIONED_ADVANTAGE'
        results=dict(verdict=verdict,seed_checks=seed_checks,protocol_sha256=sha(protocol),evaluations=evaluations,
            controller_authorized=False,new_scene_generalization=False,natural_perception_tested=False)
        write(OUT/'evaluation.json',results)
        # Verification material is written before the completion flag.
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='runtime.json'}
        write(OUT/'artifact_manifest.json',artifacts)
        runtime.update(status='COMPLETE',phase='awaiting_result_review',finished_epoch=time.time(),wall_seconds=time.time()-runtime['started_epoch'],verdict=verdict)
        write(OUT/'runtime.json',runtime);print(json.dumps(runtime),flush=True)
    except Exception:
        runtime.update(status='BLOCKED',error=traceback.format_exc(),scientific_conclusion='not assigned');write(OUT/'runtime.json',runtime);raise

if __name__=='__main__':main()
