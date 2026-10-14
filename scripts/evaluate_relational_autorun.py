#!/usr/bin/env python3
"""Validate new branches, fit fixed H24 models on OLD data only, evaluate new development."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import features,causal_window,relation,binary_state,normalization,score,Dynamics
from src.probes.relational_event_recording import validate
from src.probes.relational_event_labels import labels,EVENTS
OUT=ROOT/'results/relational_autorun'

class H24Dynamics(Dynamics):
    def __init__(self,d):
        super().__init__(d);self.action=nn.Sequential(nn.Linear(168,32),nn.ReLU())

def paired(delta,d,meta,mask):
    rows=[]
    for root,anchor in sorted({(m['root'],m['anchor']) for i,m in enumerate(meta) if mask[i]}):
        ix={m['variant']:i for i,m in enumerate(meta) if m['root']==root and m['anchor']==anchor};assert set(ix)==set(range(6))
        for cid in range(1,6):
            true=d['delta'][ix[cid],:3]-d['delta'][ix[0],:3];pred=delta[ix[cid],:3]-delta[ix[0],:3]
            rows.append(dict(root=root,anchor=anchor,variant=cid,truth_mm=float(np.linalg.norm(true)*1000),error_mm=float(np.linalg.norm(pred-true)*1000)))
    groups=sorted({r['root'] for r in rows})
    return dict(pairs=len(rows),roots=len(groups),root_macro_error_mm=float(np.mean([np.mean([r['error_mm'] for r in rows if r['root']==root]) for root in groups])),
        material_pairs=sum(r['truth_mm']>1 for r in rows),rows=rows)

def main():
    directory=OUT/'models';directory.mkdir(exist_ok=True)
    if (directory/'runtime.json').exists():raise RuntimeError('Existing model attempt retained')
    cfg=json.loads((ROOT/'configs/relational_autorun_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    rt=dict(status='RUNNING',phase='validation',started_epoch=time.time(),completed_fits=0,maximum_fits=6);atomic_json(directory/'runtime.json',rt)
    try:
        lc=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters']
        old=json.loads((ROOT/'results/relational_event_h24/manifest.json').read_text())['records']
        newmanifest=json.loads((OUT/'branches/manifest.json').read_text());assert newmanifest['complete'];new=newmanifest['records']
        xx=[];aa=[];yy=[];bb=[];cc=[];meta=[];support=[];histories={};schema=None
        for split,records in [('old_fit',old),('new_development',new)]:
            for r in records:
                path=r['dense']['arrays'] if split=='old_fit' else r['path'];digest=r['dense']['sha256'] if split=='old_fit' else r['sha256']
                geometry=r['dense']['geometry'] if split=='old_fit' else r['geometry']
                assert sha256(ROOT/path)==digest
                with np.load(ROOT/path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
                t=r['anchor'];validate(a,geometry,t)
                v=np.stack([labels(a,lc,i)['values'] for i in (0,1)],axis=1)
                feat,s=features(a,v);assert schema is None or schema==s;schema=s
                x=causal_window(feat,t);root=r['root_id'] if split=='old_fit' else r['root'];key=(root,t)
                if key in histories:np.testing.assert_array_equal(x,histories[key])
                else:histories[key]=x
                rel=relation(a);b=binary_state(a,v);xx.append(x);aa.append(a['actions'][t:t+24]);yy.append(rel[t+24]-rel[t]);bb.append(b[t+24]);cc.append(b[t])
                meta.append(dict(root=root,anchor=t,variant=r['variant'],split=split))
                support.append(dict(root=root,anchor=t,variant=r['variant'],split=split,current_events=v[t].tolist(),future_events=v[t+24].tolist(),current_grip=bool(a['gripper_contact'][t,0])))
        d=dict(x=np.array(xx,np.float32),actions=np.array(aa,np.float32),delta=np.array(yy,np.float32),binary=np.array(bb,np.int8),current=np.array(cc,np.int8))
        fit=np.array([m['split']=='old_fit' for m in meta]);dev=~fit;ids=np.array([m['root'] for m in meta])
        assert not set(ids[fit])&set(ids[dev]);weights=np.array([1/(ids==r).sum() for r in ids],np.float32)
        coverage={}
        for split in ('old_fit','new_development'):
            rows=[r for r in support if r['split']==split];coverage[split]={}
            for j,event in enumerate(EVENTS):
                pos=[r for r in rows if r['current_events'][0][j]==0 and r['future_events'][0][j]==1]
                coverage[split][event]=dict(onset_branches=len(pos),onset_roots=sorted({r['root'] for r in pos}))
            active=[r for r in rows if r['current_events'][0][1]==1 and r['future_events'][0][4]==1]
            coverage[split]['held_to_release']=dict(branches=len(active),roots=sorted({r['root'] for r in active}))
        atomic_json(OUT/'coverage.json',dict(events=coverage,rows=support,new_scenes_with_anchors=len(set(ids[dev])),new_branches=int(dev.sum())))
        if not dev.any():
            atomic_json(OUT/'evaluation.json',dict(verdict='NO_ELIGIBLE_NEW_ANCHORS',training=False));rt.update(status='COMPLETE',phase='insufficient_data_no_training');atomic_json(directory/'runtime.json',rt);return
        st=normalization(d['x'],d['actions'],d['delta'],fit,weights)
        np.savez_compressed(directory/'dataset.npz',**d,fit=fit,weights=weights);np.savez_compressed(directory/'normalization.npz',**st)
        atomic_json(directory/'samples.json',meta);atomic_json(directory/'schema.json',schema)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device='cuda';x=torch.tensor((d['x']-st['x_mean'])/st['x_std'],device=device);actions=torch.tensor((d['actions']-st['action_mean'])/st['action_std'],device=device)
        y=torch.tensor(d['delta']/st['delta_scale'],device=device);binary=torch.tensor(np.maximum(d['binary'],0),dtype=torch.float32,device=device);valid=torch.tensor(d['binary']>=0,device=device)
        fw=weights[fit,None];pos=(fw*(d['binary'][fit]==1)).sum(0);neg=(fw*(d['binary'][fit]==0)).sum(0)
        pw=torch.tensor(np.clip(neg/np.maximum(pos,1e-12),1,20),device=device);wt=torch.tensor(weights/weights[fit].mean(),device=device)
        ev={};predictions={}
        def metrics(delta,prob):
            return dict(old_fit=score(delta,prob,d,meta,fit),new_development=score(delta,prob,d,meta,dev),new_pairs=paired(delta,d,meta,dev))
        ev['unchanged']=metrics(np.zeros_like(d['delta']),np.where(d['current']>=0,d['current'],np.nan))
        for seed in (11,23,47):
            for family in ('no_action','action'):
                torch.manual_seed(seed);model=H24Dynamics(x.shape[-1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                rng=np.random.default_rng(seed);a=torch.zeros_like(actions) if family=='no_action' else actions;name=f'{family}_seed{seed}';losses=[]
                for epoch in range(80):
                    order=rng.permutation(np.flatnonzero(fit));model.train();loss_sum=0
                    for start in range(0,len(order),128):
                        ii=torch.tensor(order[start:start+128],device=device);out=model(x[ii],a[ii])
                        cont=F.smooth_l1_loss(out[:,:12],y[ii],reduction='none').mean(1)
                        bce=F.binary_cross_entropy_with_logits(out[:,12:],binary[ii],pos_weight=pw,reduction='none')
                        aux=(bce*valid[ii]).sum(1)/valid[ii].sum(1).clamp_min(1);loss=((cont+.2*aux)*wt[ii]).mean();assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
                    losses.append(loss_sum);rt.update(phase='training_old_data_only',model=name,epoch=epoch+1);atomic_json(directory/'runtime.json',rt)
                model.eval()
                with torch.no_grad():out=torch.cat([model(x[i:i+128],a[i:i+128]).cpu() for i in range(0,len(meta),128)])
                delta=out[:,:12].numpy()*st['delta_scale'];prob=torch.sigmoid(out[:,12:]).numpy();assert np.isfinite(delta).all()
                predictions[name]=(delta,prob);ev[name]=metrics(delta,prob);ev[name]['fit_loss']=losses
                np.savez_compressed(directory/f'{name}_predictions.npz',delta=delta,probability=prob)
                torch.save(dict(state_dict=model.cpu().state_dict(),feature_dimension=int(x.shape[-1]),horizon=24),directory/f'{name}.pt')
                rt['completed_fits']+=1;atomic_json(directory/'runtime.json',rt);print(json.dumps(dict(stage='models',completed_fits=rt['completed_fits'])),flush=True)
        for family in ('no_action','action'):
            delta=np.mean([predictions[f'{family}_seed{s}'][0] for s in (11,23,47)],axis=0);prob=np.mean([predictions[f'{family}_seed{s}'][1] for s in (11,23,47)],axis=0)
            name=family+'_ensemble';ev[name]=metrics(delta,prob);np.savez_compressed(directory/f'{name}_predictions.npz',delta=delta,probability=prob)
        checks=[]
        for seed in (11,23,47):
            a=ev[f'action_seed{seed}'];n=ev[f'no_action_seed{seed}'];b=ev['unchanged']
            ok=all(a['new_development'][k]<min(n['new_development'][k],b['new_development'][k]) for k in ('root_macro_target_plate_mm','root_macro_target_gripper_mm')) and a['new_pairs']['root_macro_error_mm']<.9*min(n['new_pairs']['root_macro_error_mm'],b['new_pairs']['root_macro_error_mm'])
            checks.append(dict(seed=seed,joint_pass=ok))
        verdict='H24_NEW_DEVELOPMENT_ACTION_SIGNAL_SUPPORTED' if len(set(ids[dev]))>=5 and sum(c['joint_pass'] for c in checks)>=2 else 'H24_ADVANTAGE_NOT_ESTABLISHED'
        atomic_json(OUT/'evaluation.json',dict(verdict=verdict,seed_checks=checks,models=ev,confirmation_opened=False,new_data_used_for_fitting=False))
        rt.update(status='COMPLETE',phase='trained_and_evaluated',finished_epoch=time.time());atomic_json(directory/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(directory/'runtime.json',rt);raise
if __name__=='__main__':main()
