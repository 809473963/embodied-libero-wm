"""Bounded H8 privileged dynamics: causal features, grouped data and metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

FIELDS = ('pose','contacts','gripper_contact','pad_contact','non_gripper_contact',
          'grip_pos','grip_rot','grip_velocity','qpos','qvel','on','actions','step')
BINARY_NAMES = [f'{b}/{s}' for b in ('target','other') for s in
    ('any_grip','bilateral','external_contact','on','attachment','detached','lift','quiet_on','release')]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relation(a):
    pos=a['pose'][:,:2,:3]
    plate=pos-a['pose'][:,2:3,:3]
    local=np.einsum('tji,tkj->tki',a['grip_rot'],pos-a['grip_pos'][:,None,:])
    return np.concatenate([plate,local],axis=2).reshape(len(pos),12)


def binary_state(a,v):
    return np.concatenate([a['gripper_contact'][:,:2,None],
        a['pad_contact'][:,:,:2].all(1)[:,:,None],a['non_gripper_contact'][:,:2,None],
        a['on'][:,:,None],v],axis=2).reshape(len(v),18).astype(np.int8)


def features(a,v):
    """All outputs at t depend only on observations/actions at or before t."""
    n=len(v);columns=[];schema=[]
    def add(name,x):
        x=np.asarray(x).reshape(n,-1);start=sum(c.shape[1] for c in columns)
        columns.append(x);schema.append(dict(name=name,start=start,stop=start+x.shape[1]))
    add('entity_world_pose_velocity',a['pose'])
    add('bowls_relative_plate_and_gripper',relation(a))
    for k in ('contacts','gripper_contact','pad_contact','non_gripper_contact','grip_pos','grip_rot','grip_velocity','on'):
        add(k,a[k])
    for k in ('qpos','qvel'):
        x=np.asarray(a[k]);assert x.shape[1]<=64
        padded=np.zeros((n,64));valid=np.zeros((n,64));padded[:,:x.shape[1]]=x;valid[:,:x.shape[1]]=1
        add(k,padded);add(k+'_valid',valid)
    add('event_value_unknown_zero',np.maximum(v,0));add('event_known',v>=0)
    # Compact causal memory; do not require recovering long-lived referents from 11 frames.
    memory=np.zeros((n,2,3))
    for b in range(2):
        last=None;seen=False
        for t in range(n):
            if a['non_gripper_contact'][t,b]:last=t
            seen=seen or v[t,b,1]==1
            memory[t,b]=[0 if last is None else a['pose'][t,b,2]-a['pose'][last,b,2],last is not None,seen]
    add('contact_height_reference_and_seen_detached',memory)
    add('previous_executed_action',np.concatenate([np.zeros((1,7)),a['actions'][:n-1]],axis=0))
    x=np.concatenate(columns,axis=1).astype(np.float32);assert np.isfinite(x).all()
    return x,schema


def causal_window(x,t,length=11):
    indexes=np.arange(t-length+1,t+1);valid=indexes>=0
    return np.concatenate([x[np.maximum(indexes,0)],valid[:,None]],axis=1)


def anchors(prefix,cid):
    # Prefix is fixed replay and included once. Post-root chunks are planned open-loop H8.
    return list(range(0,prefix,8) if cid==0 else [])+list(range(prefix,293,8))


def normalization(x,actions,y,fit,weights):
    assert fit.any()
    hx=x[fit];w=weights[fit,None]*hx[:,:,-1];w=w/w.sum()
    mean=(hx*w[:,:,None]).sum((0,1));std=np.sqrt(((hx-mean)**2*w[:,:,None]).sum((0,1))).clip(.001)
    mean[-1]=0;std[-1]=1
    aw=np.broadcast_to(weights[fit,None],actions[fit].shape[:2]);aw=aw/aw.sum()
    am=(actions[fit]*aw[:,:,None]).sum((0,1));ast=np.sqrt(((actions[fit]-am)**2*aw[:,:,None]).sum((0,1))).clip(.001)
    yw=weights[fit]/weights[fit].sum();ym=(y[fit]*yw[:,None]).sum(0)
    ys=np.sqrt(((y[fit]-ym)**2*yw[:,None]).sum(0)).clip(.001)
    return dict(x_mean=mean,x_std=std,action_mean=am,action_std=ast,delta_scale=ys)


def build_dataset(root):
    manifest=json.loads((root/'results/relational_coverage_expansion/coverage.json').read_text())
    xs=[];acts=[];ys=[];bs=[];cur=[];meta=[];schema=None
    for r in sorted(manifest['records'],key=lambda r:(r['root_id'],r['candidate_id'])):
        assert sha(root/r['source'])==r['source_sha256']
        assert sha(root/r['labels'])==r['labels_sha256']
        with np.load(root/r['source'],allow_pickle=False) as f:a={k:f[k] for k in FIELDS}
        with np.load(root/r['labels'],allow_pickle=False) as f:v=f['values']
        x,s=features(a,v);assert schema is None or schema==s;schema=s
        rel=relation(a);binary=binary_state(a,v)
        for t in anchors(r['root_step'],r['candidate_id']):
            xs.append(causal_window(x,t));acts.append(a['actions'][t:t+8]);ys.append(rel[t+8]-rel[t])
            bs.append(binary[t+8]);cur.append(binary[t])
            meta.append(dict(root=r['root_id'],candidate=r['candidate_id'],step=t,
                root_step=r['root_step'],phase='prefix' if t<r['root_step'] else ('root' if t==r['root_step'] else 'continuation'),
                split='development' if r['root_id'].endswith('34') else 'fit'))
    ids=np.array([m['root'] for m in meta]);fit=np.array([m['split']=='fit' for m in meta])
    assert len(set(ids[fit]))==30 and len(set(ids[~fit]))==10 and not set(ids[fit])&set(ids[~fit])
    weights=np.array([1/(ids==r).sum() for r in ids],np.float32)
    data=dict(x=np.array(xs,np.float32),actions=np.array(acts,np.float32),delta=np.array(ys,np.float32),
              binary=np.array(bs,np.int8),current=np.array(cur,np.int8),fit=fit,weights=weights)
    # Same-root candidates must start with identical inputs, while their future plans may differ.
    for rid in set(ids):
        ii=[i for i,m in enumerate(meta) if m['root']==rid and m['phase']=='root']
        assert len(ii)==3
        for i in ii[1:]:np.testing.assert_array_equal(data['x'][ii[0]],data['x'][i])
    return data,meta,schema


class Dynamics(nn.Module):
    def __init__(self,features):
        super().__init__();self.history=nn.GRU(features,64,batch_first=True)
        self.action=nn.Sequential(nn.Linear(56,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(96,128),nn.ReLU(),nn.Linear(128,30))
    def forward(self,x,a):
        _,h=self.history(x);return self.head(torch.cat([h[-1],self.action(a.flatten(1))],dim=1))


def prf(truth,pred):
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    return dict(tp=tp,fp=fp,fn=fn,positives=int(truth.sum()),predicted=int(pred.sum()),
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def score(delta,prob,data,meta,mask):
    groups=np.array([m['root'] for m in meta]);per_root={}
    # Two target vectors are the primary outcome; other bowl is reported separately.
    error=np.linalg.norm((delta-data['delta']).reshape(-1,4,3),axis=2)*1000
    changed=np.linalg.norm(data['delta'][:,:3],axis=1)>.001
    for rid in sorted(set(groups[mask])):
        ii=mask&(groups==rid);per_root[rid]=dict(n=int(ii.sum()),target_plate_mm=float(error[ii,0].mean()),
            target_gripper_mm=float(error[ii,1].mean()),other_plate_mm=float(error[ii,2].mean()),
            target_plate_changed_mm=float(error[ii&changed,0].mean()) if (ii&changed).any() else None)
    out=dict(per_root=per_root,root_macro_target_plate_mm=float(np.mean([r['target_plate_mm'] for r in per_root.values()])),
        root_macro_target_gripper_mm=float(np.mean([r['target_gripper_mm'] for r in per_root.values()])),
        changed_windows=int((mask&changed).sum()),unchanged_windows=int((mask&~changed).sum()),
        changed_target_plate_mm=float(error[mask&changed,0].mean()) if (mask&changed).any() else None,
        unchanged_target_plate_mm=float(error[mask&~changed,0].mean()) if (mask&~changed).any() else None,binary={})
    for j,name in enumerate(BINARY_NAMES):
        valid=mask&(data['binary'][:,j]>=0)&np.isfinite(prob[:,j])
        truth=data['binary'][valid,j]==1;pred=prob[valid,j]>=.5
        v=dict(n=int(valid.sum()),**prf(truth,pred));v['brier']=float(((prob[valid,j]-truth)**2).mean()) if valid.any() else None
        for prior,ending,label in ((0,1,'onset'),(1,0,'offset')):
            ok=valid&(data['current'][:,j]==prior)
            v[label]=dict(n=int(ok.sum()),**prf(data['binary'][ok,j]==ending,(prob[ok,j]>=.5)==bool(ending)))
        out['binary'][name]=v
    return out


def pair_score(delta,data,meta,mask):
    rows=[]
    for rid in sorted({m['root'] for i,m in enumerate(meta) if mask[i]}):
        ix={m['candidate']:i for i,m in enumerate(meta) if m['root']==rid and m['phase']=='root'}
        for cid in (49,50):
            a,b=ix[0],ix[cid];truth=data['delta'][b,:3]-data['delta'][a,:3];pred=delta[b,:3]-delta[a,:3]
            rows.append(dict(root=rid,candidate=cid,truth_norm_mm=float(np.linalg.norm(truth)*1000),
                error_mm=float(np.linalg.norm(pred-truth)*1000),
                dot_positive=bool(np.dot(pred,truth)>0),prediction_norm_mm=float(np.linalg.norm(pred)*1000)))
    material=[r for r in rows if r['truth_norm_mm']>1]
    return dict(n_pairs=len(rows),n_roots=len({r['root'] for r in rows}),material_pairs=len(material),
        material_roots=len({r['root'] for r in material}),mean_error_mm=float(np.mean([r['error_mm'] for r in rows])),
        material_mean_error_mm=float(np.mean([r['error_mm'] for r in material])) if material else None,
        material_direction_fraction=float(np.mean([r['dot_positive'] for r in material])) if material else None,rows=rows)
