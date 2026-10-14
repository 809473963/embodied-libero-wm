#!/usr/bin/env python3
"""New-development C0 trajectories then causally selected fixed-H24 branches."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,random,traceback,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from scripts.collect_candidate_utility_oracle import atomic_npz,plan
from scripts.run_dense_relation_preflight import Pool as BasePool
from scripts.run_relational_event_h24 import AlignedRecorder
from src.probes.relational_event_sampling import candidates,NAMES
from src.probes.relational_event_labels import labels
from src.probes.relational_autorun import select_anchors
OUT=ROOT/'results/relational_autorun'

class Recorder(AlignedRecorder):
    def save(self,path):
        a={k:np.stack([s[k] for s in self.samples]) for k in self.samples[0]}
        a.update(actions=np.array(self.actions,np.float32),step=np.arange(len(self.samples)),natural_step=np.array(self.natural_steps),
            rgb=np.array(self.images,np.uint8),proprio=np.array(self.proprios),contact_offsets=np.array(self.offsets),
            contact_geom_pairs=np.array(self.pairs,int).reshape(-1,2),contact_wrenches=np.array(self.wrenches).reshape(-1,6),contact_frames=np.array(self.frames).reshape(-1,9),contact_distances=np.array(self.distances))
        assert len(a['step'])==len(a['actions'])+1 and all(np.isfinite(v).all() for v in a.values())
        atomic_npz(path,**a);return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),geometry=self.geometry)

def worker(conn,index,directory):
    from scripts.evaluate_c0_all import make_env,refresh_policy_observation
    from src.probes.robust_grounding import BOWLS,compatibility,resolve_scores
    env=None;rec=None
    def snapshot():
        o=refresh_policy_observation(env);t=len(rec.samples)-1;j=rec.natural_steps.index(t)
        p=np.r_[o['robot0_eef_pos'],o['robot0_gripper_qpos'],o['robot0_joint_pos']]
        np.testing.assert_array_equal(rec.images[j],o['agentview_image']);np.testing.assert_array_equal(rec.proprios[j],p)
        return {k:o[k] for k in ('agentview_image','robot0_eef_pos','robot0_gripper_qpos','robot0_joint_pos')}
    try:
        while True:
            req=conn.recv()
            if req['operation']=='close':break
            try:
                if req['operation']=='initialize':
                    if env:env.close()
                    scene=req['scene'];random.seed(scene['seed']);np.random.seed(scene['seed'])
                    env=make_env(ROOT/req['contract']['bddl_path']);env.seed(scene['seed']);env.env.reset()
                    assert sha256(ROOT/scene['initial'])==scene['initial_sha256']
                    with np.load(ROOT/scene['initial'],allow_pickle=False) as f:initial={k:f[k] for k in f.files}
                    env.set_init_state(initial['sim_state'])
                    rec=Recorder(env,req['capture_anchor'],True);rec.sample()
                    np.testing.assert_array_equal(rec.samples[0]['pose'],initial['pose']);np.testing.assert_array_equal(env.sim.get_state().flatten(),initial['sim_state'])
                    o=snapshot();np.testing.assert_array_equal(o['agentview_image'],initial['rgb'])
                    scores=np.array([compatibility(env.env,req['contract'],b,'G2')[0] for b in BOWLS]);status,ref,_=resolve_scores(scores,0.,20.);assert status=='UNIQUE' and ref==0
                    prefix=req['prefix']
                    for start in range(0,len(prefix),8):
                        for a in prefix[start:start+8]:env.step(a);rec.sample(a)
                        # Snapshot cadence matches the original C0 trajectory, even before the RGB history window.
                        refresh_policy_observation(env)
                    value=snapshot()
                elif req['operation']=='step':
                    for a in req['actions']:
                        assert a.shape==(7,) and np.isfinite(a).all() and (abs(a)<=1).all()
                        env.step(a);rec.sample(a)
                    value=snapshot()
                elif req['operation']=='finish':value=rec.save(Path(directory)/req['path'])
                else:raise ValueError(req['operation'])
                conn.send(dict(ok=True,value=value))
            except Exception:conn.send(dict(ok=False,error=traceback.format_exc()))
    finally:
        if env:env.close()
        conn.close()

class Pool(BasePool):
    def __init__(self):
        self.connections=[];self.processes=[];ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,str(OUT)));p.start();b.close();self.connections.append(a);self.processes.append(p)

def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def main(stage):
    cfg=json.loads((ROOT/'configs/relational_autorun_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True)
    if (target/'runtime.json').exists():raise RuntimeError('Existing stage retained; no automatic budget reset')
    rt=dict(status='RUNNING',stage=stage,completed=0,steps_submitted=0,started_epoch=time.time());atomic_json(target/'runtime.json',rt)
    pool=None
    try:
        contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks'];scenes=cfg['scenes'];pool=Pool();records=[]
        if stage=='trajectories':
            import torch
            from src.policies.c0_actor_policy import C0ActorPolicy
            torch.set_num_threads(1);policy=C0ActorPolicy(ROOT/'results/c0_v2/seed11/best.pt','cuda')
            for k,scene in enumerate(scenes):
                contract=contracts[scene['task_id']];key=f"task{scene['task_id']:02d}_seed{scene['seed']}"
                slots=3 if k==0 else 1;rt['steps_submitted']+=slots*300;assert rt['steps_submitted']<=6600
                rt.update(current_scene=key);atomic_json(target/'runtime.json',rt)
                obs=pool.call({i:dict(operation='initialize',scene=scene,contract=contract,prefix=np.empty((0,7)),capture_anchor=10) for i in range(slots)})
                for t in range(0,300,8):
                    aa=plan(policy,[obs[i] for i in range(slots)],contract['instruction']);n=min(8,300-t)
                    obs=pool.call({i:dict(operation='step',actions=aa[i,:n]) for i in range(slots)})
                    rt['episode_step']=t+n;atomic_json(target/'runtime.json',rt)
                saved=pool.call({i:dict(operation='finish',path=f'trajectories/arrays/{key}_r{i}.npz') for i in range(slots)})
                base=load(ROOT/saved[0]['path']);assert len(base['step'])==301
                for i in range(1,slots):
                    other=load(ROOT/saved[i]['path'])
                    for field in base:np.testing.assert_array_equal(base[field],other[field],err_msg=field)
                records.append(dict(scene=scene,root=key,**saved[0]));rt['completed']+=1;atomic_json(target/'runtime.json',rt)
                atomic_json(target/'manifest.json',dict(complete=False,records=records));print(json.dumps(dict(stage=stage,scenes=rt['completed'])),flush=True)
            assert rt['steps_submitted']==6600
        else:
            old=json.loads((OUT/'trajectories/manifest.json').read_text());assert old['complete']
            lc=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters'];cases=[];missing=[]
            for r in old['records']:
                assert sha256(ROOT/r['path'])==r['sha256'];a=load(ROOT/r['path']);merged={}
                for name,t in select_anchors(a,lc).items():
                    if t is None:missing.append(dict(root=r['root'],stratum=name))
                    else:merged.setdefault(t,[]).append(name)
                for t,strata in sorted(merged.items()):cases.append(dict(source=r,anchor=t,strata=strata))
            budget=6*sum(c['anchor']+24 for c in cases)+(3*(cases[0]['anchor']+24) if cases else 0)
            assert len(cases)<=100 and budget<=159192
            atomic_json(target/'cases.json',dict(cases=cases,missing=missing,exact_budget=budget,selected_before_branch_execution=True))
            for k,c in enumerate(cases):
                r=c['source'];scene=r['scene'];a=load(ROOT/r['path']);t=c['anchor'];plans=candidates(a['actions'][t:t+8],.15)
                batches=([[(0,f'preflight_{i}') for i in range(3)]] if k==0 else [])
                specs=[(i,f"{r['root']}_t{t:03d}_v{i}") for i in range(6)];batches+=[specs[:4],specs[4:]]
                for batch in batches:
                    rt['steps_submitted']+=len(batch)*(t+24);assert rt['steps_submitted']<=budget
                    rt.update(current_scene=r['root'],anchor=t);atomic_json(target/'runtime.json',rt)
                    pool.call({i:dict(operation='initialize',scene=scene,contract=contracts[scene['task_id']],prefix=a['actions'][:t],capture_anchor=t) for i in range(len(batch))})
                    pool.call({i:dict(operation='step',actions=plans[cid]) for i,(cid,_) in enumerate(batch)})
                    saved=pool.call({i:dict(operation='finish',path=f'branches/arrays/{name}.npz') for i,(_,name) in enumerate(batch)})
                    copies=[]
                    for i,(cid,name) in enumerate(batch):
                        b=load(ROOT/saved[i]['path']);copies.append(b)
                        for field in ('pose','qpos','qvel','pad_contact','gripper_contact','non_gripper_contact','on','grip_pos','grip_rot','grip_velocity'):
                            end=t+9 if cid==0 else t+1;np.testing.assert_array_equal(b[field][:end],a[field][:end],err_msg=field)
                        np.testing.assert_array_equal(b['actions'][:t],a['actions'][:t]);np.testing.assert_array_equal(b['actions'][t:],plans[cid])
                        if not name.startswith('preflight'):
                            records.append(dict(root=r['root'],scene=scene,anchor=t,strata=c['strata'],variant=cid,variant_name=NAMES[cid],source=r['path'],**saved[i]))
                    if batch[0][1].startswith('preflight'):
                        for b in copies[1:]:
                            for field in b:np.testing.assert_array_equal(copies[0][field],b[field])
                        atomic_json(target/'preflight.json',dict(pass_all=True))
                    rt['completed']+=len(batch);atomic_json(target/'runtime.json',rt);atomic_json(target/'manifest.json',dict(complete=False,records=records))
                    print(json.dumps(dict(stage=stage,runs=rt['completed'],total=6*len(cases)+(3 if cases else 0))),flush=True)
            assert rt['steps_submitted']==budget
        atomic_json(target/'manifest.json',dict(complete=True,records=records))
        rt.update(status='COMPLETE',finished_epoch=time.time());atomic_json(target/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(target/'runtime.json',rt);raise
    finally:
        if pool:pool.close()
if __name__=='__main__':main(sys.argv[1])
