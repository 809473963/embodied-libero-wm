#!/usr/bin/env python3
"""Bounded dense-sensor instrumentation of the frozen Stage1 execution worker.
No old files are modified. Dense sampling only reads simulator fields. Its effect
is tested against root/H8 identity, action hashes, and duplicated A0 execution.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import json
import time
import hashlib
import multiprocessing as mp
import traceback
import numpy as np
from src.probes.candidate_utility_oracle import sha256, selective_init_states, identity, atomic_json
from scripts.collect_candidate_utility_oracle import plan, initialize_request, atomic_npz

OUT=ROOT/'results/dense_relation_preflight'
PROTOCOL=ROOT/'configs/dense_relation_preflight_protocol.json'
ENTITIES=['akita_black_bowl_1','akita_black_bowl_2','plate_1','cookies_1','glazed_rim_porcelain_ramekin_1','main_table','wooden_cabinet_1','flat_stove_1']


class Recorder:
    def __init__(self,env):
        import mujoco
        self.mujoco=mujoco;self.env=env;self.inner=env.env
        sim=env.sim;robot=self.inner.robots[0];self.robot=robot
        self.site=robot.gripper.important_sites['grip_site']
        self.bodies=[];self.owners={}
        for i,name in enumerate(ENTITIES):
            model=self.inner.get_object(name)
            body='table' if model is None and name=='main_table' else model.root_body
            geoms=['table_collision'] if model is None and name=='main_table' else model.contact_geoms
            self.bodies.append(body)
            for g in geoms:self.owners[sim.model.geom_name2id(g)]=i
        self.gripper={sim.model.geom_name2id(g) for g in robot.gripper.contact_geoms}
        self.left={sim.model.geom_name2id(g) for g in robot.gripper.important_geoms['left_fingerpad']}
        self.right={sim.model.geom_name2id(g) for g in robot.gripper.important_geoms['right_fingerpad']}
        assert self.left and self.right and self.left<=self.gripper and self.right<=self.gripper
        self.geometry={'entities':ENTITIES,'bodies':self.bodies,'geom_names':[sim.model.geom_id2name(i) for i in range(sim.model.ngeom)],
                       'geom_entity_owner':self.owners,'gripper_ids':sorted(self.gripper),'left_pad_ids':sorted(self.left),'right_pad_ids':sorted(self.right),
                       'control_freq':self.inner.control_freq,'model_timestep':float(sim.model.opt.timestep),
                       'contact_wrench':'MuJoCo contact frame, normal is first force component; final solver snapshot only, not control-interval impulse',
                       'body_position':'world body frame reference, not surface clearance'}
        self.reset()
    def reset(self):
        self.samples=[];self.actions=[];self.pairs=[];self.wrenches=[];self.frames=[];self.distances=[];self.offsets=[0];self.active=False
    def sample(self,action=None):
        sim=self.env.sim;inner=self.inner
        pose=np.array([np.r_[sim.data.body_xpos[sim.model.body_name2id(b)],sim.data.body_xquat[sim.model.body_name2id(b)],sim.data.get_body_xvelp(b),sim.data.get_body_xvelr(b)] for b in self.bodies])
        contacts=np.zeros((8,8),bool);anygrip=np.zeros(8,bool);pads=np.zeros((2,8),bool);nonrobot=np.zeros(8,bool)
        for k in range(sim.data.ncon):
            c=sim.data.contact[k];g1,g2=int(c.geom1),int(c.geom2);i,j=self.owners.get(g1),self.owners.get(g2)
            if i is not None and j is not None:contacts[i,j]=contacts[j,i]=True
            for target,other in [(i,g2),(j,g1)]:
                if target is not None:
                    anygrip[target]|=other in self.gripper
                    pads[0,target]|=other in self.left;pads[1,target]|=other in self.right
                    nonrobot[target]|=other not in self.gripper and self.owners.get(other)!=target
            force=np.zeros(6,np.float64)
            self.mujoco.mj_contactForce(sim.model._model,sim.data._data,k,force)
            assert np.isfinite(force).all()
            self.pairs.append([g1,g2]);self.wrenches.append(force);self.frames.append(np.array(c.frame).copy());self.distances.append(float(c.dist))
        self.offsets.append(len(self.pairs))
        grasp=bool(inner._check_grasp(gripper=self.robot.gripper,object_geoms=inner.get_object(ENTITIES[0])))
        assert grasp==bool(pads[:,0].all()), 'pad contact mapping differs from official _check_grasp'
        self.samples.append({'pose':pose,'contacts':contacts,'gripper_contact':anygrip,'pad_contact':pads,'non_gripper_contact':nonrobot,
            'grip_pos':np.array(sim.data.get_site_xpos(self.site)), 'grip_rot':np.array(sim.data.get_site_xmat(self.site)),
            'grip_velocity':np.r_[sim.data.get_site_xvelp(self.site),sim.data.get_site_xvelr(self.site)],
            'qpos':sim.data.qpos.copy(),'qvel':sim.data.qvel.copy(),
            'on':np.array([bool(inner._eval_predicate(['on',b,'plate_1'])) for b in ENTITIES[:2]]),
            'success':np.array(bool(self.env.check_success())),'grasp':np.array(grasp)})
        if action is not None:self.actions.append(np.asarray(action,np.float32).copy())
    def save(self,path):
        arrays={k:np.stack([r[k] for r in self.samples]) for k in self.samples[0]}
        arrays.update(actions=np.array(self.actions,np.float32),step=np.arange(len(self.samples),dtype=np.int32),
            contact_offsets=np.array(self.offsets,np.int64),contact_geom_pairs=np.array(self.pairs,np.int32).reshape(-1,2),
            contact_wrenches=np.array(self.wrenches,np.float64).reshape(-1,6),contact_frames=np.array(self.frames,np.float64).reshape(-1,9),
            contact_distances=np.array(self.distances,np.float64))
        assert len(self.samples)==301 and len(self.actions)==300
        for v in arrays.values():assert np.isfinite(v).all()
        atomic_npz(path,**arrays)
        return {'arrays':str(path.relative_to(ROOT)),'sha256':sha256(path),'samples':len(self.samples),'steps':len(self.actions),'geometry':self.geometry}


def worker(conn,index,directory):
    from scripts import evaluate_c0_all as adapter
    from scripts.collect_candidate_utility_oracle import worker as original_worker
    base_make=adapter.make_env;box={}
    def make(*args,**kwargs):
        env=base_make(*args,**kwargs);rec=Recorder(env);box['rec']=rec
        original_init=env.set_init_state;original_step=env.step
        def initialize(value):
            rec.reset();result=original_init(value);rec.active=True;rec.sample();return result
        def step(action):
            result=original_step(action)
            if rec.active:rec.sample(action)
            return result
        env.set_init_state=initialize;env.step=step
        return env
    adapter.make_env=make
    class Connection:
        def recv(self):
            req=conn.recv();box['op']=req['operation']
            if req['operation']=='initialize':box['name']=req.pop('dense_name')
            return req
        def send(self,response):
            if response['ok'] and box['op']=='finish':
                try:response['value']['dense']=box['rec'].save(Path(directory)/'arrays'/f"{box['name']}.npz")
                except Exception:response={'ok':False,'error':traceback.format_exc()}
            conn.send(response)
        def close(self):conn.close()
    original_worker(Connection(),index,directory)


class Pool:
    def __init__(self):
        ctx=mp.get_context('spawn');self.connections=[];self.processes=[]
        for i in range(4):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,str(OUT)));p.start();b.close();self.connections.append(a);self.processes.append(p)
    def call(self,requests):
        for i,r in requests.items():self.connections[i].send(r)
        out={}
        for i in requests:
            if not self.connections[i].poll(300):raise TimeoutError(f'worker {i} timed out')
            r=self.connections[i].recv()
            if not r['ok']:raise RuntimeError(r['error'])
            out[i]=r['value']
        return out
    def close(self):
        for c in self.connections:
            try:c.send({'operation':'close'})
            except (OSError,BrokenPipeError):pass
        for p in self.processes:
            p.join(10)
            if p.is_alive():p.terminate();p.join(5)
        for c in self.connections:c.close()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'runtime.json').exists():
        raise RuntimeError('Existing run retained; no automatic rerun or budget reset')
    import fcntl
    lock=open(OUT/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads(PROTOCOL.read_text());ph=sha256(PROTOCOL)
    for path,d in protocol['frozen_files'].items():assert sha256(ROOT/path)==d,path
    roots={r['root_id']:r for r in json.loads((ROOT/'configs/candidate_utility_oracle_roots.json').read_text())['roots']}
    contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks']
    runtime={'status':'RUNNING','protocol_sha256':ph,'pid':os.getpid(),'completed_branches':0,'maximum_branches':13,
             'maximum_environment_steps':3900,'steps_accounted':0,'training':False,'started_epoch':time.time()}
    atomic_json(OUT/'runtime.json',runtime)
    initials={};ledger=[]
    for task in sorted({roots[k]['task_id'] for k in protocol['root_ids']}):
        path=ROOT/'third_party/LIBERO/libero/libero/init_files/libero_spatial'/Path(contracts[task]['bddl_path']).name.replace('.bddl','.pruned_init')
        initials[task],entry=selective_init_states(path);entry['task_id']=task;ledger.append(entry)
    atomic_json(OUT/'access_ledger.json',{'initials':ledger,'hdf5_opened':False,'new_scenes':False,'demo_access':False,'authorization':'User 启动: bounded dense relation recording preflight'})
    import torch
    from src.policies.c0_actor_policy import C0ActorPolicy
    torch.set_num_threads(1)
    policy=C0ActorPolicy(ROOT/'results/c0_v2/seed11/best.pt','cuda')
    pool=Pool();results=[]
    try:
        for ordinal,key in enumerate(protocol['root_ids']):
            root=roots[key];contract=contracts[root['task_id']]
            old=json.loads((ROOT/'results/candidate_utility_oracle/roots'/f'{key}.json').read_text())
            with np.load(ROOT/'results/candidate_utility_oracle/arrays'/f'{key}.npz',allow_pickle=False) as a:saved={k:a[k] for k in a.files}
            ids=[0,0,49,50] if ordinal==0 else [0,49,50]
            names=[f'{key}_c{cid:02d}_r{int(i==1 and ordinal==0)}' for i,cid in enumerate(ids)]
            runtime.update(current_root=key,phase='prefix',active_branches=len(ids));atomic_json(OUT/'runtime.json',runtime)
            requests={i:{**initialize_request(contract,root,initials[root['task_id']][root['init_index']],saved['prefix']),'dense_name':names[i]} for i in range(len(ids))}
            obs=pool.call(requests);runtime['steps_accounted']+=len(ids)*root['prefix_steps']
            reference={'agentview_image':saved['root_rgb'],'robot0_eef_pos':saved['root_proprio'][:3],
                       'robot0_gripper_qpos':saved['root_proprio'][3:5],'robot0_joint_pos':saved['root_proprio'][5:],'sim_state':saved['root_state']}
            guards=[identity(reference,obs[i]) for i in obs]
            assert all(g['pass'] for g in guards),'root identity mismatch'
            obs=pool.call({i:{'operation':'step','actions':saved['candidates'][cid]} for i,cid in enumerate(ids)})
            runtime['steps_accounted']+=8*len(ids)
            for i,cid in enumerate(ids):
                assert np.array_equal(obs[i]['agentview_image'],saved['h8_rgb'][cid]),'H8 RGB mismatch'
                d=obs[i]['diagnostic'];e=old['rows'][cid]['h8']
                assert all(d[k]==e[k] for k in ['on','grasp','target','both_on'])
                assert all(np.max(abs(np.array(d[k])-np.array(e[k])))<=1e-10 for k in ['m2','both_m2','bowl_positions','plate_position','reach'])
            step=root['prefix_steps']+8
            while step<300:
                actions=plan(policy,list(obs.values()),contract['instruction']);count=min(8,300-step)
                obs=pool.call({i:{'operation':'step','actions':actions[i,:count]} for i in obs})
                step+=count;runtime['steps_accounted']+=count*len(ids)
                assert runtime['steps_accounted']<=3900
                runtime.update(phase='continuation',episode_step=step,updated_epoch=time.time());atomic_json(OUT/'runtime.json',runtime)
                if step%32==0 or step==300:print(json.dumps({'root':key,'episode_step':step,'env_steps':runtime['steps_accounted']}),flush=True)
            final=pool.call({i:{'operation':'finish'} for i in obs})
            records=[]
            for i,cid in enumerate(ids):
                value=final[i];expected=old['rows'][cid]
                match={'root':guards[i]['pass'],'h8':True,
                       'all_actions_hash':value['applied_actions_sha256']==expected['applied_actions_sha256'],
                       'first_success':value['first_success_step']==expected['first_success_step'],
                       'success':value['success']==expected['success'],
                       'final_diagnostic':value['final']==expected['final']}
                records.append({'name':names[i],'root':root,'candidate_id':cid,'identity':match,**value})
                # Preserve any mismatch; never silently relabel the historical branch.
                atomic_json(OUT/'branches'/f'{names[i]}.json',records[-1])
            if ordinal==0:
                with np.load(ROOT/records[0]['dense']['arrays'],allow_pickle=False) as a, np.load(ROOT/records[1]['dense']['arrays'],allow_pickle=False) as b:
                    assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'A0 dense repeat mismatch'
                atomic_json(OUT/'repeat_check.json',{'pass':True,'all_dense_arrays_exact':True,'names':names[:2]})
            results.extend(records)
            assert all(all(r['identity'].values()) for r in records),'old branch identity/continuation mismatch; retained for diagnosis'
            runtime['completed_branches']=len(results);atomic_json(OUT/'runtime.json',runtime)
        assert runtime['steps_accounted']==3900 and len(results)==13
        atomic_json(OUT/'manifest.json',{'complete':True,'protocol_sha256':ph,'branches':results,'environment_steps':3900,'unique_branches':12,'repeat_branches':1})
        runtime.update(status='COMPLETE',phase='recording_complete',finished_epoch=time.time(),wall_seconds=time.time()-runtime['started_epoch'])
        atomic_json(OUT/'runtime.json',runtime)
        print(json.dumps(runtime),flush=True)
    except Exception:
        runtime.update(status='BLOCKED',error=traceback.format_exc(),scientific_conclusion='not assigned')
        atomic_json(OUT/'runtime.json',runtime);raise
    finally:pool.close()

if __name__=='__main__':main()
