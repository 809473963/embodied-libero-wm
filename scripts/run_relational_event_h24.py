#!/usr/bin/env python3
"""Causal state-stratified H24 fixed-plan collection; no model training."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,traceback,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from scripts.run_dense_relation_preflight import Recorder,Pool as OldPool
from scripts.collect_candidate_utility_oracle import initialize_request,atomic_npz
from src.probes.candidate_utility_oracle import atomic_json,sha256,selective_init_states,identity
from src.probes.relational_event_sampling import NAMES,candidates
from src.probes.relational_event_recording import validate
from src.probes.relational_event_labels import labels
OUT=ROOT/'results/relational_event_h24'

class AlignedRecorder(Recorder):
    def __init__(self,env,anchor,render):
        self.anchor=anchor;self.render=render;super().__init__(env)
    def reset(self):
        super().reset();self.images=[];self.proprios=[];self.natural_steps=[]
    def sample(self,action=None):
        super().sample(action);t=len(self.samples)-1
        if self.render and (t==0 or t>=self.anchor-10):
            from robosuite import macros
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
            sim=self.env.sim;robot=self.robot
            image=sim.render(camera_name='agentview',width=128,height=128,depth=False)
            self.images.append(np.ascontiguousarray(image[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]]))
            self.proprios.append(np.r_[sim.data.site_xpos[robot.eef_site_id],sim.data.qpos[robot._ref_gripper_joint_pos_indexes],sim.data.qpos[robot._ref_joint_pos_indexes]])
            self.natural_steps.append(t)
    def save(self,path):
        arrays={k:np.stack([r[k] for r in self.samples]) for k in self.samples[0]}
        arrays.update(actions=np.array(self.actions,np.float32),step=np.arange(len(self.samples),dtype=np.int32),
            contact_offsets=np.array(self.offsets,np.int64),contact_geom_pairs=np.array(self.pairs,np.int32).reshape(-1,2),
            contact_wrenches=np.array(self.wrenches,np.float64).reshape(-1,6),contact_frames=np.array(self.frames,np.float64).reshape(-1,9),contact_distances=np.array(self.distances,np.float64),
            natural_step=np.array(self.natural_steps,np.int32),rgb=np.array(self.images,np.uint8).reshape(-1,128,128,3),proprio=np.array(self.proprios,np.float64).reshape(-1,12))
        validate(arrays,self.geometry,self.anchor,self.render)
        atomic_npz(path,**arrays)
        return dict(arrays=str(path.relative_to(ROOT)),sha256=sha256(path),geometry=self.geometry,samples=len(self.samples))

def worker(conn,index,directory):
    from scripts import evaluate_c0_all as adapter
    from scripts.collect_candidate_utility_oracle import worker as original_worker
    base_make=adapter.make_env;box={}
    def make(*args,**kwargs):
        env=base_make(*args,**kwargs);rec=AlignedRecorder(env,box['anchor'],box['render']);box['rec']=rec
        oi,ostep=env.set_init_state,env.step
        def initialize(value):
            rec.reset();result=oi(value);rec.active=True;rec.sample();return result
        def step(action):
            result=ostep(action)
            if rec.active:rec.sample(action)
            return result
        env.set_init_state=initialize;env.step=step;return env
    adapter.make_env=make
    class Connection:
        def recv(self):
            req=conn.recv();box['op']=req['operation']
            if req['operation']=='initialize':
                for k in ('name','anchor','render'):box[k]=req.pop('capture_'+k)
            return req
        def send(self,response):
            if response['ok']:
                try:
                    rec=box.get('rec')
                    if box['op'] in ('initialize','step') and rec.render:
                        v=response['value'];t=int(v['global_step'])
                        j=rec.natural_steps.index(t)
                        assert np.array_equal(rec.images[j],v['agentview_image']),'direct render vs original snapshot mismatch'
                        p=np.r_[v['robot0_eef_pos'],v['robot0_gripper_qpos'],v['robot0_joint_pos']]
                        assert np.array_equal(rec.proprios[j],p),'proprio alignment mismatch'
                    if box['op']=='finish':response['value']['dense']=rec.save(Path(directory)/'arrays'/f"{box['name']}.npz")
                except Exception:response={'ok':False,'error':traceback.format_exc()}
            conn.send(response)
        def close(self):conn.close()
    original_worker(Connection(),index,directory)

class Pool(OldPool):
    def __init__(self):
        ctx=mp.get_context('spawn');self.connections=[];self.processes=[]
        for i in range(4):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,str(OUT)));p.start();b.close();self.connections.append(a);self.processes.append(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt retained; no automatic rerun')
    cfg=json.loads((ROOT/'configs/relational_event_h24_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    rt=dict(status='RUNNING',phase='preflight',pid=os.getpid(),completed_runs=0,maximum_runs=cfg['maximum_runs'],steps_accounted=0,maximum_environment_steps=cfg['maximum_environment_steps'],started_epoch=time.time(),training=False,protocol_sha256=sha256(ROOT/'configs/relational_event_h24_protocol.json'))
    atomic_json(OUT/'runtime.json',rt);pool=None
    try:
        roots={r['root_id']:r for r in json.loads((ROOT/'configs/candidate_utility_oracle_roots.json').read_text())['roots']}
        contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks']
        source={r['root_id']:r for r in json.loads((ROOT/'results/relational_coverage_expansion/coverage.json').read_text())['records'] if r['candidate_id']==0}
        initials={};ledger=[]
        for task in range(10):
            path=ROOT/'third_party/LIBERO/libero/libero/init_files/libero_spatial'/Path(contracts[task]['bddl_path']).name.replace('.bddl','.pruned_init')
            initials[task],entry=selective_init_states(path);ledger.append(entry)
        atomic_json(OUT/'access_ledger.json',dict(initials=ledger,used_init_indices=[31,34],parser_allowed_rows=[31,32,33,34],hdf5=False,new_scenes=False))
        pool=Pool();records=[];labelcfg=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters']
        for k,case in enumerate(cfg['cases']):
            key=case['root_id']
            root=roots[key];s=source[key]
            with np.load(ROOT/s['source'],allow_pickle=False) as f:old={x:f[x] for x in f.files}
            for anchor in [case['anchor']]:
                actionset=candidates(old['actions'][anchor:anchor+8],cfg['translation_action_offset'])
                specs=[(cid,f'{key}_t{anchor:03d}_v{cid:02d}',True,False) for cid in range(6)]
                if k==0:specs=[(0,f'preflight_{i}',i!=2,True) for i in range(3)]+specs
                # Preflight occupies its own batch and must pass before any new actions.
                batches=([specs[:3]]+[specs[i:i+4] for i in range(3,len(specs),4)]) if specs[0][3] else [specs[i:i+4] for i in range(0,len(specs),4)]
                reference=None
                for batch in batches:
                    steps=len(batch)*(anchor+24)
                    assert rt['steps_accounted']+steps<=rt['maximum_environment_steps']
                    # Charge whole submitted work before execution; failed partial work cannot reset budget.
                    rt['steps_accounted']+=steps;rt.update(current_root=key,anchor=anchor,phase='preflight' if batch[0][3] else 'collection',updated_epoch=time.time());atomic_json(OUT/'runtime.json',rt)
                    req={i:{**initialize_request(contracts[root['task_id']],root,initials[root['task_id']][root['init_index']],old['actions'][:anchor]),'capture_name':name,'capture_anchor':anchor,'capture_render':render} for i,(cid,name,render,pf) in enumerate(batch)}
                    observations=pool.call(req)
                    if reference is None:reference=observations[0]
                    for obs in observations.values():assert identity(reference,obs)['pass'],'root identity mismatch'
                    pool.call({i:dict(operation='step',actions=actionset[cid]) for i,(cid,_,_,_) in enumerate(batch)})
                    final=pool.call({i:dict(operation='finish') for i in range(len(batch))})
                    saved=[]
                    for i,(cid,name,render,pf) in enumerate(batch):
                        v=final[i];path=ROOT/v['dense']['arrays']
                        with np.load(path,allow_pickle=False) as f:a={x:f[x] for x in f.files}
                        for field in ('pose','qpos','qvel','pad_contact','gripper_contact','non_gripper_contact','on','grip_pos','grip_rot','grip_velocity'):
                            end=anchor+9 if cid==0 else anchor+1
                            np.testing.assert_array_equal(a[field][:end],old[field][:end],err_msg=field)
                        np.testing.assert_array_equal(a['actions'][:anchor],old['actions'][:anchor]);np.testing.assert_array_equal(a['actions'][anchor:],actionset[cid])
                        import hashlib
                        assert hashlib.sha256(np.ascontiguousarray(a['actions']).tobytes()).hexdigest()==v['applied_actions_sha256']
                        value=dict(name=name,root_id=key,anchor=anchor,strata=case['strata'],variant=cid,variant_name=NAMES[cid],preflight=pf,render=render,source=s['source'],source_sha256=s['source_sha256'],**v)
                        atomic_json(OUT/'branches'/f'{name}.json',value);saved.append(a)
                        if not pf:
                            d=[labels(a,labelcfg,j) for j in (0,1)]
                            atomic_npz(OUT/'labels'/f'{name}.npz',values=np.stack([x['values'] for x in d],axis=1),step=a['step'])
                            records.append(value)
                    if batch[0][3]:
                        for other in saved[1:]:
                            for field in saved[0]:
                                if field not in ('rgb','proprio','natural_step'):np.testing.assert_array_equal(saved[0][field],other[field])
                        np.testing.assert_array_equal(saved[0]['rgb'],saved[1]['rgb']);np.testing.assert_array_equal(saved[0]['proprio'],saved[1]['proprio'])
                        atomic_json(OUT/'preflight.json',dict(pass_all=True,three_A0_physics_exact=True,render_on_off_physics_exact=True,RGB_repeat_exact=True,snapshot_RGB_proprio_alignment=True,old_A0_prefix_and_H8_exact=True))
                    rt['completed_runs']+=len(batch);atomic_json(OUT/'runtime.json',rt)
                    atomic_json(OUT/'manifest.json',dict(complete=False,records=records))
                    print(json.dumps(dict(root=key,anchor=anchor,runs=rt['completed_runs'],steps=rt['steps_accounted'])),flush=True)
        assert rt['completed_runs']==cfg['maximum_runs'] and rt['steps_accounted']==cfg['maximum_environment_steps'] and len(records)==6*len(cfg['cases'])
        atomic_json(OUT/'manifest.json',dict(complete=True,protocol_sha256=rt['protocol_sha256'],records=records))
        rt.update(status='COMPLETE',phase='collection_complete',finished_epoch=time.time(),wall_seconds=time.time()-rt['started_epoch']);atomic_json(OUT/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc(),scientific_conclusion='not assigned');atomic_json(OUT/'runtime.json',rt);raise
    finally:
        if pool:pool.close()

if __name__=='__main__':main()
