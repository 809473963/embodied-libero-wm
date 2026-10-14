#!/usr/bin/env python3
"""Generate development seeds only; single-attempt resets and fixed-action replay."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,random,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_new_scene import canonical_layout,fingerprint,distance,near,probe_actions
from scripts.collect_candidate_utility_oracle import atomic_npz
OUT=ROOT/'results/relational_new_scene_preflight'

def run_scene(scene,contract,rt):
    from scripts.evaluate_c0_all import make_env,refresh_policy_observation
    from scripts.run_dense_relation_preflight import Recorder
    from src.probes.robust_grounding import BOWLS,compatibility,resolve_scores
    from robosuite.utils.errors import RandomizationError
    generated=None;reference=None;rows=[]
    for mode in ('generate','same_seed','restore_saved_state'):
        env=None
        try:
            rt['reset_attempts']+=1;assert rt['reset_attempts']<=60
            atomic_json(OUT/'runtime.json',rt)
            # Constructors can consume RNG before the explicit reset; seed both boundaries.
            random.seed(scene['seed']);np.random.seed(scene['seed'])
            try:
                env=make_env(ROOT/contract['bddl_path']);env.seed(scene['seed'])
                env.env.reset()  # Explicit one attempt; wrapper reset's unbounded retry loop is avoided.
            except RandomizationError:
                if mode!='generate':raise RuntimeError('Same seed reset changed validity')
                return dict(**scene,status='REJECTED_RESET_RANDOMIZATION',error=traceback.format_exc(),replacement=False)
            if mode=='restore_saved_state':state=generated.copy()
            else:state=env.sim.get_state().flatten().copy()
            env.set_init_state(state)  # Canonical starting observation/state contract, with fresh controller context.
            if mode=='generate':generated=env.sim.get_state().flatten().copy()
            obs=refresh_policy_observation(env);rec=Recorder(env);rec.sample()
            scores=np.array([compatibility(env.env,contract,b,'G2')[0] for b in BOWLS]);status,ref,gap=resolve_scores(scores,0.,20.)
            initial=dict(pose=rec.samples[0]['pose'].copy(),rgb=obs['agentview_image'].copy(),
                proprio=np.r_[obs['robot0_eef_pos'],obs['robot0_gripper_qpos'],obs['robot0_joint_pos']],
                sim_state=env.sim.get_state().flatten().copy())
            key=f"task{scene['task_id']:02d}_seed{scene['seed']}"
            if mode=='generate':
                atomic_npz(OUT/'initials'/f'{key}.npz',**initial)
                grounding=dict(status=status,ref=int(ref) if ref is not None else None,gap=float(gap),scores=scores.tolist())
                if status!='UNIQUE' or ref!=0:
                    return dict(**scene,status='REJECTED_GROUNDING',grounding=grounding,initial=str((OUT/'initials'/f'{key}.npz').relative_to(ROOT)),replacement=False)
            else:
                for field in initial:np.testing.assert_array_equal(initial[field],reference['initial'][field],err_msg=f'{mode}/{field}')
            actions=probe_actions();rt['steps_submitted']+=24;assert rt['steps_submitted']<=1440
            atomic_json(OUT/'runtime.json',rt)
            for action in actions:
                env.step(action);rt['steps_completed']+=1;rec.sample(action)
            final=refresh_policy_observation(env)
            arrays={k:np.stack([r[k] for r in rec.samples]) for k in rec.samples[0]}
            arrays.update(actions=actions,final_rgb=final['agentview_image'],initial_rgb=initial['rgb'],initial_proprio=initial['proprio'],initial_state=initial['sim_state'])
            assert all(np.isfinite(v).all() for v in arrays.values())
            path=OUT/'arrays'/f'{key}_{mode}.npz';atomic_npz(path,**arrays)
            if mode=='generate':reference=dict(initial=initial,arrays=arrays)
            else:
                for field in arrays:np.testing.assert_array_equal(arrays[field],reference['arrays'][field],err_msg=f'{mode}/{field}')
            rows.append(dict(mode=mode,path=str(path.relative_to(ROOT)),sha256=sha256(path),exact_replay=True))
        finally:
            if env is not None:env.close()
    path=OUT/'initials'/f'{key}.npz'
    return dict(**scene,status='REPLAY_VALID',grounding=grounding,initial=str(path.relative_to(ROOT)),initial_sha256=sha256(path),layout_fingerprint=fingerprint(reference['initial']['pose']),runs=rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt preserved; no automatic retry')
    cfg=json.loads((ROOT/'configs/relational_new_scene_preflight_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    assert len(cfg['scenes'])==20 and all(s['split']=='development' for s in cfg['scenes'])
    rt=dict(status='RUNNING',phase='generation_and_replay',pid=os.getpid(),started_epoch=time.time(),completed_scenes=0,maximum_scenes=20,reset_attempts=0,maximum_reset_attempts=60,steps_submitted=0,steps_completed=0,maximum_control_steps=1440,confirmation_seeds_executed=0,training=False)
    atomic_json(OUT/'runtime.json',rt)
    try:
        contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks'];records=[]
        for scene in cfg['scenes']:
            rt.update(current_task=scene['task_id'],current_seed=scene['seed']);atomic_json(OUT/'runtime.json',rt)
            r=run_scene(scene,contracts[scene['task_id']],rt);records.append(r)
            atomic_json(OUT/'scenes'/f"task{scene['task_id']:02d}_seed{scene['seed']}.json",r)
            rt['completed_scenes']+=1;atomic_json(OUT/'runtime.json',rt);atomic_json(OUT/'manifest.json',dict(complete=False,records=records))
            print(json.dumps(dict(completed=rt['completed_scenes'],status=r['status'],steps=rt['steps_completed'])),flush=True)
        old=[]
        for r in json.loads((ROOT/'results/relational_coverage_expansion/coverage.json').read_text())['records']:
            if r['candidate_id']!=0:continue
            assert sha256(ROOT/r['source'])==r['source_sha256']
            with np.load(ROOT/r['source'],allow_pickle=False) as a:old.append((r['root_id'],a['pose'][0]))
        accepted=[];comparison=[]
        for r in records:
            if r['status']!='REPLAY_VALID':continue
            with np.load(ROOT/r['initial'],allow_pickle=False) as a:pose=a['pose']
            matches=[name for name,p in old if near(pose,p)]
            newmatches=[name for name,p in accepted if near(pose,p)]
            closest=min(((name,distance(pose,p)) for name,p in old),key=lambda z:z[1]['max_position_m'])
            eligible=not matches and not newmatches
            comparison.append(dict(seed=r['seed'],task=r['task_id'],eligible_for_development=eligible,old_near_matches=matches,new_near_matches=newmatches,nearest_old=closest))
            if eligible:accepted.append((str(r['seed']),pose.copy()))
        result=dict(status='GENERATION_REPLAY_AND_LAYOUT_CHECK_COMPLETE',records=records,comparisons=comparison,
            eligible_development_scenes=len(accepted),old_reference_scenes=len(old),confirmation_executed=False,
            globally_unseen_C0_training_layout_proven=False,scope='Layout novelty only against the 40 audited old scenes; technical probe is not task-success evaluation.')
        atomic_json(OUT/'analysis.json',result);atomic_json(OUT/'manifest.json',dict(complete=True,records=records))
        rt.update(status='COMPLETE',phase='awaiting_scientific_review',finished_epoch=time.time(),wall_seconds=time.time()-rt['started_epoch']);atomic_json(OUT/'runtime.json',rt)
        print(json.dumps(dict(eligible=len(accepted),seconds=rt['wall_seconds'])),flush=True)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc(),scientific_conclusion='not assigned');atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
