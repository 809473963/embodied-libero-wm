#!/usr/bin/env python3
"""Fresh replay, frozen N64 consequences, and fixed-C0 continuation.

GPU inference has a fixed padded batch of eight; CPU simulators run in eight
isolated processes. All privileged values are diagnostics, never policy inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / '.libero'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT))

from src.probes.candidate_utility_oracle import (
    array_hash, atomic_json, identity, selective_init_states, sha256, t1_select,
)


def worker(connection, index: int, directory: str):
    log = open(Path(directory) / f'worker_{index}.log', 'a', buffering=1)
    sys.stdout = log
    sys.stderr = log
    from scripts.evaluate_c0_all import make_env, refresh_policy_observation
    from src.probes.robust_grounding import BOWLS, compatibility, resolve_scores
    from src.probes.semantic_teacher_clean import controlled_and_reach

    env = None
    state = {}

    def diagnostic():
        inner = env.env
        plate = np.asarray(inner.object_states_dict['plate_1'].get_geom_state()['pos'], float)
        bowls = [np.asarray(inner.object_states_dict[b].get_geom_state()['pos'], float) for b in BOWLS]
        goals = [bool(inner._eval_predicate(['on', b, 'plate_1'])) for b in BOWLS]
        margins = [float(min((.03-np.linalg.norm(p[:2]-plate[:2]))/.03, (p[2]-plate[2])/.03)) for p in bowls]
        grasp, reach = controlled_and_reach(inner, state['referent'])
        ref = BOWLS.index(state['referent'])
        return {'on': goals[ref], 'm2': margins[ref], 'bowl_positions': [p.tolist() for p in bowls],
                'plate_position': plate.tolist(), 'both_on': goals, 'both_m2': margins,
                'grasp': grasp, 'reach': reach, 'target': state['referent']}

    def snapshot():
        observation = refresh_policy_observation(env)
        required = ('agentview_image', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_joint_pos')
        result = {key: np.ascontiguousarray(observation[key]).copy() for key in required}
        result['sim_state'] = env.sim.get_state().flatten().copy()
        result['diagnostic'] = diagnostic()
        result['grounding'] = state['grounding']
        result['first_success_step'] = state['first_success_step']
        result['global_step'] = state['step']
        return result

    def step_actions(actions):
        for action in actions:
            action = np.asarray(action, np.float32)
            if action.shape != (7,) or not np.isfinite(action).all() or np.any(abs(action)>1):
                raise ValueError('invalid clipped action')
            _, _, done, _ = env.step(action)
            state['step'] += 1
            success = bool(env.check_success())
            if bool(done) != success:
                raise RuntimeError('installed done behavior differs from frozen success-only return')
            if success and state['first_success_step'] is None:
                state['first_success_step'] = state['step']
            state['actions_hash'].update(action.tobytes())
        return snapshot()

    try:
        while True:
            request = connection.recv()
            operation = request['operation']
            if operation == 'close':
                break
            try:
                if operation == 'initialize':
                    if env is not None:
                        env.close()
                    env = make_env(Path(request['bddl']))
                    env.seed(int(request['seed']))
                    env.reset()
                    env.set_init_state(request['init_state'])
                    for name, observable in env.env._observables.items():
                        if name.endswith('_image'):
                            observable.set_active(False)
                    scores = np.asarray([compatibility(env.env, request['contract'], b, 'G2')[0] for b in BOWLS])
                    status, ref, gap = resolve_scores(scores, 0.0, 20.0)
                    if status != 'UNIQUE' or ref != 0:
                        raise RuntimeError(f'correct-instruction G2 differs from official goal: {status}, ref={ref}')
                    initial_success = bool(env.check_success())
                    state = {'step': 0, 'referent': BOWLS[ref], 'actions_hash': hashlib.sha256(),
                             'first_success_step': 0 if initial_success else None,
                             'grounding': {'status': status, 'ref': ref, 'gap': float(gap), 'scores': scores.tolist()}}
                    prefix = request.get('prefix', np.empty((0,7), np.float32))
                    for start in range(0, len(prefix), 8):
                        step_actions(prefix[start:start+8])
                    response = snapshot()
                elif operation == 'step':
                    response = step_actions(request['actions'])
                elif operation == 'finish':
                    response = {'first_success_step': state['first_success_step'],
                                'success': state['first_success_step'] is not None,
                                'final_step': state['step'], 'final': diagnostic(),
                                'applied_actions_sha256': state['actions_hash'].hexdigest()}
                else:
                    raise ValueError('unsupported operation')
                connection.send({'ok': True, 'value': response})
            except Exception:
                connection.send({'ok': False, 'error': traceback.format_exc()})
    finally:
        if env is not None:
            env.close()
        connection.close()
        log.close()


class Workers:
    def __init__(self, directory: Path):
        context = mp.get_context('spawn')
        self.connections = []
        self.processes = []
        for index in range(8):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child,index,str(directory)))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def call(self, requests):
        for index, request in requests.items():
            self.connections[index].send(request)
        results = {}
        for index in requests:
            response = self.connections[index].recv()
            if not response['ok']:
                raise RuntimeError(response['error'])
            results[index] = response['value']
        return results

    def close(self):
        for connection in self.connections:
            try:
                connection.send({'operation':'close'})
            except (OSError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def validate_protocol():
    path = ROOT / 'configs/candidate_utility_oracle_protocol.json'
    protocol = json.loads(path.read_text())
    relative = str(path.relative_to(ROOT))
    committed = subprocess.check_output(['git','show',f'HEAD:{relative}'], cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError('protocol must match a committed pre-execution freeze')
    for relative, digest in protocol['frozen_files'].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f'frozen input or implementation changed: {relative}')
    if protocol['scope']['init_indices'] != [31,32,33,34] or protocol['inference']['batch_size'] != 8:
        raise RuntimeError('unexpected scope or inference shape')
    return protocol, sha256(path)


def plan(policy, observations, language):
    from scripts.evaluate_c0_all import policy_observation
    values = [policy_observation(x) for x in observations]
    padded = values + [values[-1]]*(8-len(values))
    if not 1 <= len(values) <= 8:
        raise ValueError('one to eight observations required')
    return policy.plan_batch(padded,[language]*8)[:len(values)]


def initialize_request(contract, root, init, prefix=None):
    return {'operation':'initialize', 'bddl':str(ROOT/contract['bddl_path']),
            'seed':root['seed'], 'init_state':init, 'contract':contract,
            'prefix':np.empty((0,7),np.float32) if prefix is None else prefix}


def reference_root(pool, policy, contract, root, init):
    snapshot = pool.call({0:initialize_request(contract,root,init)})[0]
    chunks = []
    for _ in range(root['prefix_steps']//8):
        chunk = plan(policy,[snapshot],contract['instruction'])[0]
        chunks.append(chunk)
        snapshot = pool.call({0:{'operation':'step','actions':chunk}})[0]
    prefix = np.concatenate(chunks) if chunks else np.empty((0,7),np.float32)
    default = plan(policy,[snapshot],contract['instruction'])[0]
    return snapshot, prefix, default


def run_batch(pool, policy, contract, root, init, prefix, reference, candidates, ids, max_steps):
    observations = pool.call({slot:initialize_request(contract,root,init,prefix) for slot in range(len(ids))})
    identities = [identity(reference,observations[slot]) for slot in observations]
    if not all(x['pass'] for x in identities):
        raise RuntimeError('fresh-prefix identity gate failed: '+json.dumps(identities))
    observations = pool.call({slot:{'operation':'step','actions':candidates[identifier]} for slot,identifier in enumerate(ids)})
    h8 = [observations[slot]['diagnostic'] for slot in observations]
    h8_images = [observations[slot]['agentview_image'] for slot in observations]
    step = root['prefix_steps']+8
    while step < max_steps:
        chunks = plan(policy,list(observations.values()),contract['instruction'])
        count = min(8,max_steps-step)
        observations = pool.call({slot:{'operation':'step','actions':chunks[slot,:count]} for slot in observations})
        step += count
    final = pool.call({slot:{'operation':'finish'} for slot in observations})
    return [{'candidate_id':identifier, 'identity':identities[slot], 'h8':h8[slot], **final[slot]}
            for slot,identifier in enumerate(ids)], np.stack(h8_images)


def preflight(pool,policy,contracts,roots,initials,protocol_hash,directory):
    started = time.monotonic()
    rows = []
    for index in (0,28):
        root = roots[index]
        contract = contracts[root['task_id']]
        init = initials[root['task_id']][root['init_index']]
        reference,prefix,default = reference_root(pool,policy,contract,root,init)
        # Three A0 copies at distinct batch slots test replay and full continuation.
        outcomes,_ = run_batch(pool,policy,contract,root,init,prefix,reference,np.repeat(default[None],64,0),[0,1,2],300)
        hashes = [x['applied_actions_sha256'] for x in outcomes]
        labels = [(x['success'],x['first_success_step']) for x in outcomes]
        if len(set(hashes))!=1 or len(set(labels))!=1:
            raise RuntimeError('A0 continuation depends on branch slot or stochastic replay')
        rows.append({'root_id':root['root_id'], 'prefix_steps':root['prefix_steps'],
                     'identities':[x['identity'] for x in outcomes], 'a0_action_hash_equal':True,
                     'a0_outcome_equal':True, 'steps_per_branch':300,
                     'observed_done_contract':'returned done equals check_success; underlying horizon is 1000; continue to300'})
    result={'protocol_sha256':protocol_hash,'pass':True,'roots':rows,
            'inference_batch_size':8,'wall_seconds':time.monotonic()-started,
            'scientific_selection_performed':False}
    atomic_json(directory/'preflight.json',result)
    print(json.dumps({'preflight':'PASS','wall_seconds':result['wall_seconds']}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('preflight','collect'),required=True)
    args=parser.parse_args()
    protocol,protocol_hash=validate_protocol()
    directory=ROOT/'results/candidate_utility_oracle'
    directory.mkdir(parents=True,exist_ok=True)
    roots=json.loads((ROOT/'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks']
    initials={}
    ledger=[]
    for task,contract in enumerate(contracts):
        path=ROOT/'third_party/LIBERO/libero/libero/init_files/libero_spatial'/Path(contract['bddl_path']).name.replace('.bddl','.pruned_init')
        initials[task],record=selective_init_states(path)
        record['task_id']=task
        ledger.append(record)
    atomic_json(directory/'access_ledger.json',{'initial_states':ledger,'hdf5_opened':False,
        'demo_numeric_or_metadata_access':False,'forbidden_init_numeric_access':0,
        'diagnostic_scope_authorized_by':'user: 开始执行; isolated candidate utility oracle'})
    import torch
    from src.policies.c0_actor_policy import C0ActorPolicy
    from src.probes.expanded_candidate_contract_restart import build_contract_candidates
    from src.probes.planning_architecture import generate_unit_noise
    torch.set_num_threads(1)
    policy=C0ActorPolicy(ROOT/'results/c0_v2/seed11/best.pt','cuda')
    interface=json.loads((ROOT/'results/expanded_candidate_contract_restart/selected_interface.json').read_text())
    pool=Workers(directory)
    started=time.monotonic()
    try:
        if args.mode=='preflight':
            preflight(pool,policy,contracts,roots,initials,protocol_hash,directory)
            return
        check=json.loads((directory/'preflight.json').read_text())
        if not check['pass'] or check['protocol_sha256']!=protocol_hash:
            raise RuntimeError('committed protocol needs matching passing preflight')
        completed=[]
        for root in roots:
            root_path=directory/'roots'/f"{root['root_id']}.json"
            if root_path.exists():
                old=json.loads(root_path.read_text())
                if old['protocol_sha256']!=protocol_hash or not old['complete']:
                    raise RuntimeError('cannot overwrite a mismatched or incomplete root')
                completed.append(root['root_id'])
                continue
            contract=contracts[root['task_id']]
            init=initials[root['task_id']][root['init_index']]
            reference,prefix,default=reference_root(pool,policy,contract,root,init)
            unit=generate_unit_noise(1,24,int(interface['seed_base'])+root['root_ordinal'])
            _,candidates,category,forms=build_contract_candidates(default[None],unit,48,
                tuple(interface['structured_gripper_forms']),np.asarray(protocol['noise_std'],np.float32),2.5)
            candidates=candidates[0]
            artifact=directory/'arrays'/f"{root['root_id']}.npz"
            atomic_npz(artifact,prefix=prefix,candidates=candidates,root_rgb=reference['agentview_image'],
                       root_proprio=np.concatenate([reference[k].reshape(-1) for k in ('robot0_eef_pos','robot0_gripper_qpos','robot0_joint_pos')]),
                       root_state=reference['sim_state'])
            rows=[]
            images=[]
            for start in range(0,64,8):
                group_path=directory/'groups'/f"{root['root_id']}_{start:02d}.json"
                image_path=group_path.with_suffix('.npz')
                if group_path.exists():
                    group=json.loads(group_path.read_text())
                    if group['protocol_sha256']!=protocol_hash or group['candidates_sha256']!=array_hash(candidates):
                        raise RuntimeError('resume candidate/protocol mismatch')
                    if sha256(image_path)!=group['image_sha256']:
                        raise RuntimeError('resume image artifact hash mismatch')
                    rows.extend(group['rows'])
                    with np.load(image_path,allow_pickle=False) as a:
                        images.extend(a['h8_rgb'])
                    continue
                result,h8_rgb=run_batch(pool,policy,contract,root,init,prefix,reference,candidates,list(range(start,start+8)),300)
                atomic_npz(image_path,h8_rgb=h8_rgb)
                atomic_json(group_path,{'protocol_sha256':protocol_hash,'candidates_sha256':array_hash(candidates),
                                       'image_sha256':sha256(image_path),'rows':result})
                rows.extend(result)
                images.extend(h8_rgb)
                print(json.dumps({'root':root['root_id'],'candidate_branches_complete':start+8,
                                  'roots_complete':len(completed),'elapsed_seconds':time.monotonic()-started}),flush=True)
            repeats,_=run_batch(pool,policy,contract,root,init,prefix,reference,candidates,[0,0],300)
            base=rows[0]
            repetition_pass=all(x['applied_actions_sha256']==base['applied_actions_sha256'] and
                                x['first_success_step']==base['first_success_step'] and x['success']==base['success'] for x in repeats)
            if not repetition_pass:
                raise RuntimeError('A0 repeat failure; scientific collection blocked')
            selected=t1_select(np.asarray([x['h8']['on'] for x in rows]),
                               np.asarray([x['h8']['m2'] for x in rows]),protocol['epsilon_goal'])
            verify_ids=set()
            if selected != 0:
                verify_ids.add(selected)
            if not base['success'] and any(x['success'] for x in rows):
                verify_ids.add(next(x['candidate_id'] for x in rows if x['success']))
            intervention_repeats=[]
            for candidate_id in sorted(verify_ids):
                verified,_=run_batch(pool,policy,contract,root,init,prefix,reference,candidates,[candidate_id,candidate_id],300)
                expected=rows[candidate_id]
                if not all(x['applied_actions_sha256']==expected['applied_actions_sha256'] and
                           x['first_success_step']==expected['first_success_step'] and x['success']==expected['success'] for x in verified):
                    raise RuntimeError('selected/rescue candidate repeat failure; no scientific positive verdict')
                intervention_repeats.extend(verified)
            atomic_npz(artifact,prefix=prefix,candidates=candidates,root_rgb=reference['agentview_image'],
                       root_proprio=np.concatenate([reference[k].reshape(-1) for k in ('robot0_eef_pos','robot0_gripper_qpos','robot0_joint_pos')]),
                       root_state=reference['sim_state'],h8_rgb=np.stack(images))
            record={'schema_version':1,'protocol_sha256':protocol_hash,'root':root,
                    'reference':{'diagnostic':reference['diagnostic'],'grounding':reference['grounding'],
                                 'first_success_step':reference['first_success_step'],'rgb_sha256':array_hash(reference['agentview_image'])},
                    'prefix_sha256':array_hash(prefix),'candidates_sha256':array_hash(candidates),
                    'candidate_forms':forms,'arrays_sha256':sha256(artifact),'rows':rows,
                    'a0_repeat_pass':True,'a0_repeats':repeats,'t1_selected_id':selected,
                    'intervention_repeat_pass':True,'intervention_repeats':intervention_repeats,'complete':True}
            atomic_json(root_path,record)
            completed.append(root['root_id'])
            atomic_json(directory/'manifest.json',{'protocol_sha256':protocol_hash,'completed_roots':completed,
                'complete':len(completed)==40,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'elapsed_seconds_current_invocation':time.monotonic()-started})
            print(json.dumps({'root_complete':root['root_id'],'completed':len(completed),'total':40}),flush=True)
    except Exception:
        atomic_json(directory/'collection_blocker.json',{'protocol_sha256':protocol_hash,'error':traceback.format_exc(),
                   'scientific_verdict':'NOT_ASSIGNED','elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        pool.close()


if __name__=='__main__':
    main()
