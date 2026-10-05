#!/usr/bin/env python3
"""Replay registered prefix + H8 only; never run a new continuation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / '.libero'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from src.probes.candidate_utility_oracle import array_hash, identity, selective_init_states, sha256
from src.probes.continuation_utility_observability import (
    DIRECTORY, ENTITIES, IDS, STATE_DIM, validate_protocol, validate_raw, write_json, write_npz,
)


def init_worker():
    log = open(DIRECTORY / f'worker_{os.getpid()}.log', 'a', buffering=1)
    sys.stdout = log
    sys.stderr = log


def entity_bindings(inner):
    bodies, geoms = [], []
    for name in ENTITIES:
        model = inner.get_object(name)
        if model is None and name == 'main_table':
            # LIBERO creates its workspace in the arena, outside fixtures_dict.
            bodies.append('table')
            geoms.append(['table_collision'])
        elif model is not None:
            bodies.append(model.root_body)
            geoms.append(model.contact_geoms)
        else:
            raise ValueError(f'frozen scene entity missing: {name}')
    return bodies, geoms


def replay(request):
    from scripts.evaluate_c0_all import make_env, refresh_policy_observation
    from src.probes.robust_grounding import BOWLS, compatibility, resolve_scores
    from src.probes.semantic_teacher_clean import controlled_and_reach
    env = None
    try:
        root, contract = request['root'], request['contract']
        env = make_env(ROOT / contract['bddl_path'])
        env.seed(root['seed'])
        env.reset()
        env.set_init_state(request['init'])
        inner = env.env
        for name, observable in inner._observables.items():
            if name.endswith('_image'):
                observable.set_active(False)
        scores = np.asarray([compatibility(inner, contract, b, 'G2')[0] for b in BOWLS])
        status, ref, gap = resolve_scores(scores, 0., 20.)
        if status != 'UNIQUE' or ref != 0:
            raise ValueError('initial persistent referent differs from frozen contract')
        robot = inner.robots[0]
        site = robot.gripper.important_sites['grip_site']
        body_names, contact_geoms = entity_bindings(inner)
        owners = {}
        for index, geoms in enumerate(contact_geoms):
            for geom in geoms:
                owners.setdefault(env.sim.model.geom_name2id(geom), set()).add(index)
        gripper_ids = {env.sim.model.geom_name2id(g) for g in robot.gripper.contact_geoms}
        velocity_indices = list(robot._ref_joint_vel_indexes) + list(robot._ref_gripper_joint_vel_indexes)
        nq, nv = len(env.sim.data.qpos), len(env.sim.data.qvel)
        if max(nq, nv) > 64 or len(velocity_indices) != 9:
            raise ValueError('frozen state schema cannot represent this environment')

        def padded(value):
            x, mask = np.zeros(64), np.zeros(64)
            x[:len(value)] = value
            mask[:len(value)] = 1
            return np.concatenate((x, mask))

        def sample(step, recent_actions):
            observation = refresh_policy_observation(env)
            natural = {k: np.ascontiguousarray(observation[k]).copy() for k in
                       ('agentview_image', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_joint_pos')}
            proprio = np.concatenate([natural[k].reshape(-1) for k in
                                      ('robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_joint_pos')])
            poses = []
            for name, body in zip(ENTITIES, body_names):
                body_id = env.sim.model.body_name2id(body)
                poses.extend((env.sim.data.body_xpos[body_id], env.sim.data.body_xquat[body_id],
                              env.sim.data.get_body_xvelp(body), env.sim.data.get_body_xvelr(body)))
            contacts, grip_contacts = np.zeros((8, 8)), np.zeros(8)
            for index in range(env.sim.data.ncon):
                c = env.sim.data.contact[index]
                left, right = owners.get(c.geom1, ()), owners.get(c.geom2, ())
                for i in left:
                    for j in right:
                        contacts[i, j] = contacts[j, i] = 1
                if c.geom1 in gripper_ids:
                    for j in right:
                        grip_contacts[j] = 1
                if c.geom2 in gripper_ids:
                    for i in left:
                        grip_contacts[i] = 1
            state = np.concatenate((*poses, contacts.reshape(-1), grip_contacts, np.ones(8), proprio,
                                    env.sim.data.get_site_xmat(site).reshape(-1),
                                    env.sim.data.get_site_xvelp(site), env.sim.data.get_site_xvelr(site),
                                    env.sim.data.qvel[velocity_indices], padded(env.sim.data.qpos),
                                    padded(env.sim.data.qvel))).astype(np.float32)
            if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
                raise ValueError('privileged state shape/finiteness gate failed')
            fields = {'rgb': natural['agentview_image'], 'proprio': proprio.astype(np.float32),
                      'state': state, 'actions': np.asarray(recent_actions, np.float32),
                      'time': np.asarray(step, np.int32)}
            return fields, {**natural, 'sim_state': env.sim.get_state().flatten().copy()}

        def diagnostic():
            plate = np.asarray(inner.object_states_dict['plate_1'].get_geom_state()['pos'], float)
            bowls = [np.asarray(inner.object_states_dict[b].get_geom_state()['pos'], float) for b in BOWLS]
            goals = [bool(inner._eval_predicate(['on', b, 'plate_1'])) for b in BOWLS]
            margins = [float(min((.03 - np.linalg.norm(p[:2] - plate[:2])) / .03,
                                 (p[2] - plate[2]) / .03)) for p in bowls]
            grasp, reach = controlled_and_reach(inner, BOWLS[ref])
            return {'on': goals[ref], 'm2': margins[ref], 'bowl_positions': [p.tolist() for p in bowls],
                    'plate_position': plate.tolist(), 'both_on': goals, 'both_m2': margins,
                    'grasp': grasp, 'reach': reach, 'target': BOWLS[ref]}

        frames, branch = [], []
        frame, reference = sample(0, np.zeros(7))
        frames.append(frame)
        prefix = request['prefix']
        for start in range(0, len(prefix), 8):
            chunk = prefix[start:start + 8]
            for action in chunk:
                env.step(action)
            frame, reference = sample(start + 8, chunk.mean(0))
            frames.append(frame)
        # This additional root observation matches the original collector's read.
        _, reference = sample(len(prefix), frames[-1]['actions'])
        candidate = request['candidate']
        for start in range(0, 8, 2):
            chunk = candidate[start:start + 2]
            for action in chunk:
                env.step(action)
            frame, _ = sample(len(prefix) + start + 2, chunk.mean(0))
            branch.append(frame)
        arrays = {f'prefix_{k}': np.stack([x[k] for x in frames]) for k in frames[0]}
        arrays.update({f'branch_{k}': np.stack([x[k] for x in branch]) for k in branch[0]})
        return {'arrays': arrays, 'reference': reference, 'h8': diagnostic(),
                'grounding': {'status': status, 'ref': ref, 'gap': float(gap)}, 'nq': nq, 'nv': nv}
    finally:
        if env is not None:
            env.close()


def verify(result, old, saved, identifier):
    proprio = saved['root_proprio']
    reference = {'agentview_image': saved['root_rgb'], 'robot0_eef_pos': proprio[:3],
                 'robot0_gripper_qpos': proprio[3:5], 'robot0_joint_pos': proprio[5:],
                 'sim_state': saved['root_state']}
    check = identity(reference, result['reference'])
    image_match = np.array_equal(result['arrays']['branch_rgb'][-1], saved['h8_rgb'][identifier])
    expected, actual = old['rows'][identifier]['h8'], result['h8']
    discrete = all(actual[k] == expected[k] for k in ('on', 'both_on', 'grasp', 'target'))
    errors = {key: float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
              for key in ('m2', 'both_m2', 'bowl_positions', 'plate_position', 'reach')}
    passed = check['pass'] and image_match and discrete and max(errors.values()) <= 1e-10
    guard = {'candidate_id': identifier, 'root_identity': check, 'h8_rgb_exact': bool(image_match),
             'h8_discrete_exact': discrete, 'h8_numeric_errors': errors, 'pass': bool(passed)}
    if not passed:
        raise ValueError('label reuse identity gate failed: ' + json.dumps(guard))
    return guard


def load_source(root):
    directory = ROOT / 'results/candidate_utility_oracle'
    old = json.loads((directory / 'roots' / f"{root['root_id']}.json").read_text())
    path = directory / 'arrays' / f"{root['root_id']}.npz"
    if not old['complete'] or old['root'] != root or sha256(path) != old['arrays_sha256']:
        raise ValueError('Stage1 input hash/completion mismatch')
    with np.load(path, allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    if array_hash(saved['prefix']) != old['prefix_sha256'] or array_hash(saved['candidates']) != old['candidates_sha256']:
        raise ValueError('saved action identity mismatch')
    return old, saved


def assemble(results):
    # Prefix is identical across branches and stored only once.
    first = results[0]['arrays']
    for r in results[1:]:
        if any(not np.array_equal(first[k], r['arrays'][k]) for k in first if k.startswith('prefix_')):
            raise ValueError('prefix history differs across same-root candidate replays')
    arrays = {k: v for k, v in first.items() if k.startswith('prefix_')}
    arrays.update({k: np.stack([r['arrays'][k] for r in results]) for k in first if k.startswith('branch_')})
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'collect'), required=True)
    args = parser.parse_args()
    protocol, digest = validate_protocol()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    roots = json.loads((ROOT / 'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    contracts = json.loads((ROOT / 'configs/task_grounding_contract.json').read_text())['tasks']
    initials, ledger = {}, []
    for task, contract in enumerate(contracts):
        path = ROOT / 'third_party/LIBERO/libero/libero/init_files/libero_spatial' / Path(contract['bddl_path']).name.replace('.bddl', '.pruned_init')
        initials[task], entry = selective_init_states(path)
        ledger.append(entry)
    write_json(DIRECTORY / 'access_ledger.json', {'initial_states': ledger, 'hdf5_opened': False,
               'demo_numeric_or_metadata_access': False, 'new_continuation_steps': 0,
               'authorized_scope': 'User 开始: Stage2A prefix/H8 replay and six readouts'})
    started = time.monotonic()
    steps = 0
    try:
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
            def run(root, saved, ids):
                nonlocal steps
                requests = [dict(root=root, contract=contracts[root['task_id']], init=initials[root['task_id']][root['init_index']],
                                 prefix=saved['prefix'], candidate=saved['candidates'][i]) for i in ids]
                futures = [pool.submit(replay, x) for x in requests]
                values = [f.result(timeout=600) for f in futures]
                steps += len(ids) * (root['prefix_steps'] + 8)
                return values

            if args.mode == 'preflight':
                checks = []
                for ordinal in (0, 28):
                    root = roots[ordinal]
                    old, saved = load_source(root)
                    ids = [0, 0, 49]
                    values = run(root, saved, ids)
                    guards = [verify(r, old, saved, i) for r, i in zip(values, ids)]
                    if any(not np.array_equal(values[0]['arrays'][k], values[1]['arrays'][k]) for k in values[0]['arrays']):
                        raise ValueError('supplemented A0 repeat features differ')
                    checks.append({'root_id': root['root_id'], 'guards': guards, 'repeat_exact': True,
                                   'nq': values[0]['nq'], 'nv': values[0]['nv']})
                write_json(DIRECTORY / 'preflight.json', {'protocol_sha256': digest, 'pass': True,
                           'checks': checks, 'simulation_steps': steps, 'wall_seconds': time.monotonic() - started})
                print(json.dumps({'preflight': 'PASS', 'steps': steps, 'wall_seconds': time.monotonic() - started}), flush=True)
                return
            preflight = json.loads((DIRECTORY / 'preflight.json').read_text())
            if not preflight['pass'] or preflight['protocol_sha256'] != digest:
                raise ValueError('matching preflight required')
            completed = []
            for root in roots:
                key = root['root_id']
                metadata = DIRECTORY / 'roots' / f'{key}.json'
                artifact = DIRECTORY / 'arrays' / f'{key}.npz'
                if metadata.exists():
                    record = json.loads(metadata.read_text())
                    if record['protocol_sha256'] != digest or sha256(artifact) != record['arrays_sha256'] or not record['complete']:
                        raise ValueError('resume completed root hash mismatch')
                    completed.append(key)
                    continue
                old, saved = load_source(root)
                groups, guards = [], []
                for offset in range(0, 32, 8):
                    gp = DIRECTORY / 'groups' / f'{key}_{offset:02d}.json'
                    ap = gp.with_suffix('.npz')
                    ids = IDS[offset:offset + 8]
                    if gp.exists():
                        record = json.loads(gp.read_text())
                        if record['protocol_sha256'] != digest or record['candidate_ids'] != ids or sha256(ap) != record['arrays_sha256']:
                            raise ValueError('resume group hash/protocol mismatch')
                        with np.load(ap, allow_pickle=False) as a:
                            arrays = {k: a[k] for k in a.files}
                        group_guards = record['guards']
                    else:
                        values = run(root, saved, ids)
                        group_guards = [verify(v, old, saved, i) for v, i in zip(values, ids)]
                        arrays = assemble(values)
                        write_npz(ap, **arrays)
                        write_json(gp, {'protocol_sha256': digest, 'candidate_ids': ids, 'guards': group_guards,
                                       'arrays_sha256': sha256(ap)})
                    groups.append(arrays)
                    guards.extend(group_guards)
                    print(json.dumps({'root': key, 'branches_complete': offset + 8, 'roots_complete': len(completed),
                                      'simulation_steps_current_run': steps, 'elapsed_seconds': time.monotonic() - started}), flush=True)
                arrays = {k: v for k, v in groups[0].items() if k.startswith('prefix_')}
                arrays.update({k: np.concatenate([g[k] for g in groups]) for k in groups[0] if k.startswith('branch_')})
                for group in groups[1:]:
                    if any(not np.array_equal(arrays[k], group[k]) for k in arrays if k.startswith('prefix_')):
                        raise ValueError('prefix mismatch across stored groups')
                validate_raw(arrays)
                write_npz(artifact, **arrays)
                write_json(metadata, {'protocol_sha256': digest, 'root': root, 'candidate_ids': IDS,
                           'source_root_sha256': sha256(ROOT / 'results/candidate_utility_oracle/roots' / f'{key}.json'),
                           'arrays_sha256': sha256(artifact), 'guards': guards, 'complete': True})
                completed.append(key)
                write_json(DIRECTORY / 'manifest.json', {'protocol_sha256': digest, 'completed_roots': completed,
                           'complete': len(completed) == 40, 'simulation_steps_current_run': steps,
                           'wall_seconds_current_run': time.monotonic() - started})
                print(json.dumps({'root_complete': key, 'completed': len(completed), 'total': 40}), flush=True)
    except Exception:
        write_json(DIRECTORY / 'collection_blocker.json', {'protocol_sha256': digest,
                   'error': traceback.format_exc(), 'scientific_verdict': 'NOT_ASSIGNED'})
        raise


if __name__ == '__main__':
    main()
