#!/usr/bin/env python3
"""Read-only audit of registered Stage1/2A artifacts; no simulator or model imports.
Writes only results/relational_state_audit. Outcome-blind review selection precedes
all outcome association. Diagnostic thresholds are not stable-event labels.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/relational_state_audit'
S1 = ROOT / 'results/candidate_utility_oracle'
S2 = ROOT / 'results/continuation_utility_observability'
IDS = [0] + list(range(1, 23)) + list(range(49, 58))
ENTITIES = ['target', 'other_bowl', 'plate', 'cookies', 'ramekin', 'table', 'cabinet', 'stove']
SCHEMA = {
    'version': 'sampled-relations-v1', 'date': '2026-10-06',
    'status': 'descriptive pilot; not stable-event supervision',
    'units': {'position': 'metres', 'velocity': 'metres/second', 'time': 'control steps', 'rotation': 'radians'},
    'nominal_control_hz': 20, 'hz_source': 'make_env default OffScreenRenderEnv control_freq=20; source verified, no runtime replay',
    'candidate_samples': [0, 2, 4, 6, 8], 'entities': ENTITIES,
    'input_precision': 'float32 state; no 1e-10 geometry claims',
    'numerical_geometry_tolerance_m': 1e-6,
    'relative_geometry': 'world-frame body position and velocity differences; target relative to grip_site also computed',
    'relative_rotation': 'wxyz converted to matrices; geodesic rotation, no symmetry quotient; not semantic bowl orientation accuracy',
    'gripper_reference': 'eef_pos is grip_site position in installed robosuite single_arm.py',
    'events': {
        'sampled_target_contact_onset': 'any 0->1 transition at adjacent saved samples; onset interval censored',
        'first_sampled_contact_from_uncontacted_root': 'root has no any-gripper target contact, then a saved contact occurs; not verified grasp',
        'sampled_target_contact_loss': 'any 1->0 transition at adjacent saved samples; not necessarily failure',
        'all_samples_target_contact': 'any-gripper contact true at all five samples; NOT continuous or bilateral grasp',
        'sampled_plate_contact': 'target-plate contact at any saved sample; NOT support or stable placement',
        'h8_robot_contact_no_entity_contact': 'any-gripper target contact at H8 and no target contact with other seven tracked entities; untracked contacts not ruled out',
        'sampled_detach_and_rise_5mm': 'target initially contacts a tracked nonrobot entity; H8 no tracked nonrobot contact, gripper contact, z increase>0.005m; weak evidence only',
        'h8_bilateral_contact': 'existing Stage1 exact H8 _check_grasp; instant only',
    },
    'sensitivity': {'rise_m': [.001, .005, .01], 'gripper_relative_translation_excursion_m': [.001, .005, .01]},
    'strong_events': {k: 'unavailable: missing dense bilateral/support/holding verification; do not infer negative' for k in ['stable_grasp', 'lift_and_hold', 'released_stable_placement']},
    'review_rule': 'up to10 positive and10 negative per listed relation; root-diverse round-robin then candidate order; no Y in selection',
    'label_policy': 'missing evidence=unknown; sampled point equality does not certify intersample continuity',
    'feature_forbidden': ['Y', 'first_success_step', 'final300', 'oracle_winner'],
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rotation(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))


def relations(states):
    """states [candidate, sample, 476]; derive sampled physical relations only."""
    v = np.asarray(states, float)
    bodies = v[..., :104].reshape(v.shape[:-1] + (8,13))
    pos, vel = bodies[..., :3], bodies[..., 7:10]
    contacts = v[..., 104:168].reshape(v.shape[:-1]+(8,8)).astype(bool)
    grip = v[..., 168:176].astype(bool)
    R = v[..., 196:205].reshape(v.shape[:-1]+(3,3))
    local_pos = np.einsum('...ji,...j->...i', R, pos[...,0,:]-v[...,184:187])
    local_R = np.swapaxes(R,-1,-2) @ rotation(bodies[...,0,3:7])
    delta_R = np.swapaxes(local_R[:, :1],-1,-2) @ local_R
    rotation_change = np.arccos(np.clip((np.trace(delta_R,axis1=-2,axis2=-1)-1)/2,-1,1))
    tracked_other_contact = contacts[...,0,1:].any(-1)
    dz = pos[:,-1,0,2] - pos[:,0,0,2]
    onset = ((~grip[:,:-1,0]) & grip[:,1:,0]).any(1)
    loss = (grip[:,:-1,0] & ~grip[:,1:,0]).any(1)
    free = grip[:,-1,0] & ~tracked_other_contact[:,-1]
    excursion = np.linalg.norm(local_pos-local_pos[:,:1],axis=-1).max(1)
    return {'positions':pos, 'velocities':vel, 'contacts':contacts, 'gripper':grip,
            'target_plate_position':pos[...,0,:]-pos[...,2,:],
            'target_plate_velocity':vel[...,0,:]-vel[...,2,:],
            'target_gripper_local_position':local_pos, 'gripper_relative_rotation_change':rotation_change,
            'dz':dz, 'local_excursion':excursion,
            'events':{'sampled_target_contact_onset':onset,
                      'first_sampled_contact_from_uncontacted_root':(~grip[:,0,0]) & onset,
                      'sampled_target_contact_loss':loss,
                      'all_samples_target_contact':grip[...,0].all(1),
                      'sampled_plate_contact':contacts[...,0,2].any(1),
                      'h8_robot_contact_no_entity_contact':free,
                      'sampled_detach_and_rise_5mm':free & tracked_other_contact[:,0] & (dz>.005)},
            'initial_contacts':tracked_other_contact[:,0]}


def diverse(items, n=10):
    groups = defaultdict(list)
    for r in items: groups[r['root_id']].append(r)
    result=[]
    for depth in range(max([len(x) for x in groups.values()] or [0])):
        for key in sorted(groups):
            if depth < len(groups[key]): result.append(groups[key][depth])
            if len(result)>=n: return result
    return result


def coverage(rows, event):
    selected=[r for r in rows if r['events'][event]]
    return {'branches':len(selected),'roots':len({r['root_id'] for r in selected}),
            'tasks':sorted({r['task_id'] for r in selected})}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    def register(path):
        p=Path(path); key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        sources[key]=digest(p); return p
    def read(path): return json.loads(register(path).read_text())
    protocol=read(ROOT/'configs/continuation_utility_observability_protocol.json')
    protocol_sha=digest(ROOT/'configs/continuation_utility_observability_protocol.json')
    for path, expected in protocol['frozen_files'].items():
        assert digest(register(ROOT/path)) == expected, path
    register(Path(__file__))
    register(ROOT/'docs/action_conditioned_relational_dynamics_plan.md')
    for path in ['scripts/collect_continuation_observability.py','scripts/collect_candidate_utility_oracle.py',
                 'scripts/evaluate_c0_all.py','third_party/LIBERO/libero/libero/envs/env_wrapper.py',
                 '.venv/lib/python3.10/site-packages/robosuite/robots/single_arm.py']:
        register(ROOT/path)
    split=read(ROOT/'configs/continuation_utility_observability_split.json')
    registered=read(ROOT/'configs/candidate_utility_oracle_roots.json')['roots']
    records=[]; checks=[]; old_sources={}; images={}; prefix_points=0; prefix_contact_transitions=[]
    for root in registered:
        key=root['root_id']
        meta=read(S2/'roots'/f'{key}.json')
        source_path=S1/'roots'/f'{key}.json'
        old=read(source_path); old_sources[key]=old
        rp=register(S2/'arrays'/f'{key}.npz'); op=register(S1/'arrays'/f'{key}.npz')
        assert meta['complete'] and old['complete'] and meta['root']==old['root']==root
        assert meta['protocol_sha256']==protocol_sha
        assert digest(rp)==meta['arrays_sha256'] and digest(op)==old['arrays_sha256']
        assert digest(source_path)==meta['source_root_sha256']
        assert meta['candidate_ids']==IDS==[g['candidate_id'] for g in meta['guards']]
        assert all(g['pass'] for g in meta['guards'])
        assert [r['candidate_id'] for r in old['rows']]==list(range(64))
        with np.load(rp,allow_pickle=False) as f: raw={k:f[k] for k in f.files}
        with np.load(op,allow_pickle=False) as f: saved={k:f[k] for k in f.files}
        assert array_hash(saved['prefix'])==old['prefix_sha256']
        assert array_hash(saved['candidates'])==old['candidates_sha256']
        assert saved['candidates'].shape==(64,8,7) and np.isfinite(saved['candidates']).all()
        assert np.max(np.abs(saved['candidates']))<=1
        assert set(raw)=={f'{p}_{k}' for p in ['prefix','branch'] for k in ['rgb','state','proprio','actions','time']}
        assert raw['prefix_time'].tolist()==list(range(0,root['prefix_steps']+1,8))
        assert np.array_equal(raw['branch_time'], np.tile(root['prefix_steps']+np.array([2,4,6,8]),(32,1)))
        for phase in ['prefix','branch']:
            leading=raw[f'{phase}_time'].shape
            for name, tail in [('rgb',(128,128,3)),('state',(476,)),('proprio',(12,)),('actions',(7,))]:
                assert raw[f'{phase}_{name}'].shape==leading+tail
                assert np.isfinite(raw[f'{phase}_{name}']).all()
            assert raw[f'{phase}_state'].dtype==np.float32
            assert np.array_equal(raw[f'{phase}_state'][...,184:196],raw[f'{phase}_proprio'])
        assert np.array_equal(raw['prefix_rgb'][-1],saved['root_rgb'])
        assert np.array_equal(raw['branch_rgb'][:,-1],saved['h8_rgb'][IDS])
        assert np.array_equal(raw['prefix_proprio'][-1],saved['root_proprio'].astype(np.float32))
        assert np.array_equal(raw['branch_actions'],saved['candidates'][IDS].reshape(32,4,2,7).mean(2))
        assert not raw['prefix_actions'][0].any()
        if root['prefix_steps']:
            assert np.array_equal(raw['prefix_actions'][1:],saved['prefix'].reshape(-1,8,7).mean(1))
        allstate=np.concatenate([raw['prefix_state'],raw['branch_state'].reshape(-1,476)])
        c=allstate[:,104:168].reshape(-1,8,8)
        assert np.isin(allstate[:,104:184],[0,1]).all() and np.array_equal(c,c.transpose(0,2,1))
        assert np.all(allstate[:,176:184]==1)
        for start in [284,412]: assert np.isin(allstate[:,start:start+64],[0,1]).all()
        for value,mask in [(220,284),(348,412)]:
            assert not (allstate[:,value:value+64]*(1-allstate[:,mask:mask+64])).any()
        quat=allstate[:,:104].reshape(-1,8,13)[...,3:7]
        assert np.max(abs(np.linalg.norm(quat,axis=-1)-1))<1e-4
        rot=allstate[:,196:205].reshape(-1,3,3)
        assert np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3)))<1e-5
        seq=np.concatenate([np.broadcast_to(raw['prefix_state'][-1],(32,1,476)),raw['branch_state']],axis=1)
        rel=relations(seq)
        split_name='fit' if root['init_index']<34 else 'development'
        assert key in split[split_name+'_roots']
        max_pos_error=0.
        for j,cid in enumerate(IDS):
            diag=old['rows'][cid]['h8']
            err=float(np.max(np.abs(rel['positions'][j,-1,:3]-np.array(diag['bowl_positions']+[diag['plate_position']]))))
            assert err<1e-6
            max_pos_error=max(max_pos_error,err)
            assert diag['target']=='akita_black_bowl_1'
            # Contact API (check_contact) versus raw MuJoCo contact list are distinct diagnostics;
            # retain disagreements rather than silently forcing agreement.
            pairxy=np.linalg.norm(rel['target_plate_position'][j,-1,:2])
            near=bool(abs(pairxy-.03)<1e-6 or abs(rel['target_plate_position'][j,-1,2])<1e-6)
            derived_on=bool(rel['contacts'][j,-1,0,2] and pairxy<.03 and rel['target_plate_position'][j,-1,2]>=0)
            events={k:bool(v[j]) for k,v in rel['events'].items()}
            events['h8_bilateral_contact']=bool(diag['grasp'])
            record={'root_id':key,'task_id':root['task_id'],'candidate_id':cid,'split':split_name,
                    'times':(root['prefix_steps']+np.array([0,2,4,6,8])).tolist(),
                    'events':events,'target_contact_samples':rel['gripper'][j,:,0].tolist(),
                    'other_bowl_contact_samples':rel['gripper'][j,:,1].tolist(),
                    'target_entity_contact_samples':rel['contacts'][j,:,0,:].tolist(),
                    'target_plate_relative_position_m':rel['target_plate_position'][j].tolist(),
                    'target_plate_relative_velocity_m_s':rel['target_plate_velocity'][j].tolist(),
                    'target_gripper_local_position_m':rel['target_gripper_local_position'][j].tolist(),
                    'target_gripper_rotation_change_rad':rel['gripper_relative_rotation_change'][j].tolist(),
                    'dz_m':float(rel['dz'][j]),'local_translation_excursion_m':float(rel['local_excursion'][j]),
                    'initial_tracked_contact':bool(rel['initial_contacts'][j]),
                    'official_h8_on':bool(diag['on']),'derived_h8_on':derived_on,'on_boundary':near,
                    'on_mismatch':derived_on!=bool(diag['on']),
                    'bilateral_without_raw_any_contact':bool(diag['grasp'] and not rel['gripper'][j,-1,0]),
                    'ordered_action_max_abs_diff_from_A0':float(np.max(abs(saved['candidates'][cid]-saved['candidates'][0])))}
            records.append(record)
            images[(key,cid)]=np.concatenate([raw['prefix_rgb'][-1:],[*raw['branch_rgb'][j]]])
        prefix_points+=len(raw['prefix_time'])
        pc=raw['prefix_state'][:,168].astype(bool)
        for ix in np.flatnonzero((~pc[:-1]) & pc[1:]):
            prefix_contact_transitions.append({'root_id':key,'interval_steps':[int(raw['prefix_time'][ix]),int(raw['prefix_time'][ix+1])],
                                               'evidence':'sampled any-contact onset only; not dense bilateral grasp'})
        checks.append({'root_id':key,'pass':True,'branches':32,'raw_sha256':digest(rp),'h8_position_maxabs_m':max_pos_error})
        print('verified',key,flush=True)
    # Freeze schema and choose inspectable evidence WITHOUT success labels.
    write('relation_schema.json',SCHEMA)
    review_sets={}; picked={}
    for event in ['sampled_target_contact_onset','sampled_target_contact_loss','sampled_detach_and_rise_5mm','sampled_plate_contact','h8_bilateral_contact']:
        for flag in [True,False]:
            selected=diverse([r for r in records if r['events'][event]==flag])
            name=f'{event}:{flag}'
            review_sets[name]=[(r['root_id'],r['candidate_id']) for r in selected]
            for r in selected: picked[(r['root_id'],r['candidate_id'])]=r
    review={'schema_sha256':digest(OUT/'relation_schema.json'),'selection_uses_Y':False,
            'sets':review_sets,'unique_cases':list(picked.values()),
            'review_status':'contact sheets generated; human-style visual inspection is supplemental, not physical truth'}
    write('review_cases.json',review)
    # Contact sheets of root and all four saved frames, no Y displayed.
    from PIL import Image, ImageDraw
    chosen=list(picked.values()); sheet_files=[]
    for offset in range(0,len(chosen),8):
        batch=chosen[offset:offset+8]; im=Image.new('RGB',(940,len(batch)*174),'#f4f2e9');d=ImageDraw.Draw(im)
        for row,r in enumerate(batch):
            y=row*174; key=(r['root_id'],r['candidate_id'])
            d.text((8,y+4),f"{offset+row:02d} {key[0]} #{key[1]}  target={''.join(str(int(v)) for v in r['target_contact_samples'])} dz={r['dz_m']*1000:.2f}mm",fill='black')
            for f,pixels in enumerate(images[key]):
                im.paste(Image.fromarray(pixels),(8+f*150,y+22));d.text((8+f*150,y+153),f"step {r['times'][f]}",fill='black')
            d.text((760,y+30),f"H8 grasp {r['events']['h8_bilateral_contact']}\nH8 On {r['official_h8_on']}\nplate-contact\n{r['events']['sampled_plate_contact']}",fill='black')
        name=f'review_sheet_{offset//8:02d}.png';im.save(OUT/name);sheet_files.append(name)
    review['contact_sheets']=sheet_files;write('review_cases.json',review)
    write('sampled_relations.json',{'schema_sha256':digest(OUT/'relation_schema.json'),'rows':records})
    # Only now associate Stage1 continuation outcomes, always in a separate artifact.
    labelled=[]
    for r in records:
        old=old_sources[r['root_id']];row=old['rows'][r['candidate_id']]
        labelled.append({**r,'eligible':old['reference']['first_success_step'] is None,
                         'Y':bool(row['success']),'A0_Y':bool(old['rows'][0]['success']),
                         'final_on':bool(row['final']['on']),'final_grasp':bool(row['final']['grasp']),
                         'success_by_h8':row['first_success_step'] is not None and row['first_success_step']<=r['times'][-1]})
    eligible=[r for r in labelled if r['eligible']]
    events=list(records[0]['events'])
    availability={'status':'COMPLETE','roots':40,'branches':len(records),'eligible_roots':len({r['root_id'] for r in eligible}),
                  'eligible_branches':len(eligible),'prefix_saved_points':prefix_points,
                  'prefix_sampled_contact_onset_intervals':prefix_contact_transitions,
                  'branch_saved_points':len(records)*4,'branch_root_copies_not_independent':len(records),
                  'unique_saved_observations_by_path':prefix_points+len(records)*4,
                  'source_checks':checks,'event_coverage_all':{e:coverage(records,e) for e in events},
                  'event_coverage_eligible':{e:coverage(eligible,e) for e in events},
                  'strong_events_unavailable':{k:len(records) for k in SCHEMA['strong_events']},
                  'h8_on_mismatches':sum(r['on_mismatch'] for r in records),
                  'bilateral_without_raw_any_contact':sum(r['bilateral_without_raw_any_contact'] for r in records),
                  'other_bowl_any_gripper_contact_branches':sum(any(r['other_bowl_contact_samples']) for r in records),
                  'sensitivity':{},'simulation_steps':0,'new_fits':0,'new_continuations':0}
    for threshold in [.001,.005,.01]:
        availability['sensitivity'][str(threshold)]={
            'sampled_detach_and_rise':sum(r['initial_tracked_contact'] and r['events']['h8_robot_contact_no_entity_contact'] and r['dz_m']>threshold for r in records),
            'all_sample_contact_and_local_excursion_below':sum(r['events']['all_samples_target_contact'] and r['local_translation_excursion_m']<threshold for r in records)}
    association={'interpretation':'descriptive post-action pilot; no fitted chooser, no causal attribution',
                 'by_event':{},'roots':[]}
    for e in events:
        association['by_event'][e]={}
        for value in [False,True]:
            subset=[r for r in eligible if r['events'][e]==value]
            association['by_event'][e][str(value)]={'branches':len(subset),'successes':sum(r['Y'] for r in subset),
                'roots':len({r['root_id'] for r in subset}),'tasks':sorted({r['task_id'] for r in subset}),
                'not_independent_samples':True}
    for key in [r['root_id'] for r in registered]:
        rows=[r for r in labelled if r['root_id']==key];a0=rows[0]
        signatures=defaultdict(list)
        for r in rows:
            signature=json.dumps([r['target_contact_samples'],r['target_entity_contact_samples'],r['events']['h8_bilateral_contact'],r['official_h8_on']])
            signatures[signature].append(r)
        conflicts=[g for g in signatures.values() if len({r['Y'] for r in g})>1]
        opposite=[]
        for g in conflicts:
            neg=next(r for r in g if not r['Y']);pos=next(r for r in g if r['Y'])
            opposite.append({'failed_candidate':neg['candidate_id'],'successful_candidate':pos['candidate_id'],
                             'max_gripper_relative_position_difference_m':float(np.max(abs(np.array(neg['target_gripper_local_position_m'])-np.array(pos['target_gripper_local_position_m'])))),
                             'max_target_plate_position_difference_m':float(np.max(abs(np.array(neg['target_plate_relative_position_m'])-np.array(pos['target_plate_relative_position_m'])))),
                             'warning':'only discrete sampled signature equal; continuous geometry/actions may differ'})
        association['roots'].append({'root_id':key,'task_id':a0['task_id'],'eligible':a0['eligible'],'split':a0['split'],
            'A0_Y':a0['Y'],'any_Y':any(r['Y'] for r in rows),'success_candidates':sum(r['Y'] for r in rows),
            'rescue_opportunity':not a0['Y'] and any(r['Y'] for r in rows),
            'events_positive':{e:sum(r['events'][e] for r in rows) for e in events},
            'discrete_signature_count':len(signatures),'mixed_Y_signature_count':len(conflicts),
            'counterexamples':opposite,
            'target_plate_H8_span_m':np.ptp(np.array([r['target_plate_relative_position_m'][-1] for r in rows]),axis=0).tolist()})
    association['final_diagnostic']={'eligible_success_branches':sum(r['Y'] for r in eligible),
        'ever_success_but_final_on_false':sum(r['Y'] and not r['final_on'] for r in eligible),
        'ever_success_and_final_on_and_grasp':sum(r['Y'] and r['final_on'] and r['final_grasp'] for r in eligible),
        'success_by_h8':sum(r['success_by_h8'] for r in eligible),
        'warning':'final bilateral contact != all gripper contact; no proof of released stability, even when final_grasp is false'}
    write('availability.json',availability);write('outcome_association.json',association)
    assert sum(r['rescue_opportunity'] and r['eligible'] for r in association['roots'])==7
    assert sum(r['rescue_opportunity'] and r['eligible'] and r['split']=='development' for r in association['roots'])==3
    # Verify every source remained unchanged during this audit.
    assert all(digest(ROOT/p if not Path(p).is_absolute() else Path(p))==h for p,h in sources.items())
    write('source_manifest.json',{'complete':True,'date':'2026-10-06','source_files':[{'path':p,'sha256':h} for p,h in sources.items()],
        'scope':'registered old artifacts only; no HDF5/init containers; frozen C0 checkpoint binary hashed only, no model deserialization/inference',
        'source_immutability_rechecked':True,'schema_sha256':digest(OUT/'relation_schema.json'),
        'review_selection_sha256':digest(OUT/'review_cases.json'),
        'output_sha256':{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name not in {'source_manifest.json','review_notes.json','validation.json'} and not p.name.endswith('.part')},
        'new_training':False,'simulator_execution':False})
    print(json.dumps({'roots':40,'branches':len(records),'review_cases':len(chosen),'sheets':sheet_files,
                      'event_coverage':availability['event_coverage_all'],'final':association['final_diagnostic']},ensure_ascii=False))

if __name__=='__main__': main()
