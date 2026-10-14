#!/usr/bin/env python3
"""Analyze frozen dense-preflight recordings; never imports simulator/policy."""
from pathlib import Path
import json
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/dense_relation_preflight'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(mask):
    """Inclusive sample indices. Duration=end-start, not number of samples."""
    p=np.r_[False,np.asarray(mask,bool),False].astype(int)
    return [[int(a),int(b-1)] for a,b in zip(np.flatnonzero(np.diff(p)==1),np.flatnonzero(np.diff(p)==-1))]


def qualifying_windows(mask,local,steps,excursion):
    out=[]
    for t in range(len(mask)-steps):
        win=slice(t,t+steps+1)
        if np.all(mask[win]) and np.max(np.linalg.norm(local[win]-local[t],axis=1))<=excursion:
            out.append(t)
    return out


def derive(a,window=10,tolerance=.005):
    pos=a['pose'][:,0,:3];grip=a['grip_pos'];rot=a['grip_rot']
    local=np.einsum('tji,tj->ti',rot,pos-grip)
    bilateral=a['pad_contact'][:,:,0].all(1)
    anygrip=a['gripper_contact'][:,0]
    nongrip=a['non_gripper_contact'][:,0]
    on=a['on'][:,0]
    speed=np.linalg.norm(a['pose'][:,0,7:10],axis=1)
    hold=qualifying_windows(bilateral,local,window,tolerance)
    lift=[]
    for t in hold:
        prior=np.flatnonzero(nongrip[:t])
        if len(prior) and not nongrip[t:t+window+1].any():
            # relative to the last observed non-gripper contact, not initial world height
            if np.min(pos[t:t+window+1,2])-pos[prior[-1],2]>.005:lift.append(t)
    place_mask=on & ~anygrip & nongrip & (speed<.02)
    placed=[t for t in range(len(on)-window) if place_mask[t:t+window+1].all() and anygrip[:t].any()]
    return {'contact_intervals':intervals(anygrip),'bilateral_intervals':intervals(bilateral),
            'on_intervals':intervals(on),'released_on_low_speed_intervals':intervals(place_mask),
            'hold_candidate_window_starts':hold,'lift_hold_candidate_window_starts':lift,
            'released_placement_candidate_window_starts':placed,
            'first_target_contact_step':next((int(t) for t in np.flatnonzero(anygrip)),None),
            'first_bilateral_step':next((int(t) for t in np.flatnonzero(bilateral)),None),
            'local_position':local}


def validate_contacts(a,geometry):
    owners={int(k):v for k,v in geometry['geom_entity_owner'].items()}
    grip=set(geometry['gripper_ids']);pads=[set(geometry['left_pad_ids']),set(geometry['right_pad_ids'])]
    rebuilt=np.zeros_like(a['pad_contact']);anycontact=np.zeros_like(a['gripper_contact'])
    nongrip=np.zeros_like(a['non_gripper_contact']);entities=np.zeros_like(a['contacts'])
    for t in range(301):
        for g1,g2 in a['contact_geom_pairs'][a['contact_offsets'][t]:a['contact_offsets'][t+1]]:
            i,j=owners.get(int(g1)),owners.get(int(g2))
            if i is not None and j is not None:entities[t,i,j]=entities[t,j,i]=True
            for target,other in [(i,int(g2)),(j,int(g1))]:
                if target is not None:
                    anycontact[t,target]|=other in grip
                    for side in range(2):rebuilt[t,side,target]|=other in pads[side]
                    nongrip[t,target]|=other not in grip and owners.get(other)!=target
    assert np.array_equal(rebuilt,a['pad_contact'])
    assert np.array_equal(anycontact,a['gripper_contact'])
    assert np.array_equal(nongrip,a['non_gripper_contact'])
    assert np.array_equal(entities,a['contacts'])
    assert np.array_equal(a['grasp'],rebuilt[:,:,0].all(1))


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    protocol=json.loads((ROOT/'configs/dense_relation_preflight_protocol.json').read_text())
    assert manifest['complete'] and manifest['protocol_sha256']==sha(ROOT/'configs/dense_relation_preflight_protocol.json')
    for file,d in protocol['frozen_files'].items():assert sha(ROOT/file)==d,file
    results=[];repeat=[]
    for record in manifest['branches']:
        assert all(record['identity'].values())
        path=ROOT/record['dense']['arrays'];assert sha(path)==record['dense']['sha256']
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        assert np.array_equal(a['step'],np.arange(301)) and a['actions'].shape==(300,7)
        assert a['contact_offsets'].shape==(302,) and np.all(np.diff(a['contact_offsets'])>=0)
        assert a['contact_offsets'][0]==0 and a['contact_offsets'][-1]==len(a['contact_geom_pairs'])
        for k in ['contact_geom_pairs','contact_wrenches','contact_frames','contact_distances']:
            assert len(a[k])==a['contact_offsets'][-1]
        assert all(np.isfinite(v).all() for v in a.values())
        validate_contacts(a,record['dense']['geometry'])
        action_sha=hashlib.sha256(np.ascontiguousarray(a['actions']).tobytes()).hexdigest()
        assert action_sha==record['applied_actions_sha256']
        success=np.flatnonzero(a['success']);first=int(success[0]) if len(success) else None
        assert first==record['first_success_step']
        cid=record['candidate_id'];root=record['root'];start=root['prefix_steps']
        with np.load(ROOT/'results/candidate_utility_oracle/arrays'/f"{root['root_id']}.npz",allow_pickle=False) as old:
            assert np.array_equal(a['actions'][:start],old['prefix'])
            assert np.array_equal(a['actions'][start:start+8],old['candidates'][cid])
        d=derive(a);d.pop('local_position')
        grid={}
        for w in [4,10,20]:
            for tol in [.001,.005,.01]:
                z=derive(a,w,tol)
                grid[f'{w}steps_{tol}m']={k:len(z[k]) for k in ['hold_candidate_window_starts','lift_hold_candidate_window_starts','released_placement_candidate_window_starts']}
        d.update(name=record['name'],root_id=root['root_id'],candidate_id=cid,root_step=start,
                 success=record['success'],first_success_step=first,
                 control_hz=record['dense']['geometry']['control_freq'],
                 positive_normal_force_contacts=int((a['contact_wrenches'][:,0]>1e-8).sum()),
                 contact_record_count=len(a['contact_geom_pairs']),sensitivity_grid=grid,
                 post_root_hold_candidates=sum(t>=start for t in d['hold_candidate_window_starts']),
                 post_root_lift_hold_candidates=sum(t>=start for t in d['lift_hold_candidate_window_starts']),
                 post_root_release_candidates=sum(t>=start for t in d['released_placement_candidate_window_starts']),
                 initial_contact=bool(a['gripper_contact'][0,0]),
                 phase_of_first_contact='none' if d['first_target_contact_step'] is None else ('prefix' if d['first_target_contact_step']<start else 'H8' if d['first_target_contact_step']<=start+8 else 'continuation'))
        (repeat if record['name'].endswith('_r1') else results).append(d)
    # independently repeat-check all recorded fields, not only the old success bit
    copies=manifest['branches'][:2]
    with np.load(ROOT/copies[0]['dense']['arrays'],allow_pickle=False) as a,np.load(ROOT/copies[1]['dense']['arrays'],allow_pickle=False) as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    summary={'status':'TECHNICAL_RECORDING_VALID','protocol_sha256':manifest['protocol_sha256'],
        'runs':13,'unique_branches':12,'roots':4,'environment_steps':3900,'dense_control_samples':3913,
        'unique_branch_samples':3612,'prefix_shared_samples_not_independent':True,
        'old_branch_identity_all_pass':True,'A0_dense_repeat_exact':True,'contact_reconstruction_all_pass':True,
        'training':False,'new_scene_generalization_test':False,
        'strong_event_claim':'operational candidate evidence only, not universal stable-grasp/placement certification',
        'default_window':'10 control intervals, 11 samples; .005m gripper-local translation; lift rise .005m; placement speed<.02m/s',
        'coverage':{k:sum(bool(r[k]) for r in results) for k in ['hold_candidate_window_starts','lift_hold_candidate_window_starts','released_placement_candidate_window_starts']},
        'post_root_coverage':{k:sum(r[k]>0 for r in results) for k in ['post_root_hold_candidates','post_root_lift_hold_candidates','post_root_release_candidates']},
        'results':results,'repeat_result':repeat,
        'analyzer_sha256':sha(Path(__file__))}
    (OUT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['results','repeat_result']},ensure_ascii=False))

if __name__=='__main__':main()
