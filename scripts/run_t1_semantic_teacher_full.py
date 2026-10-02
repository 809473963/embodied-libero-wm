#!/usr/bin/env python3
"""Run the independent frozen T1 semantic goal-progress teacher audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_c0_all import make_env  # noqa: E402
from scripts.run_semantic_teacher_clean import disable_cameras  # noqa: E402
from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.robust_grounding import BOWLS, FAMILIES, maximum_score, resolve_scores  # noqa: E402
from src.probes.semantic_teacher_clean import (  # noqa: E402
    atomic_json, atomic_npz, calibrated_epsilon, compare_teacher, controlled_and_reach,
    pair_rows, proprio_from_observation, sha256, summarize_labels,
)
from src.probes.t1_semantic_teacher_full import (  # noqa: E402
    BOOTSTRAP_DRAWS, MARGINS, binary_auc, block_bootstrap_ci, choose_margin,
    distribution, margin_audit,
)


VERSION = "t1-semantic-teacher-independent-v1"


def last_commit(path: Path) -> str:
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()


def validate_protocol(path: Path, partition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != VERSION or protocol["status"] != "PREREGISTERED_BEFORE_NEW_DEMO_ACCESS":
        raise RuntimeError("invalid protocol")
    if not last_commit(path): raise RuntimeError("protocol must be committed")
    for item in protocol["frozen_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(f"frozen file changed: {item['path']}")
    if partition == "evaluation":
        for key in ("definition", "calibration"):
            target = ROOT / protocol["outputs"][key]
            if not target.exists() or not last_commit(target): raise RuntimeError("calibration freeze missing")
            if subprocess.check_output(["git", "status", "--porcelain", "--", str(target.relative_to(ROOT))], cwd=ROOT, text=True).strip(): raise RuntimeError("calibration freeze dirty")
    return protocol, json.loads((ROOT / protocol["inputs"]["access"]).read_text())


def grounding_rows(protocol: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    resolver = json.loads((ROOT / protocol["inputs"]["resolver"]).read_text())
    sidecar = np.load(ROOT / protocol["inputs"][f"grounding_{partition}_sidecar"])
    fi = FAMILIES.index(resolver["family"]); rows = []
    for episode in range(len(sidecar["task_id"])):
        refs, statuses, gaps = [], [], []
        for instruction in range(10):
            status, ref, gap = resolve_scores(sidecar["scores"][episode, instruction, fi], float(resolver["threshold"]), maximum_score(resolver["family"]))
            refs.append(ref); statuses.append(status); gaps.append(gap)
        rows.append({"episode_id": episode, "task_id": int(sidecar["task_id"][episode]), "demo_id": int(sidecar["demo_id"][episode]), "referent": refs, "status": statuses, "gap": gaps})
    return rows


def world_offset(inner: Any, name: str, attribute: str) -> np.ndarray:
    pos = np.asarray(inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)
    rotation = np.asarray(inner.sim.data.body_xmat[inner.obj_body_id[name]], dtype=np.float64).reshape(3, 3)
    return pos + rotation @ np.asarray(getattr(inner.get_object(name), attribute), dtype=np.float64)


def collect(protocol: dict[str, Any], access: dict[str, Any], partition: str) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    ground = grounding_rows(protocol, partition)
    guard = AllowedDemoGuard(partition, {int(k): ROOT / v for k, v in access["task_files"].items()}, ["states"])
    contracts = json.loads((ROOT / protocol["inputs"]["task_contract"]).read_text())["tasks"]
    task_rows=[]; demo_rows=[]; episode_rows=[]; time_rows=[]; goal_rows=[]; margin_rows=[]; geometry_rows=[]; proprio_rows=[]; controlled_rows=[]; reach_rows=[]
    episode = 0
    for task, contract in enumerate(contracts):
        env = make_env(ROOT / contract["bddl_path"]); env.seed(93000 + task); env.reset(); disable_cameras(env)
        try:
            for demo in guard.allowed_demo_ids:
                states = guard.read_dataset(task, demo, "states")
                if states.ndim != 2 or not np.isfinite(states).all(): raise ValueError("invalid state array")
                if ground[episode]["task_id"] != task or ground[episode]["demo_id"] != demo: raise RuntimeError("grounding identity mismatch")
                if ground[episode]["status"][task] != "UNIQUE" or ground[episode]["referent"][task] != 0: raise RuntimeError("frozen G2 correct referent failed")
                for timestep, state in enumerate(states):
                    observation = env.regenerate_obs_from_state(state); inner = env.env
                    plate_pos = np.asarray(inner.object_states_dict["plate_1"].get_geom_state()["pos"], dtype=np.float64)
                    plate_model = inner.get_object("plate_1"); plate_top_raw = plate_pos + np.asarray(plate_model.top_offset, dtype=np.float64); plate_top_world = world_offset(inner, "plate_1", "top_offset")
                    goals=[]; margins=[]; geometry=[]; controls=[]; reaches=[]
                    for bowl in BOWLS:
                        bowl_pos = np.asarray(inner.object_states_dict[bowl].get_geom_state()["pos"], dtype=np.float64); bowl_model = inner.get_object(bowl)
                        xy = float(np.linalg.norm(bowl_pos[:2] - plate_pos[:2])); xy_clear = 0.03 - xy; z_clear = float(bowl_pos[2] - plate_pos[2])
                        bowl_bottom_raw = bowl_pos + np.asarray(bowl_model.bottom_offset, dtype=np.float64); bowl_bottom_world = world_offset(inner, bowl, "bottom_offset")
                        gap_raw = float(bowl_bottom_raw[2] - plate_top_raw[2]); gap_world = float(bowl_bottom_world[2] - plate_top_world[2])
                        m0 = min(xy_clear / 0.03, z_clear / 0.03, -abs(gap_raw) / 0.03)
                        m1 = min(xy_clear / 0.03, z_clear / 0.03, -abs(gap_world) / 0.03)
                        m2 = min(xy_clear / 0.03, z_clear / 0.03)
                        goals.append(bool(inner._eval_predicate(["on", bowl, "plate_1"]))); margins.append([m0,m1,m2]); geometry.append([xy,xy_clear,z_clear,gap_raw,gap_world,bowl_bottom_raw[2],bowl_bottom_world[2],plate_top_raw[2],plate_top_world[2]])
                        c,r = controlled_and_reach(inner,bowl); controls.append(c); reaches.append(r)
                    task_rows.append(task); demo_rows.append(int(demo)); episode_rows.append(episode); time_rows.append(timestep); goal_rows.append(goals); margin_rows.append(margins); geometry_rows.append(geometry); controlled_rows.append(controls); reach_rows.append(reaches); proprio_rows.append(proprio_from_observation(observation))
                episode += 1
        finally: env.close()
    guard.assert_clean()
    arrays={"task_id":np.asarray(task_rows,dtype=np.int8),"demo_id":np.asarray(demo_rows,dtype=np.int8),"episode_id":np.asarray(episode_rows,dtype=np.int8),"timestep":np.asarray(time_rows,dtype=np.int32),"goal":np.asarray(goal_rows,dtype=np.bool_),"margins":np.asarray(margin_rows,dtype=np.float64),"geometry":np.asarray(geometry_rows,dtype=np.float64),"controlled":np.asarray(controlled_rows,dtype=np.bool_),"reach":np.asarray(reach_rows,dtype=np.float64),"proprio":np.asarray(proprio_rows,dtype=np.float64)}
    if not all(np.isfinite(v).all() for k,v in arrays.items() if k not in {"goal","controlled"}): raise FloatingPointError("non-finite collection")
    return arrays, guard.ledger.as_dict(), ground


def ground_arrays(ground: list[dict[str, Any]]) -> tuple[np.ndarray,np.ndarray]:
    status=np.zeros((40,10),dtype=np.int8); refs=np.full((40,10),-1,dtype=np.int8)
    for row in ground:
        e=row["episode_id"]
        for i in range(10): status[e,i]={"NOT_FOUND":0,"UNIQUE":1,"AMBIGUOUS":2}[row["status"][i]]; refs[e,i]=row["referent"][i]
    return status,refs


def pairs(arrays: dict[str,np.ndarray], refs: np.ndarray) -> dict[str,np.ndarray]:
    left=[]; right=[]; gap=[]; task=[]; demo=[]; ref=[]
    for a,b,g in pair_rows(arrays["episode_id"],arrays["timestep"]):
        t=int(arrays["task_id"][a]); e=int(arrays["episode_id"][a]); r=int(refs[e,t])
        if r>=0: left.append(a); right.append(b); gap.append(g); task.append(t); demo.append(int(arrays["demo_id"][a])); ref.append(r)
    return {"left":np.asarray(left,dtype=np.int32),"right":np.asarray(right,dtype=np.int32),"gap":np.asarray(gap,dtype=np.int8),"task_id":np.asarray(task,dtype=np.int8),"demo_id":np.asarray(demo,dtype=np.int8),"referent":np.asarray(ref,dtype=np.int8)}


def labels_for(arrays: dict[str,np.ndarray], p: dict[str,np.ndarray], margin: np.ndarray, family: str, eps_goal: float, eps_reach: float) -> np.ndarray:
    out=np.empty(len(p["left"]),dtype=np.int8)
    for i,(a,b,r) in enumerate(zip(p["left"],p["right"],p["referent"])):
        out[i]=compare_teacher(arrays["goal"][a,r],margin[a,r],arrays["controlled"][a,r],arrays["reach"][a,r],arrays["goal"][b,r],margin[b,r],arrays["controlled"][b,r],arrays["reach"][b,r],family,eps_goal,eps_reach)
    return out


def stats_with_ci(labels: np.ndarray,p:dict[str,np.ndarray],mask:np.ndarray,offset:int) -> dict[str,Any]:
    sub=labels[mask]; result=summarize_labels(sub); task=p["task_id"][mask]; demo=p["demo_id"][mask]; den=np.ones(len(sub))
    for j,(name,predicate) in enumerate((("non_tie",sub!=0),("progress",sub>0),("regression",sub<0),("tie",sub==0))): result[f"{name}_ci95"]=block_bootstrap_ci(task,demo,predicate,den,offset+j)
    return result


def fit_probe(x:np.ndarray,y:np.ndarray) -> dict[str,Any]:
    keep=y!=0; x=x[keep]; y=y[keep].astype(np.float64); mean=x.mean(0); std=x.std(0); std[std<1e-8]=1; design=np.column_stack(((x-mean)/std,np.ones(len(x)))); penalty=np.eye(design.shape[1])*.01; penalty[-1,-1]=0; w=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {"mean":mean.tolist(),"std":std.tolist(),"weights":w.tolist(),"ridge":.01}


def apply_probe(x:np.ndarray,y:np.ndarray,task:np.ndarray,demo:np.ndarray,model:dict[str,Any]) -> dict[str,Any]:
    keep=y!=0; x=x[keep]; y=y[keep]; task=task[keep]; demo=demo[keep]; score=np.column_stack(((x-np.asarray(model["mean"]))/np.asarray(model["std"]),np.ones(len(x))))@np.asarray(model["weights"]); pred=np.where(score>=0,1,-1); correct=pred==y
    per={str(t):{"rows":int(np.sum(task==t)),"accuracy":float(np.mean(correct[task==t]))} for t in range(10)}
    return {"rows":int(len(y)),"accuracy":float(correct.mean()),"accuracy_ci95":block_bootstrap_ci(task,demo,correct,np.ones(len(y)),60),"auroc":binary_auc(score,y),"per_task":per,"score":score,"correct":correct,"kept_task":task,"kept_demo":demo}


def controls(arrays:dict[str,np.ndarray],p:dict[str,np.ndarray],labels:np.ndarray,margin:np.ndarray,definition:dict[str,Any]|None) -> tuple[dict[str,Any],dict[str,Any]|None]:
    a,b=p["left"],p["right"]; x=np.concatenate((arrays["proprio"][a],arrays["proprio"][b],arrays["proprio"][b]-arrays["proprio"][a]),axis=1); local=p["gap"]<=8
    if definition is None:
        mean=arrays["proprio"].mean(0); std=arrays["proprio"].std(0); std[std<1e-8]=1; z=(arrays["proprio"]-mean)/std; distance=np.linalg.norm(z[b]-z[a],axis=1); q10=float(np.quantile(distance[local],.10)); q75=float(np.quantile(distance[local],.75)); model=fit_probe(x[local],labels[local]); freeze={"proprio_mean":mean.tolist(),"proprio_std":std.tolist(),"robot_match_threshold":q10,"robot_difference_threshold":q75,"probe":model}
    else:
        mean=np.asarray(definition["controls"]["proprio_mean"]); std=np.asarray(definition["controls"]["proprio_std"]); q10=float(definition["controls"]["robot_match_threshold"]); q75=float(definition["controls"]["robot_difference_threshold"]); model=definition["controls"]["probe"]; z=(arrays["proprio"]-mean)/std; distance=np.linalg.norm(z[b]-z[a],axis=1); freeze=None
    tie=local&(labels==0)&(distance>=q75); diff=local&(labels!=0)&(distance<=q10); margin_delta=np.abs(margin[b,p["referent"]]-margin[a,p["referent"]])
    def row(mask,semantic):
        return {"rows":int(mask.sum()),"task_count":int(len(np.unique(p["task_id"][mask]))),"proprio_distance":distribution(distance[mask]),"semantic_margin_abs_difference":distribution(margin_delta[mask]),"teacher_condition_rate":1.0,"pass":bool(mask.sum()>=50 and len(np.unique(p["task_id"][mask]))>=7),"condition":semantic}
    probe=apply_probe(x[local],labels[local],p["task_id"][local],p["demo_id"][local],model); probe.pop("score"); probe.pop("correct"); probe.pop("kept_task"); probe.pop("kept_demo")
    return {"semantic_match_robot_difference":row(tie,"TIE"),"robot_match_semantic_difference":row(diff,"NON_TIE"),"proprio_probe":probe},freeze


def language_audit(arrays:dict[str,np.ndarray],ground:list[dict[str,Any]],p:dict[str,np.ndarray],margin:np.ndarray,labels:np.ndarray,family:str,eps_goal:float,eps_reach:float) -> dict[str,Any]:
    state_total=state_changed=goal_changed=switched=0; margin_changes=[]; state_group={"same":{"n":0,"changed":0},"different":{"n":0,"changed":0}}
    pair_total=pair_changed=reversal=0; taxonomy={"BETTER_TO_WORSE":0,"WORSE_TO_BETTER":0,"NON_TIE_TO_TIE":0,"TIE_TO_NON_TIE":0,"UNCHANGED":0}; task_num=np.zeros(10); task_den=np.zeros(10); row_task=[]; row_demo=[]; row_changed=[]; row_reversal=[]
    episode_frames={e:np.flatnonzero(arrays["episode_id"]==e) for e in range(40)}
    for row in ground:
        e,t,d=row["episode_id"],row["task_id"],row["demo_id"]; cr=row["referent"][t]
        for alt in range(10):
            if alt==t or row["status"][alt]!="UNIQUE": continue
            ar=row["referent"][alt]; group="different" if ar!=cr else "same"
            for idx in episode_frames[e]:
                gc,ga=arrays["goal"][idx,cr],arrays["goal"][idx,ar]; delta=abs(margin[idx,cr]-margin[idx,ar]); change=bool(gc!=ga or delta>eps_goal)
                state_total+=1; state_changed+=change; goal_changed+=bool(gc!=ga); switched+=int(ar!=cr); margin_changes.append(delta); state_group[group]["n"]+=1; state_group[group]["changed"]+=change
    local_indices=np.flatnonzero(p["gap"]<=8)
    for i in local_indices:
        a,b=int(p["left"][i]),int(p["right"][i]); e=int(arrays["episode_id"][a]); t=int(p["task_id"][i]); d=int(p["demo_id"][i]); cr=ground[e]["referent"][t]; correct=int(labels[i])
        for alt in range(10):
            if alt==t or ground[e]["status"][alt]!="UNIQUE": continue
            ar=ground[e]["referent"][alt]; other=compare_teacher(arrays["goal"][a,ar],margin[a,ar],arrays["controlled"][a,ar],arrays["reach"][a,ar],arrays["goal"][b,ar],margin[b,ar],arrays["controlled"][b,ar],arrays["reach"][b,ar],family,eps_goal,eps_reach); changed=correct!=other; rev=(correct==1 and other==-1) or (correct==-1 and other==1)
            pair_total+=1; pair_changed+=changed; reversal+=rev; task_num[t]+=changed; task_den[t]+=1; row_task.append(t); row_demo.append(d); row_changed.append(changed); row_reversal.append(rev)
            if not changed: taxonomy["UNCHANGED"]+=1
            elif correct==1 and other==-1: taxonomy["BETTER_TO_WORSE"]+=1
            elif correct==-1 and other==1: taxonomy["WORSE_TO_BETTER"]+=1
            elif correct!=0 and other==0: taxonomy["NON_TIE_TO_TIE"]+=1
            elif correct==0 and other!=0: taxonomy["TIE_TO_NON_TIE"]+=1
    row_task=np.asarray(row_task); row_demo=np.asarray(row_demo); row_changed=np.asarray(row_changed); row_reversal=np.asarray(row_reversal)
    per_task={str(t):{"eligible_pairs":int(task_den[t]),"material_change_rate":float(task_num[t]/task_den[t]) if task_den[t] else 0.0} for t in range(10)}
    for g in state_group.values(): g["change_rate"]=g["changed"]/g["n"] if g["n"] else 0.0
    return {"eligible_state_cases":state_total,"referent_switch_rate":switched/state_total,"goal_state_change_rate":goal_changed/state_total,"semantic_state_change_rate":state_changed/state_total,"same_state_margin_abs_change":distribution(np.asarray(margin_changes)),"referent_switch_stratification":state_group,"eligible_local_pair_counterfactuals":pair_total,"material_pair_preference_change_rate":pair_changed/pair_total,"material_change_ci95":block_bootstrap_ci(row_task,row_demo,row_changed,np.ones(pair_total),40),"preference_reversal_rate":reversal/pair_total,"reversal_ci95":block_bootstrap_ci(row_task,row_demo,row_reversal,np.ones(pair_total),41),"taxonomy":{k:{"count":v,"rate":v/pair_total} for k,v in taxonomy.items()},"per_task":per_task,"tasks_material_at_least_20pct":int(sum(v["material_change_rate"]>=.20 for v in per_task.values()))}


def analyze(arrays:dict[str,np.ndarray],ground:list[dict[str,Any]],partition:str,protocol:dict[str,Any],definition:dict[str,Any]|None) -> tuple[dict[str,Any],dict[str,Any]|None]:
    status,refs=ground_arrays(ground); p=pairs(arrays,refs); audits=margin_audit(arrays["margins"],arrays["goal"],arrays["task_id"],arrays["episode_id"],arrays["timestep"])
    if partition=="calibration":
        selected=choose_margin(audits)
        if selected=="none": raise RuntimeError("no margin passed geometry audit")
        mi=MARGINS.index(selected); margin=arrays["margins"][:,:,mi]; adjacent=p["gap"]==1; changes=np.abs(margin[p["right"][adjacent],p["referent"][adjacent]]-margin[p["left"][adjacent],p["referent"][adjacent]]); eps_goal=calibrated_epsilon(changes); reach_changes=np.abs(arrays["reach"][p["right"][adjacent],p["referent"][adjacent]]-arrays["reach"][p["left"][adjacent],p["referent"][adjacent]]); eps_reach=calibrated_epsilon(reach_changes); t1=labels_for(arrays,p,margin,"T1",eps_goal,eps_reach); local=p["gap"]<=8; per_task_t1={str(t):summarize_labels(t1[local&(p["task_id"]==t)]) for t in range(10)}; usable=sum(v["non_tie_fraction"]>=.30 for v in per_task_t1.values()); family="T1" if np.mean(t1[local]!=0)>=.30 and usable>=7 else "T2"
    else:
        selected=definition["selected_margin"]; mi=MARGINS.index(selected); margin=arrays["margins"][:,:,mi]; eps_goal=float(definition["epsilon_goal"]); eps_reach=float(definition["epsilon_reach"]); family=definition["selected_teacher"]
    labels=labels_for(arrays,p,margin,family,eps_goal,eps_reach); local=p["gap"]<=8
    t0=np.asarray([1 if arrays["goal"][b,r] and not arrays["goal"][a,r] else -1 if arrays["goal"][a,r] and not arrays["goal"][b,r] else 0 for a,b,r in zip(p["left"],p["right"],p["referent"])],dtype=np.int8)
    local_stats=stats_with_ci(labels,p,local,0); t0_stats=stats_with_ci(t0,p,local,10); per_gap={}
    for j,(lo,hi) in enumerate(((1,2),(3,4),(5,8),(9,16))): per_gap[f"{lo}-{hi}"]=stats_with_ci(labels,p,(p["gap"]>=lo)&(p["gap"]<=hi),20+j*4)
    per_task={str(t):summarize_labels(labels[local&(p["task_id"]==t)]) for t in range(10)}; usable=int(sum(v["non_tie_fraction"]>=.30 for v in per_task.values()))
    language=language_audit(arrays,ground,p,margin,labels,family,eps_goal,eps_reach); control,freeze=controls(arrays,p,labels,margin,definition)
    selected_audit=next(row for row in audits if row["family"]==selected); gates={"grounding_hash_unchanged":True,"goal_margin_geometry":selected_audit["admissible"],"finite":bool(np.isfinite(margin).all()),"local_density":local_stats["non_tie_fraction"]>=.30,"usable_tasks":usable>=7,"language_pair_change":language["material_pair_preference_change_rate"]>=.20,"language_tasks":language["tasks_material_at_least_20pct"]>=7,"semantic_match_robot_difference":control["semantic_match_robot_difference"]["pass"],"robot_match_semantic_difference":control["robot_match_semantic_difference"]["pass"],"no_severe_proprio_shortcut":control["proprio_probe"]["accuracy"]<.90,"forbidden_access_zero":True}
    result={"result_version":"t1-semantic-teacher-independent-result-v1","partition":partition,"selected_margin":selected,"selected_teacher":family,"epsilon_goal":eps_goal,"epsilon_reach":eps_reach,"frames":int(len(arrays["task_id"])),"episodes":40,"pair_rows":int(len(labels)),"margin_audit":audits,"orientation_audit":{"raw_vs_world_surface_gap_abs_difference":distribution(np.abs(arrays["geometry"][:,:,3]-arrays["geometry"][:,:,4]).reshape(-1))},"t0_local":t0_stats,"t1_local":local_stats,"usable_task_count":usable,"per_task":per_task,"per_gap":per_gap,"teacher_time_agreement":local_stats["time_agreement_among_non_ties"],"language":language,"controls":control,"historical_temporal_proprio_accuracy":.96224,"semantic_minus_historical_proprio_points":100*(control["proprio_probe"]["accuracy"]-.96224),"gates":gates,"complete_gate":bool(all(gates.values()))}
    if partition=="calibration": definition_out={"definition_version":"t1-semantic-teacher-independent-definition-v1","grounding":"G2 frozen; ground once and persist physical identity","grounding_hash":protocol["expected_grounding_hash"],"selected_margin":selected,"selected_margin_equation":json.loads((ROOT/protocol["inputs"]["margin_contract"]).read_text())["families"][selected]["equation"],"selected_teacher":family,"epsilon_goal":eps_goal,"epsilon_reach":eps_reach,"conditional_t2_activated":family=="T2","controls":freeze,"bootstrap":{"draws":BOOTSTRAP_DRAWS,"seed":20261002,"unit":"task-stratified demo block"},"calibration_margin_audit":audits}
    else: definition_out=None
    sidecar={**arrays,"selected_margin_values":margin,"pair_left":p["left"],"pair_right":p["right"],"pair_gap":p["gap"],"pair_task_id":p["task_id"],"pair_demo_id":p["demo_id"],"pair_referent":p["referent"],"pair_label":labels,"t0_pair_label":t0,"ground_status":status,"ground_referent":refs}
    return result,definition_out,sidecar


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--partition",choices=("calibration","evaluation"),required=True); ap.add_argument("--protocol",type=Path,default=ROOT/"configs/t1_semantic_teacher_protocol.json"); args=ap.parse_args(); protocol,access=validate_protocol(args.protocol,args.partition); definition=json.loads((ROOT/protocol["outputs"]["definition"]).read_text()) if args.partition=="evaluation" else None
    arrays,ledger,ground=collect(protocol,access,args.partition); result,definition_out,sidecar=analyze(arrays,ground,args.partition,protocol,definition); result["access_ledger"]=ledger; result["gates"]["forbidden_access_zero"]=ledger["forbidden_access_count"]==0; result["complete_gate"]=bool(all(result["gates"].values()))
    output="calibration" if args.partition=="calibration" else "evaluation"; atomic_npz(ROOT/protocol["outputs"][f"{args.partition}_sidecar"],sidecar); result["sidecar_sha256"]=sha256(ROOT/protocol["outputs"][f"{args.partition}_sidecar"]); atomic_json(ROOT/protocol["outputs"][f"{args.partition}_ledger"],ledger); atomic_json(ROOT/protocol["outputs"][output],result)
    if definition_out is not None: definition_out["calibration_result_sha256"]=sha256(ROOT/protocol["outputs"]["calibration"]); atomic_json(ROOT/protocol["outputs"]["definition"],definition_out)
    if args.partition=="evaluation":
        calibration=json.loads((ROOT/protocol["outputs"]["calibration"]).read_text()); atomic_json(ROOT/protocol["outputs"]["language"],{"result_version":"t1-language-counterfactual-v1","calibration":calibration["language"],"evaluation":result["language"]}); atomic_json(ROOT/protocol["outputs"]["controls"],{"result_version":"t1-robot-shortcut-controls-v1","historical_temporal_proprio_accuracy":.96224,"calibration":calibration["controls"],"evaluation":result["controls"],"evaluation_reduction_points":result["semantic_minus_historical_proprio_points"]})
    print(json.dumps({"partition":args.partition,"margin":result["selected_margin"],"teacher":result["selected_teacher"],"complete_gate":result["complete_gate"],"forbidden_access_count":ledger["forbidden_access_count"]},indent=2))


if __name__=="__main__": main()
