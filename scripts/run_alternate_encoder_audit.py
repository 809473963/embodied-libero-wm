#!/usr/bin/env python3
"""Execute the conditional calibration-only alternate frozen encoder audit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.clip_internal_layer import (  # noqa: E402
    attention_diagnostics, attention_for, binary_metrics, fit_layer_statistics, parameter_count,
    predict_goal_logits, predict_m2_logits, predict_pair_logits,
    predict_referent_logits, train_goal_probe, train_m2_probe,
    train_pair_reader, train_referent_probe,
)
from src.probes.semantic_value_distillation import (  # noqa: E402
    CounterfactualSet, PairSet, cross_demo_metrics, fit_temperature,
    load_partition, make_counterfactual_set, make_cross_demo_controls,
    make_state_targets, make_training_pairs, non_tie_metrics, ordinary_pairs,
    sigmoid,
)


VERSION = "alternate-frozen-visual-encoder-audit-v1"
DEPENDENCY_DIRECTORY = ROOT / ".cache/stage_e_python"
FAMILIES = ("DINOv2-S14", "SigLIP-B16-224")


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n"); os.replace(temporary,path)


def atomic_npz(path: Path, arrays: dict[str,np.ndarray]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".part")
    with temporary.open("wb") as stream: np.savez(stream,**arrays)
    os.replace(temporary,path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(["git","log","-1","--format=%H","--",str(path.relative_to(ROOT))],cwd=ROOT,text=True).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def subset_pairset(pairs: PairSet, mask: np.ndarray) -> PairSet:
    return PairSet(*(getattr(pairs, field)[mask] for field in pairs.__dataclass_fields__))


def subset_counterfactual(cf: CounterfactualSet, mask: np.ndarray) -> CounterfactualSet:
    return CounterfactualSet(*(getattr(cf, field)[mask] for field in cf.__dataclass_fields__))


def pairset_for_counterfactual(cf: CounterfactualSet, alternate: bool) -> PairSet:
    return PairSet(
        cf.a, cf.b, cf.alternate_language if alternate else cf.original_language,
        cf.alternate_label if alternate else cf.original_label,
        cf.task_id, cf.demo_id, np.zeros(len(cf.a), dtype=np.int64),
        np.zeros(len(cf.a), dtype=np.int8),
        np.full(len(cf.a), "language_counterfactual", dtype="U32"),
    )


def main_metrics(
    ordinary_probability: np.ndarray, ordinary: PairSet,
    original_probability: np.ndarray, alternate_probability: np.ndarray,
    counterfactual: CounterfactualSet, matched_probability: np.ndarray,
    matched: PairSet, different_probability: np.ndarray, different: PairSet,
) -> dict[str, Any]:
    local = ordinary.gap <= 8
    non_tie = local & (ordinary.label != 0)
    ties = local & (ordinary.label == 0)
    changed = counterfactual.original_label != counterfactual.alternate_label
    direction = (
        counterfactual.alternate_label.astype(float)
        - counterfactual.original_label.astype(float)
    )
    agreement = (alternate_probability - original_probability) * direction > 0
    reversal = counterfactual.original_label * counterfactual.alternate_label == -1
    original_correct = np.sign(original_probability - 0.5) == counterfactual.original_label
    alternate_correct = np.sign(alternate_probability - 0.5) == counterfactual.alternate_label
    same = counterfactual.same_referent
    return {
        "ordinary_local_non_tie": non_tie_metrics(
            ordinary_probability[non_tie], ordinary.label[non_tie]
        ),
        "ordinary_local_tie": {
            "rows": int(ties.sum()),
            "mean_abs_from_half": float(np.abs(ordinary_probability[ties] - 0.5).mean()),
            "median_abs_from_half": float(np.median(np.abs(ordinary_probability[ties] - 0.5))),
        },
        "language_counterfactual": {
            "eligible_rows": int(len(changed)),
            "teacher_changed_rows": int(changed.sum()),
            "teacher_change_agreement": float(agreement[changed].mean()),
            "true_reversal": {
                "rows": int(reversal.sum()),
                "both_correct_rate": float(
                    (original_correct & alternate_correct)[reversal].mean()
                ),
            },
            "same_referent": {
                "rows": int(same.sum()),
                "mean_abs_probability_change": float(
                    np.abs(alternate_probability[same] - original_probability[same]).mean()
                ),
                "false_preference_flip_rate": float(
                    (
                        np.sign(alternate_probability[same] - 0.5)
                        != np.sign(original_probability[same] - 0.5)
                    ).mean()
                ),
            },
        },
        "cross_demo_semantic_match": cross_demo_metrics(matched_probability, matched),
        "cross_demo_semantic_difference": cross_demo_metrics(
            different_probability, different
        ),
    }


def referent_rows(
    data: Any, demos: tuple[int, ...], points: int = 8
) -> dict[str, np.ndarray]:
    rows = []
    for episode in np.unique(data.episode_id):
        ids = np.flatnonzero(data.episode_id == episode)
        demo = int(data.demo_id[ids[0]])
        if demo not in demos:
            continue
        languages = np.flatnonzero(data.ground_status[int(episode)] == 1)
        frames = ids[np.unique(np.linspace(0, len(ids) - 1, points, dtype=np.int64))]
        for frame in frames:
            for left in range(len(languages)):
                for right in range(left + 1, len(languages)):
                    language_a, language_b = int(languages[left]), int(languages[right])
                    label = int(
                        data.ground_referent[int(episode), language_a]
                        != data.ground_referent[int(episode), language_b]
                    )
                    rows.append(
                        (int(frame), language_a, language_b, label,
                         int(data.task_id[frame]), demo)
                    )
    matrix = np.asarray(rows, dtype=np.int64)
    return {
        "frame": matrix[:, 0], "language_a": matrix[:, 1],
        "language_b": matrix[:, 2], "label": matrix[:, 3].astype(np.bool_),
        "task": matrix[:, 4], "demo": matrix[:, 5],
    }


def m2_pairs(data: Any, demos: tuple[int, ...], epsilon: float) -> PairSet:
    base = ordinary_pairs(data, demos)
    local = base.gap <= 8
    episode = data.episode_id[base.a]
    referent = data.ground_referent[episode, base.language_id]
    same_goal = data.goal[base.a, referent] == data.goal[base.b, referent]
    delta = data.margin[base.b, referent] - data.margin[base.a, referent]
    keep = local & same_goal & (np.abs(delta) > epsilon)
    a, b, language = base.a[keep], base.b[keep], base.language_id[keep]
    label = np.sign(delta[keep]).astype(np.int8)
    task, demo = base.task_id[keep], base.demo_id[keep]
    gap, code = base.gap[keep], base.gap_code[keep]
    return PairSet(
        np.concatenate((a, b)), np.concatenate((b, a)),
        np.concatenate((language, language)), np.concatenate((label, -label)),
        np.concatenate((task, task)), np.concatenate((demo, demo)),
        np.concatenate((gap, -gap)), np.concatenate((code, code)),
        np.full(2 * len(a), "M2_order", dtype="U24"),
    )


def filter_m2_cross(data: Any, pairs: PairSet, epsilon: float) -> PairSet:
    episode_a, episode_b = data.episode_id[pairs.a], data.episode_id[pairs.b]
    referent_a = data.ground_referent[episode_a, pairs.language_id]
    referent_b = data.ground_referent[episode_b, pairs.language_id]
    same_goal = data.goal[pairs.a, referent_a] == data.goal[pairs.b, referent_b]
    delta = data.margin[pairs.b, referent_b] - data.margin[pairs.a, referent_a]
    keep = same_goal & (np.abs(delta) > epsilon)
    selected = subset_pairset(pairs, keep)
    return PairSet(
        selected.a, selected.b, selected.language_id,
        np.sign(delta[keep]).astype(np.int8), selected.task_id, selected.demo_id,
        selected.gap, selected.gap_code, selected.pair_type,
    )


def filter_goal_cross(data: Any, pairs: PairSet) -> PairSet:
    episode_a, episode_b = data.episode_id[pairs.a], data.episode_id[pairs.b]
    referent_a = data.ground_referent[episode_a, pairs.language_id]
    referent_b = data.ground_referent[episode_b, pairs.language_id]
    goal_a, goal_b = data.goal[pairs.a, referent_a], data.goal[pairs.b, referent_b]
    keep = goal_a != goal_b
    selected = subset_pairset(pairs, keep)
    return PairSet(
        selected.a, selected.b, selected.language_id,
        np.where(goal_b[keep], 1, -1).astype(np.int8),
        selected.task_id, selected.demo_id, selected.gap,
        selected.gap_code, selected.pair_type,
    )


def state_goal_rows(data: Any, demos: tuple[int, ...]) -> dict[str, np.ndarray]:
    frame, language, goal, _ = make_state_targets(data, demos)
    return {"frame": frame, "language": language, "label": goal.astype(np.bool_)}


def validate_protocol(path: Path) -> dict[str,Any]:
    protocol=json.loads(path.read_text())
    if protocol.get("protocol_version")!=VERSION or protocol.get("status")!="PREREGISTERED_BEFORE_STAGE_E_WEIGHT_DOWNLOAD_FEATURE_EXTRACTION_OR_TRAINING": raise ValueError("invalid Stage-E protocol")
    if not last_commit(path): raise RuntimeError("Stage-E protocol must be committed")
    contract=json.loads((ROOT/protocol["visual"]["contracts"]).read_text())
    if contract.get("status")!="PREREGISTERED_BEFORE_STAGE_E_WEIGHT_DOWNLOAD_FEATURE_EXTRACTION_OR_TRAINING": raise ValueError("invalid encoder contracts")
    for row in protocol["frozen_files"]:
        if sha256(ROOT/row["path"])!=row["sha256"]: raise RuntimeError(f"frozen hash mismatch: {row['path']}")
    return protocol


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    return torch.autocast("cuda",dtype=torch.float16) if device.type=="cuda" else contextlib.nullcontext()


def load_encoder(family: str, contract: dict[str,Any], device: torch.device) -> tuple[Any,Any,Path,str]:
    if not DEPENDENCY_DIRECTORY.is_dir(): raise FileNotFoundError("isolated Stage-E dependencies not installed")
    sys.path.insert(0,str(DEPENDENCY_DIRECTORY))
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel, __version__  # type: ignore
    if __version__!="4.57.1": raise RuntimeError(f"unexpected isolated transformers version: {__version__}")
    row=contract["families"][family]; repository=row["repository"]; revision=row["revision"]
    snapshot=ROOT/".cache/huggingface/hub"/("models--"+repository.replace("/","--"))/"snapshots"/revision
    if not snapshot.is_dir(): raise FileNotFoundError(snapshot)
    if sha256(snapshot/"config.json") != row["config_sha256"]: raise RuntimeError(f"{family} config hash mismatch")
    if sha256(snapshot/"preprocessor_config.json") != row["preprocessor_sha256"]: raise RuntimeError(f"{family} preprocessor hash mismatch")
    processor=AutoImageProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    if family=="DINOv2-S14": model=AutoModel.from_pretrained(snapshot,local_files_only=True)
    else: model=SiglipVisionModel.from_pretrained(snapshot,local_files_only=True)
    model=model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model,processor,snapshot,__version__


@torch.inference_mode()
def encode_batch(family: str, model: Any, processor: Any, images: np.ndarray, device: torch.device) -> np.ndarray:
    pixels=processor(images=[np.ascontiguousarray(image) for image in images],return_tensors="pt")["pixel_values"].to(device)
    with autocast(device):
        output=model(pixel_values=pixels,return_dict=True)
        hidden=output.last_hidden_state
        if family=="DINOv2-S14": hidden=hidden[:,1:,:]
    return hidden.float().cpu().numpy().astype(np.float16)


def extract_one(protocol: dict[str,Any], contract: dict[str,Any], family: str, data: Any, task_files: dict[int,Path], device: torch.device) -> tuple[dict[str,Any],dict[str,Any]]:
    row=contract["families"][family]; output_path=ROOT/protocol["outputs"]["feature_directory"]/f"{family}.npy"
    if output_path.exists(): raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    model,processor,snapshot,transformers_version=load_encoder(family,contract,device)
    weights=snapshot/"model.safetensors"
    if not weights.exists(): raise FileNotFoundError(weights)
    guard=AllowedDemoGuard("calibration",task_files,["obs/agentview_rgb"])
    temporary=output_path.with_suffix(".npy.part")
    output=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float16,shape=(len(data.vision),int(row["token_count"]),int(row["dimension"])))
    cursor=0; repeated_exact=True; repeated=False; batch_size=32 if family=="DINOv2-S14" else 16
    for task in range(10):
        for demo in guard.allowed_demo_ids:
            images=guard.read_dataset(task,demo,"obs/agentview_rgb"); count=len(images); frame_slice=slice(cursor,cursor+count)
            if images.shape[1:]!=(128,128,3) or images.dtype!=np.uint8: raise ValueError("unexpected RGB schema")
            if not (np.all(data.task_id[frame_slice]==task) and np.all(data.demo_id[frame_slice]==demo) and np.array_equal(data.timestep[frame_slice],np.arange(count))): raise ValueError("RGB/T1 identity mismatch")
            for start in range(0,count,batch_size):
                stop=min(start+batch_size,count); value=encode_batch(family,model,processor,images[start:stop],device)
                if value.shape!=(stop-start,int(row["token_count"]),int(row["dimension"])) or not np.isfinite(value).all(): raise ValueError(f"invalid {family} output")
                if not repeated:
                    repeated_exact &= np.array_equal(value,encode_batch(family,model,processor,images[start:stop],device)); repeated=True
                output[cursor+start:cursor+stop]=value
            cursor+=count; print(json.dumps({"stage_e_extraction":{"family":family,"task":task,"demo":demo,"frames":count}}),flush=True)
    output.flush(); del output; os.replace(temporary,output_path); guard.assert_clean()
    result={"family":family,"repository":row["repository"],"revision":row["revision"],"transformers_version":transformers_version,"weights_sha256":sha256(weights),"config_sha256":sha256(snapshot/"config.json"),"preprocessor_sha256":sha256(snapshot/"preprocessor_config.json"),"feature_sha256":sha256(output_path),"shape":[len(data.vision),int(row["token_count"]),int(row["dimension"])],"dtype":"float16","repeated_sample_float16_exact":repeated_exact,"trainable_encoder_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"all_finite":True,"complete_gate":bool(cursor==len(data.vision) and repeated_exact and guard.ledger.as_dict()["forbidden_access_count"]==0)}
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return result,guard.ledger.as_dict()


def extract_features(protocol: dict[str,Any],device: torch.device) -> dict[str,Any]:
    contract=json.loads((ROOT/protocol["visual"]["contracts"]).read_text())
    data,feature_ledger=load_partition(ROOT/"results/t1_semantic_teacher/calibration_sidecar.npz",ROOT/"artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5","calibration")
    manifest=json.loads((ROOT/"artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/feature_manifest.json").read_text()); task_files={int(row["task_id"]):ROOT/row["source_hdf5_relative_path"] for row in manifest["tasks"]}
    families={}; ledgers={}
    for family in FAMILIES:
        families[family],ledgers[family]=extract_one(protocol,contract,family,data,task_files,device)
        if not families[family]["complete_gate"]: raise RuntimeError(f"{family} extraction failed")
    access={"stage":"E","partition":"calibration_only","source_rgb_by_family":ledgers,"frozen_feature_store":feature_ledger,"opened_demo_ids":[31,32,33,34],"stage_e_not_opened_demo_ids":[36,37,38,39],"forbidden_demo_ids":list(range(40,50)),"forbidden_numeric_access":0,"forbidden_metadata_access":0}
    atomic_json(ROOT/protocol["outputs"]["access_ledger"],access)
    result={"schema_version":1,"stage":"E","mode":"alternate_frozen_encoder_feature_extraction","frames":len(data.vision),"families":families,"access_ledger_sha256":sha256(ROOT/protocol["outputs"]["access_ledger"]),"complete_gate":all(row["complete_gate"] for row in families.values()) and feature_ledger["forbidden_access_count"]==0}
    atomic_json(ROOT/protocol["outputs"]["feature_manifest"],result)
    return result


def calibrate(protocol: dict[str,Any],device: torch.device) -> dict[str,Any]:
    feature_manifest_path=ROOT/protocol["outputs"]["feature_manifest"]; extraction=json.loads(feature_manifest_path.read_text())
    if not extraction.get("complete_gate"): raise RuntimeError("Stage-E extraction invalid")
    data,feature_ledger=load_partition(ROOT/"results/t1_semantic_teacher/calibration_sidecar.npz",ROOT/"artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5","calibration")
    train_demos=tuple(protocol["partitions"]["fit_demo_ids"]); validation_demos=tuple(protocol["partitions"]["selection_demo_ids"]); train_mask=np.isin(data.demo_id,train_demos)
    definition=json.loads((ROOT/"configs/t1_teacher_frozen_definition.json").read_text()); controls=definition["controls"]; sampler=json.loads((ROOT/protocol["training"]["pair_sampler"]).read_text()); epsilon=float(protocol["frozen_teacher"]["epsilon_goal"])
    training_pairs=make_training_pairs(data,train_demos,epsilon,sampler,float(controls["robot_match_threshold"]),float(controls["robot_difference_threshold"]),np.asarray(controls["proprio_mean"]),np.asarray(controls["proprio_std"]))
    ordinary=ordinary_pairs(data,validation_demos); cf_all=make_counterfactual_set(data,epsilon,True); cf=subset_counterfactual(cf_all,np.isin(cf_all.demo_id,validation_demos)); original_pairs=pairset_for_counterfactual(cf,False); alternate_pairs=pairset_for_counterfactual(cf,True)
    cross_all=make_cross_demo_controls(data,epsilon,int(protocol["validation"]["cross_demo_points"]),float(protocol["validation"]["cross_demo_clear_margin_min"])); matched=subset_pairset(cross_all["matched"],cross_all["matched"].demo_id==34); different=subset_pairset(cross_all["different"],cross_all["different"].demo_id==34)
    ref_train=referent_rows(data,train_demos); ref_val=referent_rows(data,validation_demos); m2_train=m2_pairs(data,train_demos,epsilon); m2_val=m2_pairs(data,validation_demos,epsilon); m2_cross=filter_m2_cross(data,different,float(protocol["validation"]["cross_demo_clear_margin_min"])); goal_train=state_goal_rows(data,train_demos); goal_val=state_goal_rows(data,validation_demos); goal_cross=filter_goal_cross(data,different)
    summaries={}; probes={}; weights={}; statistics_store={}; sidecar={}; training=protocol["training"]; diagnostics=protocol["diagnostic_probes"]
    for family in FAMILIES:
        token_array=np.array(np.load(ROOT/protocol["outputs"]["feature_directory"]/f"{family}.npy",mmap_mode="r"),dtype=np.float16,copy=True); statistics=fit_layer_statistics(token_array,data.language,train_mask); token_source=torch.from_numpy(token_array).to(device=device,dtype=torch.float32); statistics_store[family]={name:np.asarray(getattr(statistics,name)) for name in statistics.__dataclass_fields__}
        started=time.perf_counter(); model=train_pair_reader(token_array.shape[-1],"P1",token_source,data.language,training_pairs,statistics,int(training["epochs"]),int(training["batch_size"]),int(training["seed"]),device)
        if parameter_count(model)>int(protocol["reader"]["parameter_max"]): raise RuntimeError("parameter ceiling exceeded")
        ordinary_logits=predict_pair_logits(model,token_source,data.language,ordinary,statistics,device,batch_size=int(training["batch_size"])); original_logits=predict_pair_logits(model,token_source,data.language,original_pairs,statistics,device,batch_size=int(training["batch_size"])); alternate_logits=predict_pair_logits(model,token_source,data.language,alternate_pairs,statistics,device,batch_size=int(training["batch_size"])); local=ordinary.gap<=8
        temperature=fit_temperature(np.concatenate((ordinary_logits[local],original_logits,alternate_logits)),np.concatenate((ordinary.label[local],cf.original_label,cf.alternate_label)),training["temperature_grid"])
        ordinary_probability=sigmoid(ordinary_logits*temperature); original_probability=sigmoid(original_logits*temperature); alternate_probability=sigmoid(alternate_logits*temperature); matched_probability=sigmoid(predict_pair_logits(model,token_source,data.language,matched,statistics,device,batch_size=int(training["batch_size"]))*temperature); different_probability=sigmoid(predict_pair_logits(model,token_source,data.language,different,statistics,device,batch_size=int(training["batch_size"]))*temperature)
        summary=main_metrics(ordinary_probability,ordinary,original_probability,alternate_probability,cf,matched_probability,matched,different_probability,different); original_attention=attention_for(model,token_source,data.language,cf.a,cf.original_language,statistics,device,batch_size=int(training["batch_size"])); alternate_attention=attention_for(model,token_source,data.language,cf.a,cf.alternate_language,statistics,device,batch_size=int(training["batch_size"])); summary.update({"family":family,"temperature":float(temperature),"trainable_parameters":parameter_count(model),"training_seconds":float(time.perf_counter()-started),"attention_diagnostics":attention_diagnostics(original_attention,alternate_attention)}); summaries[family]=summary; weights[f"{family}/P1"]=model.state_dict()
        for name,value in (("ordinary",ordinary_probability),("cf_original",original_probability),("cf_alternate",alternate_probability),("cross_match",matched_probability),("cross_difference",different_probability)): sidecar[f"{family}_{name}_probability"]=value.astype(np.float32)
        sidecar[f"{family}_original_attention"]=original_attention.astype(np.float16); sidecar[f"{family}_alternate_attention"]=alternate_attention.astype(np.float16)
        seed=int(diagnostics["seed"]); ref_model=train_referent_probe(token_array.shape[-1],token_source,data.language,ref_train["frame"],ref_train["language_a"],ref_train["language_b"],ref_train["label"],statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),seed,device); ref_logits=predict_referent_logits(ref_model,token_source,data.language,ref_val["frame"],ref_val["language_a"],ref_val["language_b"],statistics,device,batch_size=int(diagnostics["batch_size"])); m2_model=train_m2_probe(token_array.shape[-1],token_source,data.language,m2_train,statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),seed+1,device); m2_logits=predict_m2_logits(m2_model,token_source,data.language,m2_val,statistics,device,batch_size=int(diagnostics["batch_size"])); m2_cross_logits=predict_m2_logits(m2_model,token_source,data.language,m2_cross,statistics,device,batch_size=int(diagnostics["batch_size"])); goal_model=train_goal_probe(token_array.shape[-1],token_source,data.language,goal_train["frame"],goal_train["language"],goal_train["label"],statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),seed+2,device); goal_logits=predict_goal_logits(goal_model,token_source,data.language,goal_val["frame"],goal_val["language"],statistics,device,batch_size=int(diagnostics["batch_size"])); goal_a=predict_goal_logits(goal_model,token_source,data.language,goal_cross.a,goal_cross.language_id,statistics,device,batch_size=int(diagnostics["batch_size"])); goal_b=predict_goal_logits(goal_model,token_source,data.language,goal_cross.b,goal_cross.language_id,statistics,device,batch_size=int(diagnostics["batch_size"]))
        probes[family]={"referent_change":binary_metrics(ref_logits,ref_val["label"]),"M2_ordering":{"ordinary":binary_metrics(m2_logits,m2_val.label==1),"cross_demo":binary_metrics(m2_cross_logits,m2_cross.label==1)},"official_goal":{"state":binary_metrics(goal_logits,goal_val["label"]),"cross_demo":binary_metrics(goal_b-goal_a,goal_cross.label==1)},"parameter_counts":{"referent":parameter_count(ref_model),"M2":parameter_count(m2_model),"goal":parameter_count(goal_model)}}
        weights[f"{family}/referent"]=ref_model.state_dict(); weights[f"{family}/M2"]=m2_model.state_dict(); weights[f"{family}/goal"]=goal_model.state_dict(); sidecar[f"{family}_referent_logits"]=ref_logits.astype(np.float32); sidecar[f"{family}_M2_logits"]=m2_logits.astype(np.float32); sidecar[f"{family}_M2_cross_logits"]=m2_cross_logits.astype(np.float32); sidecar[f"{family}_goal_logits"]=goal_logits.astype(np.float32); sidecar[f"{family}_goal_cross_logits"]=(goal_b-goal_a).astype(np.float32)
        print(json.dumps({"stage_e_family_complete":family,"summary":summary,"probes":probes[family]}),flush=True); del token_source,token_array
        if device.type=="cuda": torch.cuda.empty_cache()
    baseline=protocol["baseline"]; gates={}
    for family,value in summaries.items():
        cf_delta=100*(value["language_counterfactual"]["teacher_change_agreement"]-float(baseline["teacher_change"])); diff_delta=100*(value["cross_demo_semantic_difference"]["accuracy"]-float(baseline["cross_difference"])); match_improvement=float(baseline["cross_match"])-value["cross_demo_semantic_match"]["mean_abs_from_half"]; ordinary_delta=100*(value["ordinary_local_non_tie"]["accuracy"]-float(baseline["ordinary"])); complete=bool(cf_delta>=5 and (diff_delta>=7 or match_improvement>=0.03) and ordinary_delta>=-2); gates[family]={"teacher_change_increment_points":cf_delta,"cross_demo_difference_increment_points":diff_delta,"cross_demo_match_improvement":match_improvement,"ordinary_increment_points":ordinary_delta,"complete":complete}
    priority=sorted(FAMILIES,key=lambda family:(-summaries[family]["language_counterfactual"]["teacher_change_agreement"],-summaries[family]["cross_demo_semantic_difference"]["accuracy"],summaries[family]["cross_demo_semantic_match"]["mean_abs_from_half"],-summaries[family]["ordinary_local_non_tie"]["accuracy"],summaries[family]["language_counterfactual"]["same_referent"]["mean_abs_probability_change"],extraction["families"][family]["shape"][-1],FAMILIES.index(family)))
    selected=next((family for family in priority if gates[family]["complete"]),None); passed=selected is not None; labels=protocol["decision_mapping"]["pass_labels" if passed else "fail_labels"]
    weights_path=ROOT/protocol["outputs"]["weights"]; weights_path.parent.mkdir(parents=True,exist_ok=True); torch.save({"version":VERSION,"models":weights,"statistics":statistics_store,"selected_encoder":selected},weights_path)
    sidecar.update({"ordinary_label":ordinary.label,"cf_original_label":cf.original_label,"cf_alternate_label":cf.alternate_label,"ref_label":ref_val["label"].astype(np.int8),"M2_label":m2_val.label,"M2_cross_label":m2_cross.label,"goal_label":goal_val["label"].astype(np.int8),"goal_cross_label":goal_cross.label}); sidecar_path=ROOT/protocol["outputs"]["sidecar"]; atomic_npz(sidecar_path,sidecar)
    probe_result={"schema_version":1,"stage":"E","partition":{"fit":[31,32,33],"cross_demo_validation":[34]},"probes":probes,"best":{"referent":max(FAMILIES,key=lambda x:probes[x]["referent_change"]["accuracy"]),"M2":max(FAMILIES,key=lambda x:probes[x]["M2_ordering"]["cross_demo"]["accuracy"]),"official_goal":max(FAMILIES,key=lambda x:probes[x]["official_goal"]["cross_demo"]["accuracy"])},"selection_authority":False,"finite":True}; atomic_json(ROOT/protocol["outputs"]["information_probes"],probe_result)
    result={"schema_version":1,"stage":"E","mode":"calibration_only_alternate_frozen_encoder_selection","protocol_version":VERSION,"protocol_commit":last_commit(ROOT/"configs/alternate_encoder_protocol.json"),"partition":{"fit":[31,32,33],"selection":[34],"evaluation_opened":False},"extraction":extraction,"training_pairs":len(training_pairs.a),"validation_rows":{"ordinary":len(ordinary.a),"counterfactual":len(cf.a),"cross_match":len(matched.a),"cross_difference":len(different.a)},"families":summaries,"selection_order":priority,"material_gates":gates,"selected_encoder":selected,"selected_visual_layer":None if selected is None else json.loads((ROOT/protocol["visual"]["contracts"]).read_text())["families"][selected]["token_source"],"complete_gate":passed,"labels":labels,"authorizations":[protocol["decision_mapping"]["pass_authorization"]] if passed else protocol["decision_mapping"]["fail_authorizations"],"continuing_blocks":protocol["decision_mapping"]["continuing_blocks"],"controller_authorization":False,"artifacts":{"weights_sha256":sha256(weights_path),"sidecar_sha256":sha256(sidecar_path),"information_probes_sha256":sha256(ROOT/protocol["outputs"]["information_probes"]),"feature_manifest_sha256":sha256(feature_manifest_path),"access_ledger_sha256":sha256(ROOT/protocol["outputs"]["access_ledger"])},"finite":True,"forbidden_activity":{"demo36_39_opened":False,"demo40_49_numeric_access":0,"demo40_49_metadata_access":0,"encoder_training":False,"stage_v":False,"candidate_bridge":False,"world_model":False,"controller":False}}
    if not finite_tree(result) or not all(np.isfinite(v).all() for v in sidecar.values()): raise FloatingPointError("non-finite Stage-E result")
    atomic_json(ROOT/protocol["outputs"]["calibration"],result); decision={"schema_version":1,"stage":"E","selected_visual_encoder":selected,"selected_visual_layer":result["selected_visual_layer"],"mechanism_labels":labels,"next_stage_authorizations":result["authorizations"],"semantic_value_redistillation_authorized":passed,"real_candidate_value_bridge_authorized":False,"real_candidate_value_bridge_executed":False,"world_model_authorization":"BLOCKED","controller_authorization":False,"stop_chain":not passed,"source_calibration_sha256":sha256(ROOT/protocol["outputs"]["calibration"])}; atomic_json(ROOT/protocol["outputs"]["decision"],decision); return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=ROOT/"configs/alternate_encoder_protocol.json"); parser.add_argument("--phase",choices=("extract","calibrate","all"),default="all"); parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); args=parser.parse_args(); protocol=validate_protocol(args.protocol.resolve()); device=torch.device(args.device)
    if args.phase in ("extract","all"): extract_features(protocol,device)
    if args.phase in ("calibrate","all"):
        result=calibrate(protocol,device); print(json.dumps({"stage_e_complete":result["complete_gate"],"selected":result["selected_encoder"],"authorizations":result["authorizations"]}),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
