#!/usr/bin/env python3
"""Extract fixed CLIP internals and execute calibration-only Stage I."""

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
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.clip_internal_layer import (  # noqa: E402
    LAYERS, READERS, InternalLayerPairReader, LayerStatistics,
    ReferentChangeProbe, M2OrderingProbe, GoalBooleanProbe,
    attention_diagnostics, attention_for, binary_metrics,
    fit_layer_statistics, fixed_patch_coordinates, parameter_count,
    predict_goal_logits, predict_m2_logits, predict_pair_logits,
    predict_referent_logits, train_goal_probe, train_m2_probe,
    train_pair_reader, train_referent_probe,
)
from src.probes.semantic_value_distillation import (  # noqa: E402
    CounterfactualSet, PairSet, compare_pairs, cross_demo_metrics,
    fit_temperature, load_partition, make_counterfactual_set,
    make_cross_demo_controls, make_state_targets, make_training_pairs,
    non_tie_metrics, ordinary_pairs, rank_auc, sigmoid,
)


VERSION = "clip-internal-spatial-layer-audit-v1"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != VERSION:
        raise ValueError("unexpected Stage-I protocol")
    if protocol.get("status") != "PREREGISTERED_BEFORE_STAGE_I_FEATURE_EXTRACTION_OR_TRAINING":
        raise ValueError("Stage-I protocol is not preregistered")
    if not last_commit(path):
        raise RuntimeError("Stage-I protocol must be committed before execution")
    contract_path = ROOT / protocol["visual"]["extraction_contract"]
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "PREREGISTERED_BEFORE_STAGE_I_FEATURE_EXTRACTION_OR_TRAINING":
        raise ValueError("extraction contract is not preregistered")
    for row in protocol["frozen_files"]:
        actual = sha256(ROOT / row["path"])
        if actual != row["sha256"]:
            raise RuntimeError(f"frozen hash mismatch: {row['path']} {actual}")
    return protocol


def clip_snapshot() -> Path:
    path = ROOT / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots" / MODEL_REVISION
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def encode_internal(
    model: CLIPModel, processor: CLIPProcessor, images: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    pixels = processor(
        images=[np.ascontiguousarray(image) for image in images], return_tensors="pt"
    )["pixel_values"].to(device)
    with autocast(device):
        output = model.vision_model(
            pixel_values=pixels, output_hidden_states=True, return_dict=True
        )
        hidden = output.hidden_states
        if hidden is None or len(hidden) != 13:
            raise RuntimeError("installed CLIP did not return 13 hidden states")
        raw_embedding = model.vision_model.embeddings(pixels)
        pre_layernorm = model.vision_model.pre_layrnorm(raw_embedding)
        preln_max_abs = float((pre_layernorm - hidden[0]).abs().max().float().cpu())
        final = hidden[12][:, 1:, :]
        post = model.vision_model.post_layernorm(final)
        projected = F.normalize(model.visual_projection(post).float(), dim=-1)
        global_feature = F.normalize(
            model.visual_projection(output.pooler_output).float(), dim=-1
        )
        values = {
            "L0": hidden[0][:, 1:, :],
            "L3": hidden[3][:, 1:, :],
            "L6": hidden[6][:, 1:, :],
            "L9": hidden[9][:, 1:, :],
            "L12-pre": final,
            "L12-postLN": post,
            "L12-proj": projected,
        }
    arrays = {name: value.float().cpu().numpy().astype(np.float16) for name, value in values.items()}
    audit = {
        "raw_embedding_shape": list(raw_embedding.shape),
        "pre_layernorm_shape": list(pre_layernorm.shape),
        "hidden_state_count": len(hidden),
        "prelayernorm_vs_hidden0_max_abs": preln_max_abs,
    }
    return global_feature.cpu().numpy().astype(np.float16), arrays, audit


def extract_features(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    feature_dir = ROOT / protocol["outputs"]["feature_directory"]
    if feature_dir.exists() and any(feature_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Stage-I features: {feature_dir}")
    feature_dir.mkdir(parents=True, exist_ok=True)
    data, feature_ledger = load_partition(
        ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz",
        ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5",
        "calibration",
    )
    manifest = json.loads((ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/feature_manifest.json").read_text())
    task_files = {int(row["task_id"]): ROOT / row["source_hdf5_relative_path"] for row in manifest["tasks"]}
    guard = AllowedDemoGuard("calibration", task_files, ["obs/agentview_rgb"])
    processor = CLIPProcessor.from_pretrained(clip_snapshot(), local_files_only=True)
    model = CLIPModel.from_pretrained(clip_snapshot(), local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimensions = {"L0": 768, "L3": 768, "L6": 768, "L9": 768,
                  "L12-pre": 768, "L12-postLN": 768, "L12-proj": 512}
    temporary: dict[str, Path] = {}
    memmaps: dict[str, np.memmap] = {}
    for layer in LAYERS:
        temp = feature_dir / f"{layer}.npy.part"
        temporary[layer] = temp
        memmaps[layer] = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float16, shape=(len(data.vision), 49, dimensions[layer])
        )
    cursor = 0
    batch_size = 64
    min_global_cosine, max_global_abs = 1.0, 0.0
    repeated_exact = True
    repeated = False
    implementation_audit: dict[str, Any] | None = None
    for task in range(10):
        for demo in guard.allowed_demo_ids:
            images = guard.read_dataset(task, demo, "obs/agentview_rgb")
            if images.shape[1:] != (128, 128, 3) or images.dtype != np.uint8:
                raise ValueError("unexpected RGB schema")
            count = len(images); frame_slice = slice(cursor, cursor + count)
            if not (np.all(data.task_id[frame_slice] == task)
                    and np.all(data.demo_id[frame_slice] == demo)
                    and np.array_equal(data.timestep[frame_slice], np.arange(count))):
                raise ValueError("RGB/T1 identity mismatch")
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                global_feature, layers, audit = encode_internal(
                    model, processor, images[start:stop], device
                )
                implementation_audit = implementation_audit or audit
                if not repeated:
                    repeat_global, repeat_layers, _ = encode_internal(
                        model, processor, images[start:stop], device
                    )
                    repeated_exact &= np.array_equal(global_feature, repeat_global)
                    repeated_exact &= all(np.array_equal(layers[name], repeat_layers[name]) for name in LAYERS)
                    repeated = True
                absolute = slice(cursor + start, cursor + stop)
                for layer in LAYERS:
                    expected = (stop - start, 49, dimensions[layer])
                    if layers[layer].shape != expected or not np.isfinite(layers[layer]).all():
                        raise ValueError(f"invalid {layer} feature tensor")
                    memmaps[layer][absolute] = layers[layer]
                reference = data.vision[frame_slice][start:stop].astype(np.float32)
                candidate = global_feature.astype(np.float32)
                cosine = np.sum(candidate * reference, axis=-1) / np.maximum(
                    np.linalg.norm(candidate, axis=-1) * np.linalg.norm(reference, axis=-1), 1e-12
                )
                min_global_cosine = min(min_global_cosine, float(cosine.min()))
                max_global_abs = max(max_global_abs, float(np.abs(candidate - reference).max()))
            cursor += count
            print(json.dumps({"stage_i_extraction": {"task": task, "demo": demo, "frames": count}}), flush=True)
    if cursor != len(data.vision):
        raise ValueError("frame total mismatch")
    for value in memmaps.values():
        value.flush()
    del memmaps
    final_paths = {}
    for layer in LAYERS:
        final = feature_dir / f"{layer}.npy"
        os.replace(temporary[layer], final)
        final_paths[layer] = final
    previous = np.load(ROOT / "results/spatial_visual_representation/patch_features_calibration.npz")["patch"]
    projected = np.load(final_paths["L12-proj"], mmap_mode="r")
    projected_parity = bool(np.array_equal(projected, previous))
    projected_max_abs = float(np.max(np.abs(projected.astype(np.float32) - previous.astype(np.float32))))
    guard.assert_clean()
    ledger = {
        "stage": "I", "partition": "calibration_only",
        "source_rgb": guard.ledger.as_dict(), "frozen_feature_store": feature_ledger,
        "opened_demo_ids": sorted(set(guard.ledger.opened_demo_ids) | set(feature_ledger["opened_demo_ids"])),
        "stage_i_not_opened_demo_ids": [36, 37, 38, 39],
        "forbidden_demo_ids": list(range(40, 50)),
        "forbidden_numeric_access": 0, "forbidden_metadata_access": 0,
    }
    atomic_json(ROOT / protocol["outputs"]["access_ledger"], ledger)
    complete = bool(
        repeated_exact and min_global_cosine >= 0.9999 and max_global_abs <= 0.002
        and projected_parity and projected_max_abs == 0.0
        and implementation_audit is not None
        and implementation_audit["prelayernorm_vs_hidden0_max_abs"] == 0.0
        and guard.ledger.as_dict()["forbidden_access_count"] == 0
        and feature_ledger["forbidden_access_count"] == 0
        and not any(parameter.requires_grad for parameter in model.parameters())
    )
    result = {
        "schema_version": 1, "stage": "I", "mode": "fixed_internal_feature_extraction",
        "frames": len(data.vision), "layers": {
            layer: {"shape": [len(data.vision), 49, dimensions[layer]], "dtype": "float16",
                    "sha256": sha256(final_paths[layer])} for layer in LAYERS
        },
        "coordinates": fixed_patch_coordinates().astype(float).tolist(),
        "implementation_audit": implementation_audit,
        "repeated_sample_float16_exact": repeated_exact,
        "global_parity_min_cosine": min_global_cosine,
        "global_parity_max_abs": max_global_abs,
        "prior_L12_proj_float16_exact": projected_parity,
        "prior_L12_proj_max_abs": projected_max_abs,
        "trainable_encoder_parameters": 0,
        "access_ledger_sha256": sha256(ROOT / protocol["outputs"]["access_ledger"]),
        "complete_gate": complete,
    }
    atomic_json(ROOT / protocol["outputs"]["feature_manifest"], result)
    if not complete:
        raise RuntimeError("Stage-I extraction integrity failed")
    return result


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
    non_tie = local & (ordinary.label != 0); ties = local & (ordinary.label == 0)
    changed = counterfactual.original_label != counterfactual.alternate_label
    direction = counterfactual.alternate_label.astype(float) - counterfactual.original_label.astype(float)
    agreement = (alternate_probability - original_probability) * direction > 0
    reversal = counterfactual.original_label * counterfactual.alternate_label == -1
    original_correct = np.sign(original_probability - 0.5) == counterfactual.original_label
    alternate_correct = np.sign(alternate_probability - 0.5) == counterfactual.alternate_label
    same = counterfactual.same_referent
    return {
        "ordinary_local_non_tie": non_tie_metrics(ordinary_probability[non_tie], ordinary.label[non_tie]),
        "ordinary_local_tie": {
            "rows": int(ties.sum()),
            "mean_abs_from_half": float(np.abs(ordinary_probability[ties] - 0.5).mean()),
            "median_abs_from_half": float(np.median(np.abs(ordinary_probability[ties] - 0.5))),
        },
        "language_counterfactual": {
            "eligible_rows": int(len(changed)), "teacher_changed_rows": int(changed.sum()),
            "teacher_change_agreement": float(agreement[changed].mean()),
            "true_reversal": {"rows": int(reversal.sum()),
                              "both_correct_rate": float((original_correct & alternate_correct)[reversal].mean())},
            "same_referent": {
                "rows": int(same.sum()),
                "mean_abs_probability_change": float(np.abs(alternate_probability[same] - original_probability[same]).mean()),
                "false_preference_flip_rate": float((np.sign(alternate_probability[same] - 0.5) != np.sign(original_probability[same] - 0.5)).mean()),
            },
        },
        "cross_demo_semantic_match": cross_demo_metrics(matched_probability, matched),
        "cross_demo_semantic_difference": cross_demo_metrics(different_probability, different),
    }


def referent_rows(data: Any, demos: tuple[int, ...], points: int = 8) -> dict[str, np.ndarray]:
    rows=[]
    for episode in np.unique(data.episode_id):
        ids=np.flatnonzero(data.episode_id==episode)
        demo=int(data.demo_id[ids[0]])
        if demo not in demos: continue
        languages=np.flatnonzero(data.ground_status[int(episode)]==1)
        frames=ids[np.unique(np.linspace(0,len(ids)-1,points,dtype=np.int64))]
        for frame in frames:
            for left in range(len(languages)):
                for right in range(left+1,len(languages)):
                    a,b=int(languages[left]),int(languages[right])
                    label=int(data.ground_referent[int(episode),a] != data.ground_referent[int(episode),b])
                    rows.append((int(frame),a,b,label,int(data.task_id[frame]),demo))
    matrix=np.asarray(rows,dtype=np.int64)
    return {"frame":matrix[:,0],"language_a":matrix[:,1],"language_b":matrix[:,2],
            "label":matrix[:,3].astype(np.bool_),"task":matrix[:,4],"demo":matrix[:,5]}


def m2_pairs(data: Any, demos: tuple[int, ...], epsilon: float) -> PairSet:
    base=ordinary_pairs(data,demos); local=base.gap<=8
    episode=data.episode_id[base.a]; ref=data.ground_referent[episode,base.language_id]
    same_goal=data.goal[base.a,ref]==data.goal[base.b,ref]
    delta=data.margin[base.b,ref]-data.margin[base.a,ref]
    keep=local & same_goal & (np.abs(delta)>epsilon)
    a,b,language=base.a[keep],base.b[keep],base.language_id[keep]
    label=np.sign(delta[keep]).astype(np.int8); task,demo=base.task_id[keep],base.demo_id[keep]
    gap=base.gap[keep]; code=base.gap_code[keep]
    return PairSet(np.concatenate((a,b)),np.concatenate((b,a)),np.concatenate((language,language)),
                   np.concatenate((label,-label)),np.concatenate((task,task)),np.concatenate((demo,demo)),
                   np.concatenate((gap,-gap)),np.concatenate((code,code)),
                   np.full(2*len(a),"M2_order",dtype="U24"))


def filter_m2_cross(data: Any, pairs: PairSet, epsilon: float) -> PairSet:
    episode_a=data.episode_id[pairs.a]; episode_b=data.episode_id[pairs.b]
    ref_a=data.ground_referent[episode_a,pairs.language_id]
    ref_b=data.ground_referent[episode_b,pairs.language_id]
    same_goal=data.goal[pairs.a,ref_a]==data.goal[pairs.b,ref_b]
    delta=data.margin[pairs.b,ref_b]-data.margin[pairs.a,ref_a]
    keep=same_goal & (np.abs(delta)>epsilon)
    selected=subset_pairset(pairs,keep)
    labels=np.sign(delta[keep]).astype(np.int8)
    return PairSet(selected.a,selected.b,selected.language_id,labels,selected.task_id,
                   selected.demo_id,selected.gap,selected.gap_code,selected.pair_type)


def filter_goal_cross(data: Any, pairs: PairSet) -> PairSet:
    episode_a=data.episode_id[pairs.a]; episode_b=data.episode_id[pairs.b]
    ref_a=data.ground_referent[episode_a,pairs.language_id]
    ref_b=data.ground_referent[episode_b,pairs.language_id]
    ga=data.goal[pairs.a,ref_a]; gb=data.goal[pairs.b,ref_b]; keep=ga!=gb
    selected=subset_pairset(pairs,keep); label=np.where(gb[keep],1,-1).astype(np.int8)
    return PairSet(selected.a,selected.b,selected.language_id,label,selected.task_id,
                   selected.demo_id,selected.gap,selected.gap_code,selected.pair_type)


def state_goal_rows(data: Any, demos: tuple[int, ...]) -> dict[str, np.ndarray]:
    frame,language,goal,_=make_state_targets(data,demos)
    return {"frame":frame,"language":language,"label":goal.astype(np.bool_)}


def diagnostic_metrics(logits: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    return binary_metrics(logits,label)


def calibrate(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    feature_manifest_path=ROOT/protocol["outputs"]["feature_manifest"]
    extraction=json.loads(feature_manifest_path.read_text())
    if not extraction.get("complete_gate"): raise RuntimeError("Stage-I extraction not valid")
    data,feature_ledger=load_partition(ROOT/"results/t1_semantic_teacher/calibration_sidecar.npz",
        ROOT/"artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5","calibration")
    if feature_ledger["forbidden_access_count"]: raise RuntimeError("forbidden feature access")
    train_demos=tuple(protocol["partitions"]["fit_demo_ids"]); validation_demos=tuple(protocol["partitions"]["selection_demo_ids"])
    train_mask=np.isin(data.demo_id,train_demos)
    definition=json.loads((ROOT/"configs/t1_teacher_frozen_definition.json").read_text()); controls=definition["controls"]
    sampler=json.loads((ROOT/protocol["training"]["pair_sampler"]).read_text())
    epsilon=float(protocol["frozen_teacher"]["epsilon_goal"])
    training_pairs=make_training_pairs(data,train_demos,epsilon,sampler,float(controls["robot_match_threshold"]),float(controls["robot_difference_threshold"]),np.asarray(controls["proprio_mean"]),np.asarray(controls["proprio_std"]))
    ordinary=ordinary_pairs(data,validation_demos)
    counterfactual_all=make_counterfactual_set(data,epsilon,True)
    counterfactual=subset_counterfactual(counterfactual_all,np.isin(counterfactual_all.demo_id,validation_demos))
    original_pairs=pairset_for_counterfactual(counterfactual,False); alternate_pairs=pairset_for_counterfactual(counterfactual,True)
    cross_all=make_cross_demo_controls(data,epsilon,int(protocol["validation"]["cross_demo_points"]),float(protocol["validation"]["cross_demo_clear_margin_min"]))
    matched=subset_pairset(cross_all["matched"],cross_all["matched"].demo_id==34)
    different=subset_pairset(cross_all["different"],cross_all["different"].demo_id==34)
    ref_train=referent_rows(data,train_demos); ref_val=referent_rows(data,validation_demos)
    m2_train=m2_pairs(data,train_demos,epsilon); m2_val=m2_pairs(data,validation_demos,epsilon)
    m2_cross=filter_m2_cross(data,different,float(protocol["validation"]["cross_demo_clear_margin_min"]))
    goal_train=state_goal_rows(data,train_demos); goal_val=state_goal_rows(data,validation_demos)
    goal_cross=filter_goal_cross(data,different)
    training=protocol["training"]; diagnostics=protocol["diagnostic_probes"]
    summaries={}; probe_results={}; saved_weights={}; sidecar={}
    statistics_store={}
    feature_dir=ROOT/protocol["outputs"]["feature_directory"]
    for layer_index,layer in enumerate(LAYERS):
        tokens=np.load(feature_dir/f"{layer}.npy",mmap_mode="r")
        statistics=fit_layer_statistics(tokens,data.language,train_mask)
        statistics_store[layer]={name:np.asarray(getattr(statistics,name)) for name in statistics.__dataclass_fields__}
        summaries[layer]={}
        for family in READERS:
            started=time.perf_counter(); seed=int(training["seed"])+(0 if family=="P0" else 1)
            model=train_pair_reader(tokens.shape[-1],family,tokens,data.language,training_pairs,statistics,int(training["epochs"]),int(training["batch_size"]),seed,device)
            if parameter_count(model)>int(protocol["readers"]["parameter_max"]): raise RuntimeError("parameter ceiling exceeded")
            ordinary_logits=predict_pair_logits(model,tokens,data.language,ordinary,statistics,device)
            original_logits=predict_pair_logits(model,tokens,data.language,original_pairs,statistics,device)
            alternate_logits=predict_pair_logits(model,tokens,data.language,alternate_pairs,statistics,device)
            local=ordinary.gap<=8
            temperature=fit_temperature(np.concatenate((ordinary_logits[local],original_logits,alternate_logits)),np.concatenate((ordinary.label[local],counterfactual.original_label,counterfactual.alternate_label)),training["temperature_grid"])
            ordinary_probability=sigmoid(ordinary_logits*temperature); original_probability=sigmoid(original_logits*temperature); alternate_probability=sigmoid(alternate_logits*temperature)
            matched_probability=sigmoid(predict_pair_logits(model,tokens,data.language,matched,statistics,device)*temperature)
            different_probability=sigmoid(predict_pair_logits(model,tokens,data.language,different,statistics,device)*temperature)
            summary=main_metrics(ordinary_probability,ordinary,original_probability,alternate_probability,counterfactual,matched_probability,matched,different_probability,different)
            summary.update({"family":family,"layer":layer,"temperature":float(temperature),"trainable_parameters":parameter_count(model),"training_seconds":float(time.perf_counter()-started)})
            if family=="P1":
                original_attention=attention_for(model,tokens,data.language,counterfactual.a,counterfactual.original_language,statistics,device)
                alternate_attention=attention_for(model,tokens,data.language,counterfactual.a,counterfactual.alternate_language,statistics,device)
                summary["attention_diagnostics"]=attention_diagnostics(original_attention,alternate_attention)
                sidecar[f"{layer}_P1_original_attention"]=original_attention.astype(np.float16)
                sidecar[f"{layer}_P1_alternate_attention"]=alternate_attention.astype(np.float16)
            summaries[layer][family]=summary; saved_weights[f"{layer}/{family}"]=model.state_dict()
            for name,value in (("ordinary",ordinary_probability),("cf_original",original_probability),("cf_alternate",alternate_probability),("cross_match",matched_probability),("cross_difference",different_probability)):
                sidecar[f"{layer}_{family}_{name}_probability"]=value.astype(np.float32)
            print(json.dumps({"stage_i_reader_complete":f"{layer}/{family}","summary":summary}),flush=True)
        diag_seed=int(diagnostics["seed"])
        referent_model=train_referent_probe(tokens.shape[-1],tokens,data.language,ref_train["frame"],ref_train["language_a"],ref_train["language_b"],ref_train["label"],statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),diag_seed,device)
        ref_logits=predict_referent_logits(referent_model,tokens,data.language,ref_val["frame"],ref_val["language_a"],ref_val["language_b"],statistics,device)
        m2_model=train_m2_probe(tokens.shape[-1],tokens,data.language,m2_train,statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),diag_seed+1,device)
        m2_logits=predict_m2_logits(m2_model,tokens,data.language,m2_val,statistics,device); m2_cross_logits=predict_m2_logits(m2_model,tokens,data.language,m2_cross,statistics,device)
        goal_model=train_goal_probe(tokens.shape[-1],tokens,data.language,goal_train["frame"],goal_train["language"],goal_train["label"],statistics,int(diagnostics["epochs"]),int(diagnostics["batch_size"]),diag_seed+2,device)
        goal_logits=predict_goal_logits(goal_model,tokens,data.language,goal_val["frame"],goal_val["language"],statistics,device)
        goal_a=predict_goal_logits(goal_model,tokens,data.language,goal_cross.a,goal_cross.language_id,statistics,device); goal_b=predict_goal_logits(goal_model,tokens,data.language,goal_cross.b,goal_cross.language_id,statistics,device)
        probe_results[layer]={
            "referent_change": {**diagnostic_metrics(ref_logits,ref_val["label"]),"construction":"same-scene language pair; same-versus-different persistent physical referent; demo34 cross-demo validation"},
            "M2_ordering": {"ordinary":diagnostic_metrics(m2_logits,m2_val.label==1),"cross_demo":diagnostic_metrics(m2_cross_logits,m2_cross.label==1)},
            "official_goal": {"state":diagnostic_metrics(goal_logits,goal_val["label"]),"cross_demo":diagnostic_metrics(goal_b-goal_a,goal_cross.label==1)},
            "parameter_counts": {"referent":parameter_count(referent_model),"M2":parameter_count(m2_model),"goal":parameter_count(goal_model)},
        }
        saved_weights[f"{layer}/referent"]=referent_model.state_dict(); saved_weights[f"{layer}/M2"]=m2_model.state_dict(); saved_weights[f"{layer}/goal"]=goal_model.state_dict()
        sidecar[f"{layer}_referent_logits"]=ref_logits.astype(np.float32); sidecar[f"{layer}_M2_logits"]=m2_logits.astype(np.float32); sidecar[f"{layer}_M2_cross_logits"]=m2_cross_logits.astype(np.float32); sidecar[f"{layer}_goal_logits"]=goal_logits.astype(np.float32); sidecar[f"{layer}_goal_cross_logits"]=(goal_b-goal_a).astype(np.float32)
        print(json.dumps({"stage_i_probes_complete":layer,"summary":probe_results[layer]}),flush=True)
    priority=sorted(LAYERS,key=lambda layer:(-summaries[layer]["P1"]["language_counterfactual"]["teacher_change_agreement"],summaries[layer]["P1"]["cross_demo_semantic_match"]["mean_abs_from_half"],-summaries[layer]["P1"]["cross_demo_semantic_difference"]["accuracy"],-summaries[layer]["P1"]["ordinary_local_non_tie"]["accuracy"],summaries[layer]["P1"]["language_counterfactual"]["same_referent"]["mean_abs_probability_change"],summaries[layer]["P1"]["language_counterfactual"]["same_referent"]["false_preference_flip_rate"],summaries[layer]["P1"]["ordinary_local_tie"]["mean_abs_from_half"],-LAYERS.index(layer)))
    baseline=summaries["L12-proj"]["P1"]; gates={}
    for layer in LAYERS:
        value=summaries[layer]["P1"]
        cf=100*(value["language_counterfactual"]["teacher_change_agreement"]-baseline["language_counterfactual"]["teacher_change_agreement"])
        match=baseline["cross_demo_semantic_match"]["mean_abs_from_half"]-value["cross_demo_semantic_match"]["mean_abs_from_half"]
        diff=100*(value["cross_demo_semantic_difference"]["accuracy"]-baseline["cross_demo_semantic_difference"]["accuracy"])
        ordinary_delta=100*(value["ordinary_local_non_tie"]["accuracy"]-baseline["ordinary_local_non_tie"]["accuracy"])
        complete=bool(layer!="L12-proj" and cf>=4.0 and (match>=0.03 or diff>=7.0) and ordinary_delta>=-2.0)
        gates[layer]={"teacher_change_increment_points":cf,"cross_demo_match_improvement":match,"cross_demo_difference_increment_points":diff,"ordinary_increment_points":ordinary_delta,"complete":complete}
    selected=next((layer for layer in priority if gates[layer]["complete"]),None); passed=selected is not None
    representative=selected or priority[0]
    rp=probe_results[representative]
    referent_good=bool(rp["referent_change"]["accuracy"]>=0.65 and rp["referent_change"]["auroc"]>=0.65)
    m2_good=bool(rp["M2_ordering"]["ordinary"]["accuracy"]>=0.65 and rp["M2_ordering"]["cross_demo"]["accuracy"]>=0.65)
    goal_good=bool(rp["official_goal"]["state"]["auroc"]>=0.80 and rp["official_goal"]["cross_demo"]["accuracy"]>=0.60)
    interpretation="A" if referent_good and m2_good else ("B" if referent_good else ("D" if not m2_good and not goal_good else "C"))
    labels=protocol["decision_mapping"]["pass_labels" if passed else "fail_labels"]; authorization=protocol["decision_mapping"]["pass_authorization" if passed else "fail_authorization"]
    weights_path=ROOT/protocol["outputs"]["weights"]; weights_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"version":VERSION,"models":saved_weights,"statistics":statistics_store,"selected_layer":selected,"representative_layer":representative},weights_path)
    sidecar.update({"ordinary_a":ordinary.a,"ordinary_b":ordinary.b,"ordinary_label":ordinary.label,"cf_a":counterfactual.a,"cf_b":counterfactual.b,"cf_original_label":counterfactual.original_label,"cf_alternate_label":counterfactual.alternate_label,"ref_label":ref_val["label"].astype(np.int8),"M2_label":m2_val.label,"M2_cross_label":m2_cross.label,"goal_label":goal_val["label"].astype(np.int8),"goal_cross_label":goal_cross.label})
    sidecar_path=ROOT/protocol["outputs"]["sidecar"]; atomic_npz(sidecar_path,sidecar)
    probe_document={"schema_version":1,"stage":"I","partition":{"fit":[31,32,33],"cross_demo_validation":[34]},"probes":probe_results,"best_layers":{"referent":max(LAYERS,key=lambda x:probe_results[x]["referent_change"]["accuracy"]),"M2":max(LAYERS,key=lambda x:probe_results[x]["M2_ordering"]["cross_demo"]["accuracy"]),"official_goal":max(LAYERS,key=lambda x:probe_results[x]["official_goal"]["cross_demo"]["accuracy"])},"representative_layer":representative,"representative_good":{"referent":referent_good,"M2":m2_good,"official_goal":goal_good},"interpretation_case":interpretation,"deployment_input":False,"finite":True}
    atomic_json(ROOT/protocol["outputs"]["information_probes"],probe_document)
    result={"schema_version":1,"stage":"I","mode":"calibration_only_internal_layer_selection","protocol_version":VERSION,"protocol_commit":last_commit(ROOT/"configs/clip_internal_layer_protocol.json"),"partition":{"fit":[31,32,33],"selection":[34],"evaluation_opened":False},"extraction":extraction,"training_pairs":len(training_pairs.a),"validation_rows":{"ordinary":len(ordinary.a),"counterfactual":len(counterfactual.a),"cross_match":len(matched.a),"cross_difference":len(different.a),"referent":len(ref_val["frame"]),"M2":len(m2_val.a),"M2_cross":len(m2_cross.a),"goal_states":len(goal_val["frame"]),"goal_cross":len(goal_cross.a)},"layers":summaries,"selection_order":priority,"fair_baseline":"L12-proj/P1","material_gates":gates,"selected_layer":selected,"diagnostic_representative":representative,"diagnostic_interpretation_case":interpretation,"complete_gate":passed,"labels":labels,"authorization":authorization,"continuing_blocks":protocol["decision_mapping"]["continuing_blocks"],"controller_authorization":False,"artifacts":{"weights_sha256":sha256(weights_path),"sidecar_sha256":sha256(sidecar_path),"information_probes_sha256":sha256(ROOT/protocol["outputs"]["information_probes"]),"feature_manifest_sha256":sha256(feature_manifest_path),"access_ledger_sha256":sha256(ROOT/protocol["outputs"]["access_ledger"])},"implementation":{"probe_sha256":sha256(ROOT/"src/probes/clip_internal_layer.py"),"runner_sha256":sha256(Path(__file__))},"finite":True,"forbidden_activity":{"demo36_39_opened":False,"demo40_49_numeric_access":0,"demo40_49_metadata_access":0,"alternate_encoder":False,"stage_v":False,"candidate_bridge":False,"world_model":False,"controller":False}}
    if not finite_tree(result) or not all(np.isfinite(v).all() for v in sidecar.values()): raise FloatingPointError("non-finite Stage-I result")
    atomic_json(ROOT/protocol["outputs"]["calibration"],result)
    decision={"schema_version":1,"stage":"I","selected_clip_internal_layer":selected,"diagnostic_representative":representative,"mechanism_labels":labels,"next_stage_authorization":authorization,"semantic_value_redistillation_authorized":passed,"real_candidate_value_bridge_authorized":False,"real_candidate_value_bridge_executed":False,"world_model_authorization":"BLOCKED","controller_authorization":False,"source_calibration_sha256":sha256(ROOT/protocol["outputs"]["calibration"])}
    atomic_json(ROOT/protocol["outputs"]["decision"],decision)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=ROOT/"configs/clip_internal_layer_protocol.json"); parser.add_argument("--phase",choices=("extract","calibrate","all"),default="all"); parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); args=parser.parse_args()
    protocol=validate_protocol(args.protocol.resolve()); device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.phase in ("extract","all"): extract_features(protocol,device)
    if args.phase in ("calibrate","all"):
        result=calibrate(protocol,device); print(json.dumps({"stage_i_complete":result["complete_gate"],"selected":result["selected_layer"],"authorization":result["authorization"]}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
