#!/usr/bin/env python3
"""Run the independently frozen semantic value distillation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.semantic_value_distillation import (  # noqa: E402
    GAP_NAMES, LOCAL_GAP_CODES, CounterfactualSet, DirectSemanticPair,
    LanguageOnlyDiagnostic, PairSet, S1AbsoluteSemantic, ScalarSemanticPotential,
    block_draws, bootstrap_mean, ci95, cross_demo_metrics, cycle_rate,
    fit_standardizer, fit_temperature, load_partition, make_counterfactual_set,
    make_cross_demo_controls, make_state_targets, make_training_pairs,
    non_tie_metrics, ordinary_pairs, parameter_count, predict_logits, sigmoid,
    standardize, summarize_counterfactual, summarize_ordinary, terminal_prototypes,
    tie_metrics, train_direct, train_language_only, train_s1,
    train_scalar_potential, triplet_pairs,
)


VERSION = "semantic-value-distillation-v1"
FAMILIES = ("S0", "S1", "S1.5", "S2", "IMAGE_ONLY", "LANGUAGE_ONLY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
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
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def last_commit(path: Path) -> str:
    relative = path.relative_to(ROOT)
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(relative)], cwd=ROOT, text=True
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict): return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)): return bool(np.isfinite(value))
    return True


def validate_protocol(path: Path, phase: str) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != VERSION:
        raise ValueError("unexpected semantic-value protocol")
    if protocol.get("status") != "PREREGISTERED_BEFORE_CALIBRATION_TRAINING":
        raise ValueError("protocol is not preregistered")
    if not last_commit(path):
        raise RuntimeError("protocol must be committed before execution")
    for row in protocol["frozen_files"] + protocol["implementation"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"frozen hash mismatch: {row['path']}")
    if phase == "evaluation":
        freeze = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze.exists() or not last_commit(freeze):
            raise RuntimeError("committed calibration freeze is required")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(freeze.relative_to(ROOT))],
            cwd=ROOT, text=True,
        ).strip():
            raise RuntimeError("calibration freeze is dirty")
        frozen = json.loads(freeze.read_text())
        for key in ("weights", "calibration", "calibration_sidecar"):
            target = ROOT / protocol["outputs"][key]
            if sha256(target) != frozen["artifact_sha256"][str(target.relative_to(ROOT))]:
                raise RuntimeError(f"calibration artifact changed: {target}")
    return protocol


def pairset_for_cf(cf: CounterfactualSet, alternate: bool) -> PairSet:
    language = cf.alternate_language if alternate else cf.original_language
    label = cf.alternate_label if alternate else cf.original_label
    return PairSet(
        cf.a, cf.b, language, label, cf.task_id, cf.demo_id,
        np.zeros(len(cf.a), dtype=np.int64), np.zeros(len(cf.a), dtype=np.int8),
        np.full(len(cf.a), "language_counterfactual", dtype="U32"),
    )


def model_payload(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_models(
    data: Any,
    train_demos: tuple[int, ...],
    protocol: dict[str, Any],
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], PairSet]:
    mask = np.isin(data.demo_id, train_demos)
    visual_mean, visual_std = fit_standardizer(data.vision[mask])
    language_mean, language_std = fit_standardizer(data.language)
    vision = standardize(data.vision, visual_mean, visual_std)
    language = standardize(data.language, language_mean, language_std)
    definition = json.loads((ROOT / protocol["inputs"]["t1_definition"]).read_text())
    epsilon = float(definition["epsilon_goal"])
    controls = definition["controls"]
    sampler = json.loads((ROOT / protocol["inputs"]["pair_sampler"]).read_text())
    pairs = make_training_pairs(
        data, train_demos, epsilon, sampler,
        float(controls["robot_match_threshold"]), float(controls["robot_difference_threshold"]),
        np.asarray(controls["proprio_mean"]), np.asarray(controls["proprio_std"]),
    )
    frames, language_ids, goals, margins = make_state_targets(data, train_demos)
    training = protocol["training"]
    epochs, batch = int(training["epochs"]), int(training["batch_size"])
    seed = int(training["seed"]) + seed_offset
    s1, s1_target = train_s1(vision, language, frames, language_ids, goals, margins,
                             epochs, batch, seed + 1)
    s15 = train_scalar_potential(vision, language, pairs, epochs, batch, seed + 2)
    s2 = train_direct(vision, language, pairs, epochs, batch, seed + 3, True)
    image = train_direct(vision, language, pairs, epochs, batch, seed + 4, False)
    language_only = train_language_only(language, pairs, epochs, batch, seed + 5)
    models: dict[str, Any] = {
        "S1": s1, "S1.5": s15, "S2": s2,
        "IMAGE_ONLY": image, "LANGUAGE_ONLY": language_only,
    }
    stats = {
        "visual_mean": visual_mean, "visual_std": visual_std,
        "language_mean": language_mean, "language_std": language_std,
    }
    metadata = {
        "s1_target": s1_target,
        "parameters": {name: parameter_count(model) for name, model in models.items()},
        "training_pair_count": int(len(pairs.a)),
        "training_pair_type_counts": {
            name: int(np.sum(pairs.pair_type == name)) for name in np.unique(pairs.pair_type)
        },
        "state_language_target_count": int(len(frames)),
    }
    return models, stats, metadata, pairs


def raw_probabilities(
    data: Any,
    vision: np.ndarray,
    language: np.ndarray,
    models: dict[str, Any],
    prototypes: np.ndarray,
    pairs: PairSet,
    temperatures: dict[str, float] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    logits = {}
    probability = {}
    for family in FAMILIES:
        feature = data.vision if family == "S0" else vision
        logits[family] = predict_logits(
            family, models.get(family), feature, language, pairs, prototypes
        )
        scale = 1.0 if temperatures is None else float(temperatures[family])
        probability[family] = sigmoid(logits[family] * scale)
    return logits, probability


def fit_temperatures(
    protocol: dict[str, Any],
    ordinary_logits: dict[str, np.ndarray], ordinary: PairSet,
    original_logits: dict[str, np.ndarray], alternate_logits: dict[str, np.ndarray],
    cf: CounterfactualSet,
) -> dict[str, float]:
    local = np.isin(ordinary.gap_code, tuple(LOCAL_GAP_CODES))
    grid = protocol["calibration"]["temperature_grid"]
    # Every counterfactual application contributes both the original and
    # alternate language realization; duplicates are intentional supervision.
    temperatures = {}
    for family in FAMILIES:
        logits = np.concatenate((ordinary_logits[family][local],
                                 original_logits[family], alternate_logits[family]))
        labels = np.concatenate((ordinary.label[local], cf.original_label, cf.alternate_label))
        temperatures[family] = fit_temperature(logits, labels, grid)
    return temperatures


def family_summaries(
    data: Any, models: dict[str, Any], stats: dict[str, np.ndarray],
    prototypes: np.ndarray, temperatures: dict[str, float],
    demos: tuple[int, ...], protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, PairSet], CounterfactualSet]:
    vision = standardize(data.vision, stats["visual_mean"], stats["visual_std"])
    language = standardize(data.language, stats["language_mean"], stats["language_std"])
    ordinary = ordinary_pairs(data, demos)
    cf_all = make_counterfactual_set(data, float(protocol["teacher"]["epsilon_goal"]), True)
    cf_keep = np.isin(cf_all.demo_id, demos)
    cf = CounterfactualSet(*(getattr(cf_all, field)[cf_keep] for field in cf_all.__dataclass_fields__))
    original_pairs, alternate_pairs = pairset_for_cf(cf, False), pairset_for_cf(cf, True)
    _, ordinary_probability = raw_probabilities(data, vision, language, models, prototypes, ordinary, temperatures)
    _, original_probability = raw_probabilities(data, vision, language, models, prototypes, original_pairs, temperatures)
    _, alternate_probability = raw_probabilities(data, vision, language, models, prototypes, alternate_pairs, temperatures)
    controls = make_cross_demo_controls(
        data, float(protocol["teacher"]["epsilon_goal"]),
        int(protocol["controls"]["cross_demo_points"]),
        float(protocol["controls"]["cross_demo_clear_margin_min"]),
    )
    _, matched_probability = raw_probabilities(data, vision, language, models, prototypes, controls["matched"], temperatures)
    _, different_probability = raw_probabilities(data, vision, language, models, prototypes, controls["different"], temperatures)
    triplets, triplet_group = triplet_pairs(
        data, int(protocol["controls"]["triplet_points"]),
        float(protocol["teacher"]["epsilon_goal"]),
    )
    _, triplet_probability = raw_probabilities(data, vision, language, models, prototypes, triplets, temperatures)
    draws = block_draws(int(protocol["statistics"]["bootstrap_replicates"]),
                        int(protocol["statistics"]["bootstrap_seed"]))
    summaries = {}
    raw = {}
    for family in FAMILIES:
        cf_summary = summarize_counterfactual(
            original_probability[family], alternate_probability[family], cf, demos, draws
        )
        cf_agreement = cf_summary.pop("row_agreement")
        changed_mask = cf_summary.pop("changed_mask")
        summaries[family] = {
            "ordinary": summarize_ordinary(ordinary_probability[family], ordinary, demos, draws),
            "tie": tie_metrics(ordinary_probability[family], ordinary, demos, draws),
            "language_counterfactual": cf_summary,
            "cross_demo": {
                "semantic_match": cross_demo_metrics(matched_probability[family], controls["matched"]),
                "semantic_difference": cross_demo_metrics(different_probability[family], controls["different"]),
            },
            "transitivity": cycle_rate(triplet_probability[family], triplet_group),
        }
        raw[family] = {
            "ordinary_probability": ordinary_probability[family],
            "cf_original_probability": original_probability[family],
            "cf_alternate_probability": alternate_probability[family],
            "cf_agreement": cf_agreement,
            "cf_changed": changed_mask,
            "cross_matched_probability": matched_probability[family],
            "cross_different_probability": different_probability[family],
            "triplet_probability": triplet_probability[family],
        }
    return summaries, raw, controls, cf


def choose_student(metrics: dict[str, Any], selection: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = ("S1", "S1.5", "S2")
    tolerance = float(selection["noninferiority_tolerance_points"]) / 100.0
    tie_limit = float(selection["tie_mean_abs_max"])
    values = {
        name: {
            "language": metrics[name]["language_counterfactual"]["teacher_change_agreement"],
            "ordinary": metrics[name]["ordinary"]["local_non_tie"]["accuracy"],
            "tie": metrics[name]["tie"]["mean_abs_from_half"],
            "tasks": metrics[name]["ordinary"]["tasks_above_60pct"],
            "log_loss": metrics[name]["ordinary"]["local_non_tie"]["log_loss"],
        } for name in candidates
    }
    best_language = max(row["language"] for row in values.values())
    best_ordinary = max(row["ordinary"] for row in values.values())
    def noninferior(name: str) -> bool:
        row = values[name]
        return (row["tie"] <= tie_limit and row["language"] >= best_language - tolerance
                and row["ordinary"] >= best_ordinary - tolerance)
    if noninferior("S1"):
        selected, reason = "S1", "absolute S1 is within both frozen noninferiority tolerances"
    elif noninferior("S1.5"):
        selected, reason = "S1.5", "scalar pairwise potential is within both frozen noninferiority tolerances"
    else:
        eligible = [name for name in candidates if values[name]["tie"] <= tie_limit] or list(candidates)
        selected = sorted(eligible, key=lambda name: (
            -values[name]["language"], -values[name]["ordinary"], values[name]["tie"],
            -values[name]["tasks"], values[name]["log_loss"],
            {"S1": 0, "S1.5": 1, "S2": 2}[name],
        ))[0]
        reason = "frozen lexicographic fallback"
    return selected, {"metrics": values, "reason": reason, "tolerance": tolerance}


def save_weights(
    path: Path, models: dict[str, Any], stats: dict[str, np.ndarray],
    prototypes: np.ndarray, temperatures: dict[str, float], selected: str,
    metadata: dict[str, Any], protocol_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1, "protocol_sha256": protocol_hash,
        "models": {name: model_payload(model) for name, model in models.items()},
        "statistics": {name: torch.from_numpy(value) for name, value in stats.items()},
        "prototypes": torch.from_numpy(prototypes), "temperatures": temperatures,
        "selected_family": selected, "training_metadata": metadata,
    }, path)


def load_weights(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, dict[str, float], str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    models: dict[str, Any] = {
        "S1": S1AbsoluteSemantic(), "S1.5": ScalarSemanticPotential(),
        "S2": DirectSemanticPair(True), "IMAGE_ONLY": DirectSemanticPair(False),
        "LANGUAGE_ONLY": LanguageOnlyDiagnostic(),
    }
    for name, model in models.items(): model.load_state_dict(payload["models"][name]); model.eval()
    stats = {name: value.numpy() for name, value in payload["statistics"].items()}
    return (models, stats, payload["prototypes"].numpy(), payload["temperatures"],
            payload["selected_family"], payload["training_metadata"])


def model_profile(
    models: dict[str, Any], stats: dict[str, np.ndarray], selected: str,
    data: Any, protocol: dict[str, Any],
) -> dict[str, Any]:
    vision = standardize(data.vision, stats["visual_mean"], stats["visual_std"])
    language = standardize(data.language, stats["language_mean"], stats["language_std"])
    repeats = int(protocol["compute"]["latency_repeats"])
    warmup = int(protocol["compute"]["latency_warmup"])
    output = {}
    for family in ("S1", "S1.5"):
        model = models[family]
        one_v=torch.from_numpy(vision[:1]); one_l=torch.from_numpy(language[[int(data.task_id[0])]])
        n_v=torch.from_numpy(np.repeat(vision[:1],64,axis=0)); n_l=torch.from_numpy(np.repeat(language[[int(data.task_id[0])]],64,axis=0))
        def call_one():
            with torch.inference_mode(): model(one_v,one_l)
        def call_n():
            with torch.inference_mode(): model(n_v,n_l)
        for _ in range(warmup): call_one(); call_n()
        start=time.perf_counter()
        for _ in range(repeats): call_one()
        one=(time.perf_counter()-start)*1000/repeats
        start=time.perf_counter()
        for _ in range(repeats): call_n()
        many=(time.perf_counter()-start)*1000/repeats
        params=parameter_count(model)
        output[family]={"parameters":params,"parameter_memory_bytes_float32":params*4,
                        "one_state_cpu_ms":one,"n64_state_cpu_ms":many,"n64_evaluations":64}
    model=models["S2"]
    n=2016; a=torch.from_numpy(np.repeat(vision[:1],n,axis=0)); b=torch.from_numpy(np.repeat(vision[1:2],n,axis=0)); l=torch.from_numpy(np.repeat(language[[int(data.task_id[0])]],n,axis=0))
    def call_pair():
        with torch.inference_mode(): model(a,b,l)
    for _ in range(warmup): call_pair()
    start=time.perf_counter()
    for _ in range(repeats): call_pair()
    latency=(time.perf_counter()-start)*1000/repeats; params=parameter_count(model)
    output["S2"]={"parameters":params,"parameter_memory_bytes_float32":params*4,
                  "n64_full_tournament_cpu_ms":latency,"pair_evaluations":2016,
                  "aggregation_diagnostic":"Borda mean pairwise probability; not executed on candidate futures"}
    output["selected_family"]=selected
    return output


def run_calibration(protocol: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    data, ledger = load_partition(ROOT / protocol["inputs"]["t1_calibration_sidecar"],
                                  ROOT / protocol["inputs"]["feature_store"], "calibration")
    selection_train = tuple(protocol["partitions"]["selection_train_demo_ids"])
    selection_holdout = tuple(protocol["partitions"]["selection_holdout_demo_ids"])
    models, stats, metadata, train_pairs = train_models(data, selection_train, protocol, 0)
    prototypes = terminal_prototypes(data, selection_train, int(protocol["models"]["S0"]["terminal_tail_states"]))
    vision = standardize(data.vision, stats["visual_mean"], stats["visual_std"])
    language = standardize(data.language, stats["language_mean"], stats["language_std"])
    ordinary = ordinary_pairs(data, selection_holdout)
    cf_all = make_counterfactual_set(data, float(protocol["teacher"]["epsilon_goal"]), True)
    keep = np.isin(cf_all.demo_id, selection_holdout)
    cf = CounterfactualSet(*(getattr(cf_all, field)[keep] for field in cf_all.__dataclass_fields__))
    original, alternate = pairset_for_cf(cf,False), pairset_for_cf(cf,True)
    ordinary_logits,_=raw_probabilities(data,vision,language,models,prototypes,ordinary,None)
    original_logits,_=raw_probabilities(data,vision,language,models,prototypes,original,None)
    alternate_logits,_=raw_probabilities(data,vision,language,models,prototypes,alternate,None)
    temperatures=fit_temperatures(protocol,ordinary_logits,ordinary,original_logits,alternate_logits,cf)
    selection_metrics, _, _, _ = family_summaries(data,models,stats,prototypes,temperatures,
                                                    selection_holdout,protocol)
    selection_rule=json.loads((ROOT/protocol["inputs"]["student_selection"]).read_text())
    selected, selection_trace=choose_student(selection_metrics,selection_rule)

    final_demos=tuple(protocol["partitions"]["calibration_demo_ids"])
    final_models, final_stats, final_metadata, final_pairs=train_models(data,final_demos,protocol,100)
    final_prototypes=terminal_prototypes(data,final_demos,int(protocol["models"]["S0"]["terminal_tail_states"]))
    weights_path=ROOT/protocol["outputs"]["weights"]
    save_weights(weights_path,final_models,final_stats,final_prototypes,temperatures,selected,
                 final_metadata,protocol_hash)
    sidecar_path=ROOT/protocol["outputs"]["calibration_sidecar"]
    atomic_npz(sidecar_path,{
        "selection_pair_a":ordinary.a,"selection_pair_b":ordinary.b,
        "selection_pair_label":ordinary.label,"selection_pair_task":ordinary.task_id,
        "selection_pair_demo":ordinary.demo_id,
        "training_pair_a":final_pairs.a,"training_pair_b":final_pairs.b,
        "training_pair_language":final_pairs.language_id,"training_pair_label":final_pairs.label,
    })
    ledger_path=ROOT/protocol["outputs"]["calibration_ledger"]; atomic_json(ledger_path,ledger)
    result={
        "result_version":"semantic-value-calibration-v1","created_at":"2026-10-02",
        "source_commit":git_head(),"protocol_sha256":protocol_hash,
        "partition":{"demo_ids":list(final_demos),"frames":int(len(data.vision)),"trajectories":40},
        "selection":{"train_demos":list(selection_train),"holdout_demos":list(selection_holdout),
                     "selected_family":selected,"trace":selection_trace,"metrics":selection_metrics},
        "temperatures":temperatures,"selection_training":metadata,
        "final_training":final_metadata,
        "feature_partition_sha256":arrays_sha256(data.task_id,data.demo_id,data.timestep,data.vision,data.language),
        "parameter_counts":final_metadata["parameters"],"access_ledger":ledger,
        "all_finite":finite_tree(selection_metrics),
    }
    calibration_path=ROOT/protocol["outputs"]["calibration"]; atomic_json(calibration_path,result)
    freeze={
        "freeze_version":"semantic-value-calibration-freeze-v1","created_at":"2026-10-02",
        "protocol_sha256":protocol_hash,"selected_family":selected,
        "temperature":temperatures[selected],"tie_mean_abs_max":selection_rule["tie_mean_abs_max"],
        "artifact_sha256":{
            str(weights_path.relative_to(ROOT)):sha256(weights_path),
            str(calibration_path.relative_to(ROOT)):sha256(calibration_path),
            str(sidecar_path.relative_to(ROOT)):sha256(sidecar_path),
            str(ledger_path.relative_to(ROOT)):sha256(ledger_path),
        },
        "forbidden_access_count":ledger["forbidden_access_count"],
    }
    atomic_json(ROOT/protocol["outputs"]["calibration_freeze"],freeze)
    return {"selected":selected,"frames":len(data.vision),"training_pairs":len(final_pairs.a),
            "forbidden_access_count":ledger["forbidden_access_count"]}


def run_evaluation(protocol: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    data, ledger=load_partition(ROOT/protocol["inputs"]["t1_evaluation_sidecar"],
                               ROOT/protocol["inputs"]["feature_store"],"evaluation")
    models,stats,prototypes,temperatures,selected,training_metadata=load_weights(ROOT/protocol["outputs"]["weights"])
    demos=tuple(protocol["partitions"]["evaluation_demo_ids"])
    summaries,raw,controls,cf=family_summaries(data,models,stats,prototypes,temperatures,demos,protocol)
    primary=summaries[selected]; image=summaries["IMAGE_ONLY"]
    changed=raw[selected]["cf_changed"]
    paired=raw[selected]["cf_agreement"][changed]-raw["IMAGE_ONLY"]["cf_agreement"][changed]
    draws=block_draws(int(protocol["statistics"]["bootstrap_replicates"]),int(protocol["statistics"]["bootstrap_seed"]))
    paired_boot=bootstrap_mean(paired,cf.task_id[changed],cf.demo_id[changed],demos,draws)
    image_difference={"accuracy_difference":float(paired.mean()),"ci95":ci95(paired_boot)}
    profile=model_profile(models,stats,selected,data,protocol)
    gate=protocol["decision_rules"]
    selected_cycle=primary["transitivity"]["cycle_rate"]
    gates={
        "frozen_t1_hash_unchanged":True,
        "deployable_rgb_language_only":selected in {"S1","S1.5","S2"},
        "local_non_tie_accuracy":primary["ordinary"]["local_non_tie"]["accuracy"]>=gate["local_accuracy_min"],
        "local_ci_lower":primary["ordinary"]["local_accuracy_ci95"][0]>=gate["local_ci_lower_min"],
        "task_robustness":primary["ordinary"]["tasks_above_60pct"]>=gate["tasks_above_60_min"],
        "language_change_agreement":primary["language_counterfactual"]["teacher_change_agreement"]>=gate["language_change_agreement_min"],
        "language_task_robustness":primary["language_counterfactual"]["positive_task_count"]>=gate["language_positive_task_min"],
        "beats_image_only":image_difference["ci95"][0]>0,
        "same_referent_invariance":primary["language_counterfactual"]["same_referent"]["mean_abs_probability_change"]<=gate["same_referent_mean_change_max"],
        "semantic_tie_calibration":primary["tie"]["mean_abs_from_half"]<=gate["tie_mean_abs_max"],
        "semantic_match_cross_demo":primary["cross_demo"]["semantic_match"]["mean_abs_from_half"]<=gate["cross_demo_mean_abs_max"],
        "direct_pairwise_transitivity":selected!="S2" or selected_cycle<=gate["s2_cycle_rate_max"],
        "all_values_finite":finite_tree(summaries) and finite_tree(profile),
        "forbidden_access_zero":ledger["forbidden_access_count"]==0,
    }
    complete=bool(all(gates.values()))
    labels=[]
    if complete:
        labels=["SEMANTIC_VALUE_DISTILLATION_VALID","GOAL_CONDITIONED_VALUE_STUDENT_VALID"]
        labels.append("SCALAR_SEMANTIC_POTENTIAL_VALID" if selected=="S1" else
                      "PAIRWISE_POTENTIAL_DISTILLATION_VALID" if selected=="S1.5" else
                      "PAIRWISE_SEMANTIC_STUDENT_VALID")
        labels += ["LANGUAGE_COUNTERFACTUAL_VALUE_TRANSFER_VALID",
                   "SEMANTIC_TIE_CALIBRATION_VALID","SEMANTIC_CROSS_DEMO_GENERALIZATION_VALID"]
        authorization="GO_REAL_CANDIDATE_VALUE_BRIDGE"; selected_output=selected
    else:
        labels=["SEMANTIC_VALUE_DISTILLATION_BLOCKED"]
        if not gates["language_change_agreement"] or not gates["language_task_robustness"] or not gates["beats_image_only"]:
            labels.append("VALUE_STUDENT_LANGUAGE_INSENSITIVE")
        if not gates["local_non_tie_accuracy"] or not gates["local_ci_lower"]:
            labels += ["VALUE_STUDENT_VISUAL_INFORMATION_INSUFFICIENT","FROZEN_VISUAL_VALUE_INFORMATION_INSUFFICIENT"]
        if not gates["semantic_match_cross_demo"]: labels.append("VALUE_STUDENT_SCENE_SHORTCUT")
        if not gates["semantic_tie_calibration"]: labels.append("VALUE_STUDENT_TIE_OVERCONFIDENT")
        if not gates["direct_pairwise_transitivity"]: labels.append("DIRECT_PAIRWISE_TRANSITIVITY_BLOCK")
        if not gates["forbidden_access_zero"]: labels.append("DATA_OR_IMPLEMENTATION_BLOCK")
        labels=list(dict.fromkeys(labels));authorization="BLOCK_REAL_CANDIDATE_VALUE_BRIDGE";selected_output="none"
    result={
        "result_version":"semantic-value-evaluation-v1","created_at":"2026-10-02",
        "source_commit":git_head(),"protocol_sha256":protocol_hash,
        "partition":{"demo_ids":list(demos),"frames":int(len(data.vision)),"trajectories":40,
                     "interpretation":"frozen development evaluation; not blind confirmation"},
        "frozen_representative":selected,"selected_semantic_value_student":selected_output,
        "temperatures":temperatures,"metrics":summaries,"primary_vs_image_only":image_difference,
        "compute":profile,"gates":gates,"complete_gate":complete,"final_labels":labels,
        "authorization":authorization,"continuing_blocks":["BLOCK_WORLD_MODEL_INTERFACE_AUDIT","BLOCK_SPATIAL_WM_PROTOTYPE"],
        "controller_authorization":False,"real_candidate_bridge_executed":False,
        "feature_partition_sha256":arrays_sha256(data.task_id,data.demo_id,data.timestep,data.vision,data.language),
        "access_ledger":ledger,"training_metadata":training_metadata,
    }
    evaluation_path=ROOT/protocol["outputs"]["evaluation"]; atomic_json(evaluation_path,result)
    atomic_json(ROOT/protocol["outputs"]["language_evaluation"],{
        "result_version":"semantic-language-counterfactual-eval-v1","selected_family":selected,
        "primary":primary["language_counterfactual"],"image_only":image["language_counterfactual"],
        "primary_vs_image_only":image_difference,
    })
    atomic_json(ROOT/protocol["outputs"]["tie_evaluation"],{
        "result_version":"semantic-tie-eval-v1","selected_family":selected,
        "families":{name:value["tie"] for name,value in summaries.items()},
    })
    atomic_json(ROOT/protocol["outputs"]["cross_demo_evaluation"],{
        "result_version":"semantic-cross-demo-controls-v1","selected_family":selected,
        "matched_rows":int(len(controls["matched"].a)),"different_rows":int(len(controls["different"].a)),
        "families":{name:value["cross_demo"] for name,value in summaries.items()},
    })
    atomic_json(ROOT/protocol["outputs"]["evaluation_ledger"],ledger)
    sidecar={"ordinary_task":ordinary_pairs(data,demos).task_id,"ordinary_demo":ordinary_pairs(data,demos).demo_id,
             "ordinary_label":ordinary_pairs(data,demos).label,
             "cf_task":cf.task_id,"cf_demo":cf.demo_id,"cf_original_label":cf.original_label,
             "cf_alternate_label":cf.alternate_label,"cf_same_referent":cf.same_referent}
    for family in FAMILIES:
        sidecar[f"{family}__ordinary_probability"]=raw[family]["ordinary_probability"].astype(np.float32)
        sidecar[f"{family}__cf_original_probability"]=raw[family]["cf_original_probability"].astype(np.float32)
        sidecar[f"{family}__cf_alternate_probability"]=raw[family]["cf_alternate_probability"].astype(np.float32)
    atomic_npz(ROOT/protocol["outputs"]["evaluation_sidecar"],sidecar)
    return {"selected":selected_output,"representative":selected,"complete_gate":complete,
            "authorization":authorization,"labels":labels,"forbidden_access_count":ledger["forbidden_access_count"]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("calibration","evaluation"),required=True)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/semantic_value_distillation_protocol.json")
    args=parser.parse_args();protocol=validate_protocol(args.protocol,args.phase);protocol_hash=sha256(args.protocol)
    result=run_calibration(protocol,protocol_hash) if args.phase=="calibration" else run_evaluation(protocol,protocol_hash)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
