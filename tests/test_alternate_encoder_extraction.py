from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_stage_e_features_match_frozen_contract_and_hashes() -> None:
    manifest = json.loads((ROOT / "results/alternate_encoder_feature_manifest.json").read_text())
    contracts = json.loads((ROOT / "configs/alternate_encoder_contracts.json").read_text())
    assert manifest["complete_gate"] is True
    assert manifest["frames"] == 5044
    for family, result in manifest["families"].items():
        contract = contracts["families"][family]
        feature_path = ROOT / "results/alternate_frozen_encoder/features" / f"{family}.npy"
        feature = np.load(feature_path, mmap_mode="r")
        assert list(feature.shape) == [
            5044, contract["token_count"], contract["dimension"]
        ]
        assert feature.dtype == np.float16
        assert sha256(feature_path) == result["feature_sha256"]
        assert result["config_sha256"] == contract["config_sha256"]
        assert result["preprocessor_sha256"] == contract["preprocessor_sha256"]
        assert result["revision"] == contract["revision"]
        assert result["repeated_sample_float16_exact"] is True
        assert result["trainable_encoder_parameters"] == 0
        assert result["all_finite"] is True


def test_stage_e_extraction_access_is_calibration_only() -> None:
    ledger = json.loads((ROOT / "results/alternate_frozen_encoder/access_ledger.json").read_text())
    assert ledger["partition"] == "calibration_only"
    assert ledger["opened_demo_ids"] == [31, 32, 33, 34]
    assert ledger["stage_e_not_opened_demo_ids"] == [36, 37, 38, 39]
    assert ledger["forbidden_demo_ids"] == list(range(40, 50))
    assert ledger["forbidden_numeric_access"] == 0
    assert ledger["forbidden_metadata_access"] == 0
    for source in ledger["source_rgb_by_family"].values():
        assert source["opened_demo_ids"] == [31, 32, 33, 34]
        assert source["metadata_operations"] == []
        assert source["forbidden_attempts"] == []
        assert source["forbidden_access_count"] == 0
        assert all("obs/agentview_rgb" in path for path in source["numeric_datasets_accessed"])
    frozen = ledger["frozen_feature_store"]
    assert frozen["opened_demo_ids"] == [31, 32, 33, 34]
    assert frozen["metadata_operations"] == []
    assert frozen["forbidden_attempts"] == []
    assert frozen["forbidden_access_count"] == 0
