import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_at_commit(commit: str, path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def test_manifest_hashes_and_tags_are_exact():
    manifest = json.loads((ROOT / "results/grounding_to_semantic_teacher_manifest.json").read_text())
    for stage in ("stage_g_freeze", "stage_t_freeze"):
        row = manifest[stage]
        for path, expected in row["artifacts"].items():
            # Later independently preregistered studies may reuse a required
            # output path.  The historical manifest is a commit/tag ledger, so
            # verify its immutable Git object rather than the current worktree.
            assert sha256_at_commit(row["commit"], path) == expected
        resolved = subprocess.check_output(
            ["git", "rev-parse", f"{row['annotated_tag']}^{{}}"], cwd=ROOT, text=True
        ).strip()
        assert resolved == row["commit"]


def test_final_decision_is_go_only_without_execution():
    manifest = json.loads((ROOT / "results/grounding_to_semantic_teacher_manifest.json").read_text())
    assert "SEMANTIC_GOAL_PROGRESS_TEACHER_VALID" in manifest["final_labels"]
    assert manifest["authorization"] == "GO_VALUE_DISTILLATION"
    assert manifest["value_distillation_executed"] is False
    assert manifest["controller_authorization"] is False
