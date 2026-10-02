"""Source-grounded LIBERO BDDL semantic-contract inventory utilities.

This module deliberately operates on task definitions only.  It never opens a
demonstration, feature row, simulator state, reward, success value, or action.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    without_comments = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", without_comments)


def parse_sexpression(text: str) -> list[Any]:
    """Parse the single top-level BDDL S-expression without importing LIBERO."""

    tokens = _tokens(text)
    stack: list[list[Any]] = []
    roots: list[Any] = []
    for token in tokens:
        if token == "(":
            node: list[Any] = []
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("unmatched closing parenthesis")
            stack.pop()
        else:
            if not stack:
                raise ValueError("token outside an S-expression")
            stack[-1].append(token)
    if stack:
        raise ValueError("unclosed parenthesis")
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise ValueError("expected one top-level expression")
    return roots[0]


def _section(root: list[Any], name: str) -> list[Any]:
    for item in root:
        if isinstance(item, list) and item and str(item[0]).lower() == name.lower():
            return item
    raise ValueError(f"missing BDDL section {name}")


def _atom_signature(expression: Any) -> Any:
    if isinstance(expression, list):
        return [_atom_signature(item) for item in expression]
    return str(expression).lower()


def _goal_atoms(goal: list[Any]) -> list[list[str]]:
    if len(goal) != 2 or not isinstance(goal[1], list):
        raise ValueError("unsupported goal wrapper")
    expression = goal[1]
    if not expression or str(expression[0]).lower() != "and":
        raise ValueError("only conjunction goals are admitted")
    atoms: list[list[str]] = []
    for item in expression[1:]:
        if not isinstance(item, list) or len(item) not in (2, 3):
            raise ValueError("only unary/binary atomic goal predicates are admitted")
        atoms.append([str(part).lower() for part in item])
    return sorted(atoms)


def _initial_focus(initial: list[Any], relevant: str) -> list[list[str]]:
    rows = []
    for item in initial[1:]:
        if isinstance(item, list) and relevant in [str(part) for part in item[1:]]:
            rows.append([str(part).lower() for part in item])
    return sorted(rows)


def parse_task_contract(path: Path, task_id: int) -> dict[str, Any]:
    root = parse_sexpression(path.read_text())
    if not root or str(root[0]).lower() != "define":
        raise ValueError(f"{path} is not a define expression")
    problem = _section(root, "problem")
    language = _section(root, ":language")
    interest = _section(root, ":obj_of_interest")
    initial = _section(root, ":init")
    goal = _section(root, ":goal")
    objects = _section(root, ":objects")
    fixtures = _section(root, ":fixtures")
    if len(problem) != 2 or len(interest) < 2:
        raise ValueError(f"invalid problem or objects-of-interest section in {path}")
    goal_atoms = _goal_atoms(goal)
    relevant = str(interest[1]).lower()
    goal_arguments = {part for atom in goal_atoms for part in atom[1:]}
    targets = sorted(goal_arguments - {relevant})
    return {
        "task_id": int(task_id),
        "task_name": path.stem,
        "problem_name": str(problem[1]),
        "instruction": " ".join(str(item) for item in language[1:]),
        "objects_of_interest": [str(item).lower() for item in interest[1:]],
        "relevant_object": relevant,
        "target_objects_or_regions": targets,
        "goal_atoms": goal_atoms,
        "goal_signature": _atom_signature(goal[1]),
        "relevant_initial_atoms": _initial_focus(initial, str(interest[1])),
        "object_section": _atom_signature(objects[1:]),
        "fixture_section": _atom_signature(fixtures[1:]),
        "bddl_path": path.as_posix(),
        "bddl_sha256": sha256(path),
    }


def inventory_contracts(paths: list[Path]) -> dict[str, Any]:
    contracts = [parse_task_contract(path, task_id) for task_id, path in enumerate(paths)]
    goal_signatures = {
        repr(contract["goal_signature"]): contract["goal_signature"] for contract in contracts
    }
    initial_signatures = {
        repr(contract["relevant_initial_atoms"]): contract["relevant_initial_atoms"]
        for contract in contracts
    }
    directed_language_swaps = 0
    semantic_counterfactuals: list[dict[str, int]] = []
    for source in contracts:
        for alternative in contracts:
            if source["task_id"] == alternative["task_id"]:
                continue
            directed_language_swaps += 1
            if source["goal_signature"] != alternative["goal_signature"]:
                semantic_counterfactuals.append(
                    {"task_id": source["task_id"], "counterfactual_task_id": alternative["task_id"]}
                )
    coverage = {
        str(task_id): sum(row["task_id"] == task_id for row in semantic_counterfactuals)
        for task_id in range(len(contracts))
    }
    return {
        "task_count": len(contracts),
        "contracts": contracts,
        "distinct_goal_signature_count": len(goal_signatures),
        "distinct_goal_signatures": list(goal_signatures.values()),
        "distinct_relevant_initial_signature_count": len(initial_signatures),
        "directed_real_language_swap_count": directed_language_swaps,
        "directed_goal_changing_counterfactual_count": len(semantic_counterfactuals),
        "goal_changing_counterfactuals": semantic_counterfactuals,
        "counterfactual_coverage_by_task": coverage,
        "task_count_with_goal_changing_counterfactual": sum(value > 0 for value in coverage.values()),
    }

