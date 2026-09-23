"""Candidate generation and control logic layered on the frozen C0 actor."""

from .candidate_reranker import generate_antithetic_candidates, score_candidates
from .candidate_reranker_v2 import prepare_candidate_action_interface
from .verification import verification_decision

__all__ = [
    "generate_antithetic_candidates",
    "prepare_candidate_action_interface",
    "score_candidates",
    "verification_decision",
]
