"""Candidate generation and control logic layered on the frozen C0 actor."""

from .candidate_reranker import generate_antithetic_candidates, score_candidates

__all__ = ["generate_antithetic_candidates", "score_candidates"]
