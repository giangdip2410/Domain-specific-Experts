"""
Expert specialization analysis package.

The package provides utilities for token importance scoring, MoE expert
activation capture, expert importance scoring, and domain steering.
"""

from .token_expert_analysis import (
    analyze_domain_data,
    capture_expert_activations,
    classify_tokens_by_importance,
    compute_gradient_importance,
    count_expert_activations_by_importance,
    ensure_dir,
    get_token_importance_stats,
    load_token_classifications,
    safe_model_name,
    save_token_classifications,
)
from .expert_importance import (
    compute_expert_importance_scores,
    format_expert_scores_for_saving,
    get_bottom_experts,
    get_top_experts,
    group_experts_by_layer,
    load_expert_scores,
    save_expert_scores,
)
from .domain_steering import (
    DomainSteering,
    DomainSteeringConfig,
    evaluate_domain,
    load_domain_examples,
    run_domain_steering,
    select_top_experts_from_scores,
)

__all__ = [
    "analyze_domain_data",
    "capture_expert_activations",
    "classify_tokens_by_importance",
    "compute_gradient_importance",
    "count_expert_activations_by_importance",
    "ensure_dir",
    "get_token_importance_stats",
    "load_token_classifications",
    "safe_model_name",
    "save_token_classifications",
    "compute_expert_importance_scores",
    "format_expert_scores_for_saving",
    "get_bottom_experts",
    "get_top_experts",
    "group_experts_by_layer",
    "load_expert_scores",
    "save_expert_scores",
    "DomainSteering",
    "DomainSteeringConfig",
    "evaluate_domain",
    "load_domain_examples",
    "run_domain_steering",
    "select_top_experts_from_scores",
]
