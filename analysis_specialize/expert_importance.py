"""
Expert Importance Scoring Module

Part 2: Expert importance scoring based on activation counts

This module provides:
1. Weighted importance score computation: e_score = (f_important - f_unimportant) * p_e
2. Score formatting and normalization
3. I/O functions for saving/loading scores
4. Helper functions for getting top/bottom experts
"""

import os
import json
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# Handle both package and script imports
try:
    from .token_expert_analysis import safe_model_name, ensure_dir
except ImportError:
    from token_expert_analysis import safe_model_name, ensure_dir


# =============================================================================
# Expert Importance Score Computation
# =============================================================================

def compute_expert_importance_scores(
    expert_activation_counts: Dict[Tuple[int, int], Dict[str, int]],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute expert importance scores using the formula:
    e_score = (f_important - f_unimportant) * p_e

    where:
    - f_important: frequency of expert activation on important tokens
    - f_unimportant: frequency of expert activation on unimportant tokens
    - p_e: overall activation frequency of the expert

    Args:
        expert_activation_counts: Dict mapping (layer_idx, expert_idx) to
            {'total': int, 'important': int, 'unimportant': int}

    Returns:
        Dict mapping (layer_idx, expert_idx) to score details
    """
    total_activations = sum(
        counts['total'] for counts in expert_activation_counts.values()
    )

    expert_scores = {}

    for (layer_idx, expert_idx), counts in expert_activation_counts.items():
        f_important = counts['important']
        f_unimportant = counts['unimportant']
        total_e = counts['total']

        # p_e: frequency of activation for this expert
        p_e = total_e / total_activations if total_activations > 0 else 0.0

        # e_score = (f_important - f_unimportant) * p_e
        e_score = (f_important - f_unimportant) * p_e

        expert_scores[(layer_idx, expert_idx)] = {
            'weighted_score': float(e_score),
            'f_important': f_important,
            'f_unimportant': f_unimportant,
            'p_e': float(p_e),
            'total_activations': total_e,
        }

    return expert_scores


# =============================================================================
# Score Formatting
# =============================================================================

def format_expert_scores_for_saving(
    expert_scores: Dict[Tuple[int, int], Dict[str, float]],
) -> Dict:
    """
    Format expert scores into JSON-compatible structure.

    Args:
        expert_scores: Dict mapping (layer_idx, expert_idx) to score details

    Returns:
        Formatted dict ready for JSON serialization
    """
    formatted = {
        "per_layer": {},
        "per_layer_normalized": {},
        "overall": {},
        "overall_normalized": {},
    }

    # Organize by layer
    for (layer_idx, expert_idx), data in expert_scores.items():
        layer_key = str(layer_idx)
        expert_key = str(expert_idx)

        if layer_key not in formatted["per_layer"]:
            formatted["per_layer"][layer_key] = {}

        formatted["per_layer"][layer_key][expert_key] = {
            "expert_name": f"Expert_L{layer_idx}_E{expert_idx}",
            "weighted_score": data['weighted_score'],
        }

        # Add to overall scores
        expert_name = f"Expert_L{layer_idx}_E{expert_idx}"
        formatted["overall"][expert_name] = data['weighted_score']

    # Normalize per layer
    for layer_key, experts in formatted["per_layer"].items():
        scores = [exp['weighted_score'] for exp in experts.values()]
        total_score = sum(scores) if scores else 1.0

        formatted["per_layer_normalized"][layer_key] = {}
        for expert_key, expert_data in experts.items():
            normalized = expert_data['weighted_score'] / total_score if total_score != 0 else 0.0
            formatted["per_layer_normalized"][layer_key][expert_key] = {
                "expert_name": expert_data['expert_name'],
                "weighted_score": expert_data['weighted_score'],
                "normalized_score": normalized,
            }

    # Normalize overall
    total_overall = sum(formatted["overall"].values()) if formatted["overall"] else 1.0
    for expert_name, score in formatted["overall"].items():
        formatted["overall_normalized"][expert_name] = score / total_overall if total_overall != 0 else 0.0

    return formatted


# =============================================================================
# I/O Functions
# =============================================================================

def save_expert_scores(
    expert_scores: Dict[Tuple[int, int], Dict[str, float]],
    domain: str,
    model_name: str,
    output_dir: str = "expert_scores_output",
) -> str:
    """Save expert scores to JSON file."""
    model_safe = safe_model_name(model_name)
    model_output_dir = os.path.join(output_dir, model_safe)
    ensure_dir(model_output_dir)

    formatted = format_expert_scores_for_saving(expert_scores)

    output_file = os.path.join(
        model_output_dir,
        f"{domain}_trend_results_importance_weighted_scores.json"
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2)

    print(f"[+] Expert scores saved to: {output_file}")
    return output_file


def load_expert_scores(
    domain: str,
    model_name: str,
    scores_dir: str = "expert_scores_output",
) -> Optional[Dict]:
    """Load expert scores from JSON file."""
    model_safe = safe_model_name(model_name)

    model_scores_dir = os.path.join(scores_dir, model_safe)
    scores_file = os.path.join(
        model_scores_dir,
        f"{domain}_trend_results_importance_weighted_scores.json"
    )

    if not os.path.exists(scores_file):
        scores_file = os.path.join(
            scores_dir,
            f"{domain}_trend_results_importance_weighted_scores.json"
        )

    if not os.path.exists(scores_file):
        print(f"[!] Scores file not found for domain '{domain}'")
        return None

    with open(scores_file, 'r') as f:
        data = json.load(f)

    print(f"[+] Loaded expert scores from: {scores_file}")
    return data


# =============================================================================
# Helper Functions
# =============================================================================

def get_top_experts(
    scores_data: Dict,
    top_k: int,
    reverse: bool = True,
) -> List[Tuple[int, int, float]]:
    """Get top-k experts globally by score."""
    if "per_layer" not in scores_data:
        return []

    all_experts = []
    for layer_key, experts_dict in scores_data["per_layer"].items():
        layer_idx = int(layer_key)
        for expert_key, expert_info in experts_dict.items():
            expert_idx = int(expert_key)
            score = expert_info.get("weighted_score", 0.0)
            all_experts.append((layer_idx, expert_idx, score))

    all_experts.sort(key=lambda x: x[2], reverse=reverse)
    return all_experts[:top_k]


def get_bottom_experts(
    scores_data: Dict,
    bottom_k: int,
) -> List[Tuple[int, int, float]]:
    """Get bottom-k experts globally by score (lowest scores)."""
    return get_top_experts(scores_data, bottom_k, reverse=False)


def group_experts_by_layer(
    experts: List[Tuple[int, int, float]],
) -> Dict[int, Set[int]]:
    """Group expert list by layer."""
    grouped = defaultdict(set)
    for layer_idx, expert_idx, score in experts:
        grouped[layer_idx].add(expert_idx)
    return dict(grouped)
