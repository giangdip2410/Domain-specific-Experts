"""
Token and Expert Activation Analysis Module

Part 1: Token importance classification (gradient-based)
Part 1.5: Expert activation capture and counting

This module provides:
1. Gradient-based token importance computation
2. Token classification (important/unimportant)
3. Expert activation capture via hooks
4. Counting expert activations by token importance
"""

import torch
import torch.nn.functional as F
import numpy as np
import string
import os
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from tqdm import tqdm

# Try to load NLTK stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except (ImportError, LookupError):
    # Fallback: common English stopwords
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
        "i", "me", "my", "who", "what", "which", "when", "where", "why", "how",
    }


# =============================================================================
# Helper Functions
# =============================================================================

def safe_model_name(model_name: str) -> str:
    """Convert model name to safe filename format."""
    name = model_name.split('/')[-1]
    name = name.replace('-', '_').replace('.', '_')
    return name


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# PART 1: Token Importance Classification (Gradient-based)
# =============================================================================

def compute_gradient_importance(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute token importance using gradient × input magnitude (Saliency method).

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        device: Device to use

    Returns:
        Numpy array of importance scores [batch_size, seq_len]
    """
    input_ids = input_ids.to(device)

    # Get input embeddings
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds.requires_grad_(True)
    input_embeds.retain_grad()

    # Forward pass with labels for loss computation (disable cache for compatibility)
    outputs = model(inputs_embeds=input_embeds, labels=input_ids, use_cache=False)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Gradient × input norm as importance score
    grads = input_embeds.grad
    token_importance = (grads * input_embeds).norm(dim=-1)
    token_importance = token_importance.to(torch.float32).detach().cpu().numpy()

    return token_importance


def classify_tokens_by_importance(
    tokens: List[str],
    importance_scores: List[float],
    threshold_percentile: float = 50.0,
    filter_stopwords: bool = True,
    filter_punctuation: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Classify tokens as important or unimportant based on gradient scores.

    Args:
        tokens: List of token strings
        importance_scores: List of importance scores
        threshold_percentile: Percentile threshold for classification
        filter_stopwords: Whether to classify stopwords as unimportant
        filter_punctuation: Whether to classify punctuation as unimportant

    Returns:
        Tuple of (important_indices, unimportant_indices)
    """
    threshold = np.percentile(importance_scores, threshold_percentile)

    important_indices = []
    unimportant_indices = []

    for idx, (token, score) in enumerate(zip(tokens, importance_scores)):
        token_lower = token.lower().strip()

        # Check for automatic unimportant classification
        is_punctuation = token in string.punctuation
        is_stopword = token_lower in STOPWORDS
        is_below_threshold = score <= threshold

        if filter_punctuation and is_punctuation:
            unimportant_indices.append(idx)
        elif filter_stopwords and is_stopword:
            unimportant_indices.append(idx)
        elif is_below_threshold:
            unimportant_indices.append(idx)
        else:
            important_indices.append(idx)

    return important_indices, unimportant_indices


def get_token_importance_stats(
    importance_scores: List[float],
) -> dict:
    """
    Compute statistics for token importance scores.

    Args:
        importance_scores: List of importance scores

    Returns:
        Dict with statistics
    """
    scores = np.array(importance_scores)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
    }


# =============================================================================
# PART 1.5: Expert Activation Capture (Bridge between Part 1 and Part 3)
# =============================================================================

def capture_expert_activations(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: str = "cuda",
) -> Dict[int, Dict[int, List[int]]]:
    """
    Capture which experts are activated for each token.

    This function hooks into the MoE layers and records which experts
    are selected by the router for each token position.

    Args:
        model: Loaded MoE model
        tokenizer: Tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        device: Device to use

    Returns:
        Dict: {layer_idx: {token_idx: [activated_expert_indices]}}
    """
    input_ids = input_ids.to(device)

    expert_activations = defaultdict(lambda: defaultdict(list))

    # Get model config for top_k detection (for models like Qwen3 that store it in config)
    model_config = getattr(model, 'config', None)

    def get_top_k(router, moe_module, model_config):
        """Get top_k from router, moe_module, or model config."""
        # Note: 'k' is removed as it's too generic and can match unrelated attributes
        top_k_attrs = [
            'top_k', 'num_selects', 'num_experts_per_tok', 'topk',
            'num_selected_experts', 'routing_top_k', 'moe_top_k',
            'experts_per_token', 'n_experts_per_tok', 'num_experts_per_token'
        ]

        # Get num_experts for validation
        num_experts = 0
        if hasattr(router, 'weight'):
            num_experts = router.weight.shape[0]
        elif hasattr(router, 'linear') and hasattr(router.linear, 'weight'):
            num_experts = router.linear.weight.shape[0]
        max_valid_topk = num_experts if num_experts > 0 else 128

        def check_attrs(obj):
            """Check an object for top_k attributes."""
            for attr in top_k_attrs:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    # top_k must be positive int and <= num_experts
                    if isinstance(val, int) and val > 0 and val <= max_valid_topk:
                        return val
            return None

        # Check router attributes first
        val = check_attrs(router)
        if val is not None:
            return val

        # Check moe_module attributes
        if moe_module is not None:
            val = check_attrs(moe_module)
            if val is not None:
                return val
            # Also check moe_module.config if exists
            if hasattr(moe_module, 'config'):
                val = check_attrs(moe_module.config)
                if val is not None:
                    return val

        # Check model config
        if model_config is not None:
            val = check_attrs(model_config)
            if val is not None:
                return val

            # Check nested configs (text_config, moe_config, etc.)
            nested_config_attrs = ['text_config', 'moe_config', 'router_config', 'model_config']
            for nested_attr in nested_config_attrs:
                if hasattr(model_config, nested_attr):
                    nested_config = getattr(model_config, nested_attr)
                    if nested_config is not None:
                        val = check_attrs(nested_config)
                        if val is not None:
                            return val

        return 2  # default

    def make_hook(layer_idx, moe_module):
        def hook(module, input, output):
            try:
                hidden_states = input[0]

                # Get router (different models use different names)
                router = None
                if hasattr(module, 'router'):
                    router = module.router
                elif hasattr(module, 'gate'):
                    router = module.gate

                if router is None:
                    return

                # Get router weight for dtype and dimension detection
                router_weight = None
                if hasattr(router, 'weight'):
                    router_weight = router.weight
                elif hasattr(router, 'linear') and hasattr(router.linear, 'weight'):
                    router_weight = router.linear.weight

                # Get hidden dimension
                if hasattr(router, 'hidden_dim'):
                    h_reshaped = hidden_states.reshape(-1, router.hidden_dim)
                elif router_weight is not None:
                    h_reshaped = hidden_states.reshape(-1, router_weight.shape[1])
                else:
                    h_reshaped = hidden_states.reshape(-1, hidden_states.shape[-1])

                # Always use float32 for router computation to avoid dtype issues
                # (handles Half vs BFloat16 mismatches in quantized models)
                compute_dtype = torch.float32

                # Compute router logits
                if router_weight is not None:
                    h_compute = h_reshaped.to(compute_dtype)
                    weight_compute = router_weight.to(compute_dtype)

                    # Get bias and convert if needed
                    bias = None
                    if hasattr(router, 'bias') and router.bias is not None:
                        bias = router.bias.to(compute_dtype)
                    elif hasattr(router, 'linear') and hasattr(router.linear, 'bias') and router.linear.bias is not None:
                        bias = router.linear.bias.to(compute_dtype)

                    router_logits = F.linear(h_compute, weight_compute, bias)
                else:
                    # Call router directly with float32
                    h_compute = h_reshaped.to(compute_dtype)
                    original_dtype = next(router.parameters()).dtype
                    router.to(compute_dtype)
                    router_logits = router(h_compute)
                    router.to(original_dtype)

                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]

                # Get top-k experts (check router, moe_module, and model config)
                top_k = get_top_k(router, moe_module, model_config)
                _, top_indices = torch.topk(
                    router_logits,
                    min(top_k, router_logits.shape[-1]),
                    dim=-1
                )

                # Store activations per token
                seq_len = input_ids.shape[1]
                for token_idx in range(seq_len):
                    if token_idx < top_indices.shape[0]:
                        experts = top_indices[token_idx].cpu().tolist()
                        expert_activations[layer_idx][token_idx].extend(experts)
            except Exception:
                # Skip this layer if there's an error
                pass

        return hook

    # Register hooks on MoE layers
    hooks = []

    # Find layers (different model architectures use different paths)
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h

    if layers is not None:
        # Common MoE layer attribute names across different models
        moe_layer_attrs = ['mlp', 'block_sparse_moe', 'moe', 'feed_forward']

        for layer_idx, layer in enumerate(layers):
            moe_module = None
            for attr in moe_layer_attrs:
                if hasattr(layer, attr):
                    moe_module = getattr(layer, attr)
                    break

            if moe_module is not None:
                # Check if this layer has a router (gate) - confirming it's a MoE layer
                has_router = hasattr(moe_module, 'router') or hasattr(moe_module, 'gate')
                if has_router:
                    hook = moe_module.register_forward_hook(make_hook(layer_idx, moe_module))
                    hooks.append(hook)

    # Forward pass to trigger hooks (disable cache to avoid compatibility issues)
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return dict(expert_activations)


def count_expert_activations_by_importance(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    important_token_indices: Set[int],
    unimportant_token_indices: Set[int],
    device: str = "cuda",
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    Count expert activations separately for important and unimportant tokens.

    This bridges token importance with expert-score computation and downstream
    domain steering.

    Args:
        model: Loaded MoE model
        tokenizer: Tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        important_token_indices: Set of token indices classified as important
        unimportant_token_indices: Set of token indices classified as unimportant
        device: Device to use

    Returns:
        Dict mapping (layer_idx, expert_idx) to
        {'total': int, 'important': int, 'unimportant': int}
    """
    expert_counts = defaultdict(lambda: {'total': 0, 'important': 0, 'unimportant': 0})

    # Capture which experts activate for each token
    activations = capture_expert_activations(model, tokenizer, input_ids, device)

    # Count activations by importance category
    for layer_idx, token_acts in activations.items():
        for token_idx, expert_list in token_acts.items():
            is_important = token_idx in important_token_indices
            is_unimportant = token_idx in unimportant_token_indices

            for expert_idx in expert_list:
                key = (layer_idx, expert_idx)
                expert_counts[key]['total'] += 1
                if is_important:
                    expert_counts[key]['important'] += 1
                elif is_unimportant:
                    expert_counts[key]['unimportant'] += 1

    return dict(expert_counts)


def save_token_classifications(
    token_data: Dict[str, Any],
    domain: str,
    model_name: str,
    output_dir: str = "output/token_classifications",
) -> str:
    """
    Save token classification results to JSON file.

    Args:
        token_data: Dictionary containing token classification data
        domain: Domain name
        model_name: Model name
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    import json

    model_safe = safe_model_name(model_name)
    model_output_dir = os.path.join(output_dir, model_safe)
    ensure_dir(model_output_dir)

    output_file = os.path.join(
        model_output_dir,
        f"{domain}_token_classifications.json"
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(token_data, f, indent=2, ensure_ascii=False)

    print(f"[+] Token classifications saved to: {output_file}")
    return output_file


def load_token_classifications(
    domain: str,
    model_name: str,
    output_dir: str = "output/token_classifications",
) -> Optional[Dict[str, Any]]:
    """
    Load token classification results from JSON file.

    Args:
        domain: Domain name
        model_name: Model name
        output_dir: Output directory

    Returns:
        Token classification data or None if not found
    """
    import json

    model_safe = safe_model_name(model_name)
    results_file = os.path.join(output_dir, model_safe, f"{domain}_token_classifications.json")

    if not os.path.exists(results_file):
        print(f"[!] Token classifications not found: {results_file}")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[+] Loaded token classifications from: {results_file}")
    return data


def analyze_domain_data(
    model,
    tokenizer,
    examples: List[Dict],
    importance_threshold: float = 50.0,
    device: str = "cuda",
    max_length: int = 2048,
    save_tokens: bool = False,
    domain: str = "",
    model_name: str = "",
    output_dir: str = "output/token_classifications",
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    Complete analysis pipeline for a set of domain examples.

    This function runs the FULL PIPELINE:
    1. Compute token importance for all examples
    2. Classify tokens as important/unimportant
    3. Capture expert activations
    4. Count activations for important vs unimportant tokens

    Args:
        model: Loaded MoE model
        tokenizer: Tokenizer
        examples: List of examples, each with 'question' and 'options' keys
        importance_threshold: Percentile threshold for token classification
        device: Device to use
        max_length: Maximum sequence length
        save_tokens: Whether to save token classifications
        domain: Domain name (required if save_tokens=True)
        model_name: Model name (required if save_tokens=True)
        output_dir: Output directory for token classifications

    Returns:
        Dict mapping (layer_idx, expert_idx) to
        {'total': int, 'important': int, 'unimportant': int}
    """
    # Step 1: Compute token importance for all examples
    all_tokens = []
    all_importance = []
    token_details = []  # Store detailed info for each token

    print("  Computing token importance...")
    for example_idx, example in enumerate(tqdm(examples, desc="  Token importance", leave=False)):
        question = example.get("question", "")
        options = example.get("options", [])

        # Format prompt
        prompt = question + "\n" + "\n".join(
            [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
        ) + "\nAnswer:"

        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens.input_ids

        # Compute gradient-based importance
        with torch.enable_grad():
            importance = compute_gradient_importance(model, tokenizer, input_ids, device)

        for token_idx, (token_id, score) in enumerate(zip(input_ids[0], importance[0])):
            token_str = tokenizer.decode([token_id]).strip()
            if token_str:
                all_tokens.append(token_str)
                all_importance.append(float(score))
                token_details.append({
                    "example_idx": example_idx,
                    "token_idx": token_idx,
                    "token_id": int(token_id),
                    "token": token_str,
                    "importance_score": float(score),
                })

    # Step 2: Classify tokens as important/unimportant
    important_indices, unimportant_indices = classify_tokens_by_importance(
        all_tokens, all_importance, importance_threshold
    )
    important_indices_set = set(important_indices)
    unimportant_indices_set = set(unimportant_indices)

    print(f"    Important tokens: {len(important_indices)}, Unimportant: {len(unimportant_indices)}")

    # Update token details with classification
    for idx, detail in enumerate(token_details):
        if idx in important_indices_set:
            detail["classification"] = "important"
        elif idx in unimportant_indices_set:
            detail["classification"] = "unimportant"
        else:
            detail["classification"] = "neutral"

    # Save token classifications if requested
    if save_tokens and domain and model_name:
        # Separate important and unimportant tokens for easier analysis
        important_tokens = [d for d in token_details if d["classification"] == "important"]
        unimportant_tokens = [d for d in token_details if d["classification"] == "unimportant"]

        # Get unique tokens with their average importance
        important_unique = {}
        for t in important_tokens:
            tok = t["token"]
            if tok not in important_unique:
                important_unique[tok] = {"count": 0, "total_score": 0.0}
            important_unique[tok]["count"] += 1
            important_unique[tok]["total_score"] += t["importance_score"]

        unimportant_unique = {}
        for t in unimportant_tokens:
            tok = t["token"]
            if tok not in unimportant_unique:
                unimportant_unique[tok] = {"count": 0, "total_score": 0.0}
            unimportant_unique[tok]["count"] += 1
            unimportant_unique[tok]["total_score"] += t["importance_score"]

        # Sort by count
        important_summary = [
            {"token": k, "count": v["count"], "avg_score": v["total_score"] / v["count"]}
            for k, v in sorted(important_unique.items(), key=lambda x: x[1]["count"], reverse=True)
        ]
        unimportant_summary = [
            {"token": k, "count": v["count"], "avg_score": v["total_score"] / v["count"]}
            for k, v in sorted(unimportant_unique.items(), key=lambda x: x[1]["count"], reverse=True)
        ]

        token_data = {
            "domain": domain,
            "model_name": model_name,
            "importance_threshold": importance_threshold,
            "statistics": {
                "total_tokens": len(token_details),
                "num_important": len(important_indices),
                "num_unimportant": len(unimportant_indices),
                "pct_important": len(important_indices) / len(token_details) * 100 if token_details else 0,
                "pct_unimportant": len(unimportant_indices) / len(token_details) * 100 if token_details else 0,
                "importance_score_stats": get_token_importance_stats(all_importance),
            },
            "important_tokens_summary": important_summary[:500],  # Top 500
            "unimportant_tokens_summary": unimportant_summary[:500],  # Top 500
            "all_tokens": token_details,  # Full details
        }

        save_token_classifications(token_data, domain, model_name, output_dir)

    # Step 3 & 4: Capture expert activations and count by importance
    expert_counts = defaultdict(lambda: {'total': 0, 'important': 0, 'unimportant': 0})
    token_offset = 0

    print("  Counting expert activations...")
    for example in tqdm(examples, desc="  Expert activations", leave=False):
        question = example.get("question", "")
        options = example.get("options", [])

        prompt = question + "\n" + "\n".join(
            [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
        ) + "\nAnswer:"

        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens.input_ids
        seq_len = input_ids.shape[1]

        # Capture expert activations
        activations = capture_expert_activations(model, tokenizer, input_ids, device)

        # Update counts based on global token indices
        for token_idx in range(seq_len):
            global_idx = token_offset + token_idx
            is_important = global_idx in important_indices_set
            is_unimportant = global_idx in unimportant_indices_set

            for layer_idx, token_acts in activations.items():
                if token_idx in token_acts:
                    for expert_idx in token_acts[token_idx]:
                        key = (layer_idx, expert_idx)
                        expert_counts[key]['total'] += 1
                        if is_important:
                            expert_counts[key]['important'] += 1
                        elif is_unimportant:
                            expert_counts[key]['unimportant'] += 1

        token_offset += seq_len

    return dict(expert_counts)
