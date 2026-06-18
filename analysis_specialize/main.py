#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for domain expert discovery.

This script runs the complete pipeline:
1. Load model and data
2. Compute token importance (gradient-based)
3. Classify tokens as important/unimportant
4. Capture expert activations
5. Count activations for important vs unimportant tokens
6. Save weighted expert scores for domain steering

Usage:
    python -m analysis_specialize.main --model Qwen/Qwen1.5-MoE-A2.7B --domains biology physics chemistry

    python -m analysis_specialize.main --config configs/sample_config.yaml
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch
import yaml
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

try:
    from .token_expert_analysis import analyze_domain_data, ensure_dir, safe_model_name
    from .expert_importance import compute_expert_importance_scores, save_expert_scores
except ImportError:
    from token_expert_analysis import analyze_domain_data, ensure_dir, safe_model_name
    from expert_importance import compute_expert_importance_scores, save_expert_scores


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for domain expert discovery."""
    # Model settings
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B"
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Domain settings
    domains: List[str] = field(default_factory=lambda: [
        "biology", "physics", "chemistry", "computer_science",
        "economics", "math", "law", "psychology", "history",
        "philosophy", "business", "health", "engineering", "other"
    ])
    data_path: Optional[str] = None

    # Sampling settings
    sample_percentage: float = 10.0  # Percentage of data to sample per domain
    max_samples_per_domain: int = 100  # Maximum samples per domain
    random_seed: int = 42

    # Token importance settings
    importance_threshold: float = 15.0  # Percentile threshold
    max_sequence_length: int = 2048

    # Output settings
    output_dir: str = "output/expert_discovery"
    save_expert_scores: bool = True
    save_token_classifications: bool = True  # Save important/unimportant tokens
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Data Loading
# =============================================================================

def _canonical_domain(value: Any) -> str:
    """Normalize domain labels for comparison."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _read_records(data_path: str) -> List[Dict[str, Any]]:
    """Read examples from a JSON or JSONL file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "records"):
                if isinstance(data.get(key), list):
                    return data[key]

    raise ValueError("data_path must point to a JSON list or JSONL file")


def _normalize_example(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw record to the question/options format used by the pipeline."""
    question = record.get("question") or record.get("prompt")
    options = record.get("options") or record.get("choices") or []

    if not question:
        return None
    if isinstance(options, dict):
        options = list(options.values())
    if not isinstance(options, list):
        options = [options]

    normalized_options = []
    for option in options:
        if isinstance(option, dict):
            option = option.get("text") or option.get("value") or option.get("answer") or option
        normalized_options.append(str(option))

    return {
        "question": str(question),
        "options": normalized_options,
    }


def _load_examples_from_file(data_path: str, domain: str) -> List[Dict[str, Any]]:
    """Load examples for one domain from a public JSON/JSONL file."""
    domain_keys = ("domain", "category", "subject", "field")
    target_domain = _canonical_domain(domain)
    examples = []

    for record in _read_records(data_path):
        if not isinstance(record, dict):
            continue

        record_domain = next((record.get(key) for key in domain_keys if record.get(key)), None)
        if record_domain is None or _canonical_domain(record_domain) != target_domain:
            continue

        example = _normalize_example(record)
        if example is not None:
            examples.append(example)

    return examples


def _sample_examples(
    examples: List[Dict[str, Any]],
    domain: str,
    sample_percentage: float,
    max_samples: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    """Sample domain examples deterministically."""
    if not examples:
        print(f"  [!] No examples found for domain '{domain}'")
        return []

    random.seed(random_seed)
    num_to_sample = min(
        max(1, int(len(examples) * sample_percentage / 100)),
        max_samples,
        len(examples),
    )
    sampled = random.sample(examples, num_to_sample)

    print(f"  {domain}: {len(sampled)} samples (from {len(examples)} total)")
    return sampled


def load_domain_data(
    domain: str,
    sample_percentage: float,
    max_samples: int,
    random_seed: int,
    data_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load and sample data for a domain.

    If data_path is provided, it must be a JSON or JSONL file with records that
    include a domain/category/subject/field label plus question and options fields.
    If data_path is not provided, the loader tries a local src.data integration
    and falls back to small placeholder data for smoke testing.

    Args:
        domain: Domain name
        sample_percentage: Percentage to sample
        max_samples: Maximum number of samples
        random_seed: Random seed for reproducibility
        data_path: Optional JSON/JSONL data file

    Returns:
        List of example dicts
    """
    if data_path:
        examples = _load_examples_from_file(data_path, domain)
        return _sample_examples(
            examples=examples,
            domain=domain,
            sample_percentage=sample_percentage,
            max_samples=max_samples,
            random_seed=random_seed,
        )

    # Try to import from src if available
    try:
        from src.data import load_mmlu_pro, filter_by_domain

        data = load_mmlu_pro()
        all_examples = filter_by_domain(data, domain, "mmlu_pro", max_samples=100000)

        return _sample_examples(
            examples=all_examples,
            domain=domain,
            sample_percentage=sample_percentage,
            max_samples=max_samples,
            random_seed=random_seed,
        )

    except ImportError:
        print(f"  [!] Could not import data module. Using placeholder data for '{domain}'")
        # Return placeholder data for testing
        return [
            {
                "question": f"Sample question about {domain}?",
                "options": ["Option A", "Option B", "Option C", "Option D"]
            }
            for _ in range(min(10, max_samples))
        ]


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """
    Load model and tokenizer.

    Args:
        model_name: Model name or path
        device: Device to use

    Returns:
        Tuple of (model, tokenizer)
    """
    # Try to import from src if available
    try:
        from src.models import load_model_and_tokenizer as load_func
        return load_func(model_name)
    except ImportError:
        pass

    # Fallback: use transformers directly
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device == "cpu":
        model = model.to(device)

    model.eval()

    return model, tokenizer


# =============================================================================
# Main Pipeline
# =============================================================================

def save_expert_discovery_summary(
    results: Dict[str, Any],
    model_name: str,
    output_dir: str,
) -> str:
    """Save a compact summary of the expert discovery run."""
    model_output_dir = ensure_dir(os.path.join(output_dir, safe_model_name(model_name)))
    output_file = os.path.join(model_output_dir, "expert_discovery_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_file


def run_expert_discovery(config: Config) -> Dict[str, Any]:
    """
    Run expert discovery and save weighted expert scores.

    Args:
        config: Configuration object

    Returns:
        Results dictionary
    """
    # Determine device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    # Print configuration
    if config.verbose:
        print("=" * 70)
        print("DOMAIN EXPERT DISCOVERY")
        print("=" * 70)
        print(f"Model: {config.model_name}")
        print(f"Domains: {', '.join(config.domains)}")
        print(f"Sample percentage: {config.sample_percentage}%")
        print(f"Importance threshold: {config.importance_threshold} percentile")
        print(f"Device: {device}")
        print("=" * 70)

    # Step 1: Load model and tokenizer
    if config.verbose:
        print("\n[1/3] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device)

    # Step 2: Load data for each domain
    if config.verbose:
        print("\n[2/3] Loading domain data...")
    domain_examples = {}
    for domain in config.domains:
        examples = load_domain_data(
            domain=domain,
            sample_percentage=config.sample_percentage,
            max_samples=config.max_samples_per_domain,
            random_seed=config.random_seed,
            data_path=config.data_path,
        )
        if examples:
            domain_examples[domain] = examples

    if not domain_examples:
        print("[!] No valid domains found, exiting")
        return {}

    # Step 3: Analyze each domain
    if config.verbose:
        print("\n[3/3] Analyzing expert activations per domain...")

    domain_expert_counts = {}
    domain_score_files = {}
    domain_top_experts_preview = {}
    for domain, examples in domain_examples.items():
        if config.verbose:
            print(f"\n  Processing domain: {domain}")

        expert_counts = analyze_domain_data(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            importance_threshold=config.importance_threshold,
            device=device,
            max_length=config.max_sequence_length,
            save_tokens=config.save_token_classifications,
            domain=domain,
            model_name=config.model_name,
            output_dir=config.output_dir,
        )
        domain_expert_counts[domain] = expert_counts

        # Optionally save expert importance scores
        if config.save_expert_scores:
            scores = compute_expert_importance_scores(expert_counts)
            score_file = save_expert_scores(
                expert_scores=scores,
                domain=domain,
                model_name=config.model_name,
                output_dir=config.output_dir,
            )
            domain_score_files[domain] = score_file
            top_preview = sorted(
                (
                    {
                        "layer_idx": layer_idx,
                        "expert_idx": expert_idx,
                        "weighted_score": data["weighted_score"],
                    }
                    for (layer_idx, expert_idx), data in scores.items()
                ),
                key=lambda item: item["weighted_score"],
                reverse=True,
            )[:10]
            domain_top_experts_preview[domain] = top_preview

            if config.verbose:
                print(f"  Top experts for {domain}:")
                for rank, expert in enumerate(top_preview[:5], 1):
                    print(
                        f"    {rank}. Layer {expert['layer_idx']}, Expert {expert['expert_idx']}: "
                        f"score={expert['weighted_score']:.6f}"
                    )

    # Prepare results
    results = {
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "score_files": domain_score_files,
        "top_experts_preview": domain_top_experts_preview,
        "num_experts_tracked": {
            domain: len(expert_counts)
            for domain, expert_counts in domain_expert_counts.items()
        },
    }

    output_file = save_expert_discovery_summary(
        results=results,
        model_name=config.model_name,
        output_dir=config.output_dir,
    )

    if config.verbose:
        print("\n" + "=" * 70)
        print(f"Results saved to: {output_file}")
        print("=" * 70)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find domain-specific experts by token importance and router activation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides other arguments)"
    )

    # Model settings
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
        help="Model name or path"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )

    # Domain settings
    parser.add_argument(
        "--domains", type=str, nargs="+",
        default=["biology", "physics", "chemistry", "computer_science",
                 "economics", "math", "law", "psychology", "history",
                 "philosophy", "business", "health", "engineering", "other"],
        help="Domains to analyze"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Optional JSON/JSONL dataset with domain, question, and options fields"
    )

    # Sampling settings
    parser.add_argument(
        "--sample_percentage", type=float, default=10.0,
        help="Percentage of data to sample per domain"
    )
    parser.add_argument(
        "--max_samples", type=int, default=100,
        help="Maximum samples per domain"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    # Token importance settings
    parser.add_argument(
        "--importance_threshold", type=float, default=15.0,
        help="Percentile threshold for token importance classification"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Maximum sequence length"
    )

    # Output settings
    parser.add_argument(
        "--output_dir", type=str, default="output/expert_discovery",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no_save_scores", action="store_true",
        help="Do not save expert importance scores"
    )
    parser.add_argument(
        "--save_tokens", action="store_true", default=True,
        help="Save important/unimportant token classifications (default: True)"
    )
    parser.add_argument(
        "--no_save_tokens", action="store_false", dest="save_tokens",
        help="Do not save token classifications"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    # Utility
    parser.add_argument(
        "--save_config", type=str, default=None,
        help="Save configuration to YAML file and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config from file or create from args
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config(
            model_name=args.model,
            device=args.device,
            domains=args.domains,
            data_path=args.data_path,
            sample_percentage=args.sample_percentage,
            max_samples_per_domain=args.max_samples,
            random_seed=args.seed,
            importance_threshold=args.importance_threshold,
            max_sequence_length=args.max_length,
            output_dir=args.output_dir,
            save_expert_scores=not args.no_save_scores,
            save_token_classifications=args.save_tokens,
            verbose=not args.quiet,
        )

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
        return

    # Run the pipeline
    run_expert_discovery(config)


if __name__ == "__main__":
    main()
