#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain expert steering evaluation.

This stage consumes weighted expert-score files from analysis_specialize.main,
selects the top-N experts for each target domain, applies router steering for
those experts during inference, and compares MCQA accuracy against the unsteered
baseline.
"""

import argparse
import gc
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from tqdm import tqdm

try:
    from .main import load_model_and_tokenizer
    from .token_expert_analysis import ensure_dir, safe_model_name
except ImportError:
    from main import load_model_and_tokenizer
    from token_expert_analysis import ensure_dir, safe_model_name


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DOMAIN_KEYS = ("domain", "category", "subject", "field")
ANSWER_KEYS = ("answer", "correct_answer", "target", "label")
ANSWER_INDEX_KEYS = ("answer_index", "correct_index", "label_index")
ROUTER_ATTRIBUTES = ("gate", "router", "gating_network")
EXPERT_ATTRIBUTES = ("experts", "local_experts", "mlp_experts")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")


@dataclass
class DomainSteeringConfig:
    """Configuration for domain steering evaluation."""

    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B"
    device: str = "auto"

    expert_scores_path: str = "output/expert_discovery"
    data_path: str = ""
    domains: List[str] = field(default_factory=lambda: ["biology"])

    sample_percentage: float = 100.0
    max_samples_per_domain: int = 50
    random_seed: int = 42
    max_sequence_length: int = 2048

    steering_strength: float = 0.10
    steering_mode: str = "multiplicative"  # "multiplicative" or "additive"
    top_k_experts: int = 50

    output_dir: str = "output/domain_steering"
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DomainSteeringConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_domain(value: Any) -> str:
    """Normalize a domain label for matching."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def read_records(data_path: str) -> List[Dict[str, Any]]:
    """Read JSON or JSONL records."""
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
            for key in ("data", "examples", "records", "test"):
                if isinstance(data.get(key), list):
                    return data[key]

    raise ValueError("data_path must point to a JSON list or JSONL file")


def normalize_text(value: Any) -> str:
    """Normalize text for answer matching."""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_options(raw_options: Any) -> Tuple[List[str], Dict[str, str]]:
    """Normalize option structures into an ordered text list and optional labels."""
    if raw_options is None:
        return [], {}

    labels_to_text = {}
    if isinstance(raw_options, dict):
        for key, value in raw_options.items():
            if isinstance(value, dict):
                value = value.get("text") or value.get("value") or value.get("answer") or value
            labels_to_text[str(key).strip().upper()] = str(value)
        sorted_items = sorted(labels_to_text.items(), key=lambda item: item[0])
        return [text for _, text in sorted_items], labels_to_text

    if not isinstance(raw_options, list):
        raw_options = [raw_options]

    options = []
    for option in raw_options:
        if isinstance(option, dict):
            label = option.get("label") or option.get("key")
            text = option.get("text") or option.get("value") or option.get("answer") or option
            if label is not None:
                labels_to_text[str(label).strip().upper()] = str(text)
            options.append(str(text))
        else:
            options.append(str(option))

    return options, labels_to_text


def answer_to_letter(record: Dict[str, Any], options: Sequence[str], labels_to_text: Dict[str, str]) -> Optional[str]:
    """Convert an answer field to an option letter."""
    for key in ANSWER_INDEX_KEYS:
        if key in record and record[key] is not None:
            try:
                idx = int(record[key])
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(options):
                return LETTERS[idx]
            if 1 <= idx <= len(options):
                return LETTERS[idx - 1]

    answer = next((record.get(key) for key in ANSWER_KEYS if record.get(key) is not None), None)
    if answer is None:
        return None

    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return LETTERS[answer]
        if 1 <= answer <= len(options):
            return LETTERS[answer - 1]

    answer_text = str(answer).strip()
    answer_label = answer_text.rstrip(".:)").upper()
    if len(answer_label) == 1 and answer_label in LETTERS[:len(options)]:
        return answer_label

    if answer_label in labels_to_text:
        label_index = list(sorted(labels_to_text.keys())).index(answer_label)
        if label_index < len(options):
            return LETTERS[label_index]

    normalized_answer = normalize_text(answer_text)
    for idx, option in enumerate(options):
        if normalize_text(option) == normalized_answer:
            return LETTERS[idx]

    return None


def normalize_example(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw MCQA record to the format used by the evaluator."""
    question = record.get("question") or record.get("prompt")
    raw_options = record.get("options") or record.get("choices")
    if not question or raw_options is None:
        return None

    options, labels_to_text = normalize_options(raw_options)
    if not options or len(options) > len(LETTERS):
        return None

    answer = answer_to_letter(record, options, labels_to_text)
    if answer is None:
        return None

    return {
        "question": str(question),
        "options": options,
        "answer": answer,
    }


def load_domain_examples(
    data_path: str,
    domain: str,
    sample_percentage: float,
    max_samples: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    """Load and sample evaluation examples for a domain."""
    target_domain = canonical_domain(domain)
    examples = []

    for record in read_records(data_path):
        if not isinstance(record, dict):
            continue
        record_domain = next((record.get(key) for key in DOMAIN_KEYS if record.get(key)), None)
        if record_domain is None or canonical_domain(record_domain) != target_domain:
            continue
        example = normalize_example(record)
        if example is not None:
            examples.append(example)

    if not examples:
        return []

    rng = random.Random(random_seed)
    n_samples = min(
        max(1, int(len(examples) * sample_percentage / 100.0)),
        max_samples,
        len(examples),
    )
    return rng.sample(examples, n_samples)


def resolve_expert_scores_file(
    path_or_dir: str,
    model_name: str,
    domain: str,
) -> str:
    """Resolve one domain expert-score file from a file or directory."""
    path = Path(path_or_dir)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Expert scores path not found: {path_or_dir}")

    model_dir = path / safe_model_name(model_name)
    filename = f"{domain}_trend_results_importance_weighted_scores.json"
    direct_path = model_dir / filename
    if direct_path.exists():
        return str(direct_path)

    matches = sorted(path.glob(f"**/{filename}"))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f"No expert score file found for domain '{domain}' under {path_or_dir}")
    raise ValueError(
        f"Multiple expert score files found for domain '{domain}'. Pass one explicit file: "
        + ", ".join(str(match) for match in matches[:5])
    )


def load_expert_scores_for_domain(
    path_or_dir: str,
    model_name: str,
    domain: str,
) -> Dict[str, Any]:
    """Load a saved expert-score file for one domain."""
    scores_file = resolve_expert_scores_file(path_or_dir, model_name, domain)
    with open(scores_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_scores_file"] = scores_file
    return data


def select_top_experts_from_scores(
    scores_data: Dict[str, Any],
    top_k_experts: int,
) -> Dict[int, List[int]]:
    """Select the top-N experts globally by weighted score."""
    if top_k_experts <= 0:
        raise ValueError("top_k_experts must be positive")
    if "per_layer" not in scores_data:
        raise KeyError("Expert score file must contain a 'per_layer' section")

    selected = []
    for layer_key, experts in scores_data["per_layer"].items():
        layer_idx = int(layer_key)
        for expert_key, item in experts.items():
            expert_idx = int(expert_key)
            score = float(item.get("weighted_score", 0.0))
            selected.append((layer_idx, expert_idx, score))

    selected.sort(key=lambda row: row[2], reverse=True)
    selected = selected[:top_k_experts]

    experts_by_layer = defaultdict(list)
    for layer_idx, expert_idx, _ in selected:
        experts_by_layer[layer_idx].append(expert_idx)

    return {layer_idx: sorted(set(experts)) for layer_idx, experts in experts_by_layer.items()}


def flatten_selected_experts(experts_by_layer: Dict[int, Sequence[int]], scores_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return selected expert metadata for saved results."""
    selected = []
    per_layer = scores_data.get("per_layer", {})
    for layer_idx, experts in sorted(experts_by_layer.items()):
        layer_scores = per_layer.get(str(layer_idx), {})
        for expert_idx in experts:
            score = layer_scores.get(str(expert_idx), {}).get("weighted_score", 0.0)
            selected.append({
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "weighted_score": score,
            })
    selected.sort(key=lambda item: item["weighted_score"], reverse=True)
    return selected


def extract_layer_index(module_name: str) -> Optional[int]:
    """Extract a transformer layer index from a module name."""
    match = LAYER_RE.search(module_name)
    if match:
        return int(match.group(1))
    return None


def shift_router_tensor(
    tensor: torch.Tensor,
    expert_indices: Sequence[int],
    logit_delta: float,
) -> torch.Tensor:
    """Apply an additive logit shift to selected experts in a router tensor."""
    if not expert_indices or tensor.ndim == 0:
        return tensor
    if max(expert_indices) >= tensor.shape[-1]:
        return tensor

    shifted = tensor.clone()
    index = torch.tensor(expert_indices, dtype=torch.long, device=shifted.device)
    delta = torch.tensor(logit_delta, dtype=shifted.dtype, device=shifted.device)
    source = torch.ones_like(shifted.index_select(shifted.ndim - 1, index)) * delta
    shifted.index_add_(
        dim=shifted.ndim - 1,
        index=index,
        source=source,
    )
    return shifted


def shift_router_output(output: Any, expert_indices: Sequence[int], logit_delta: float) -> Any:
    """Shift the first router output tensor whose last dimension can index experts."""
    if isinstance(output, torch.Tensor):
        return shift_router_tensor(output, expert_indices, logit_delta)

    if isinstance(output, tuple):
        updated = list(output)
        for idx, value in enumerate(updated):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and max(expert_indices, default=-1) < value.shape[-1]:
                updated[idx] = shift_router_tensor(value, expert_indices, logit_delta)
                return tuple(updated)
        return output

    if isinstance(output, list):
        updated = list(output)
        for idx, value in enumerate(updated):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and max(expert_indices, default=-1) < value.shape[-1]:
                updated[idx] = shift_router_tensor(value, expert_indices, logit_delta)
                return updated
        return output

    return output


class DomainSteering:
    """Context manager that applies domain-specific router steering hooks."""

    def __init__(
        self,
        model,
        experts_by_layer: Dict[int, Sequence[int]],
        steering_strength: float = 0.10,
        steering_mode: str = "multiplicative",
        verbose: bool = True,
    ):
        self.model = model
        self.experts_by_layer = experts_by_layer
        self.steering_strength = steering_strength
        self.steering_mode = steering_mode
        self.verbose = verbose
        self.handles = []
        self.patched_layers = []

        if steering_mode == "multiplicative":
            if steering_strength <= -1.0:
                raise ValueError("multiplicative steering_strength must be greater than -1")
            self.logit_delta = math.log1p(steering_strength)
        elif steering_mode == "additive":
            self.logit_delta = steering_strength
        else:
            raise ValueError("steering_mode must be 'multiplicative' or 'additive'")

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False

    def apply(self) -> None:
        """Register router forward hooks."""
        seen_router_ids = set()

        for module_name, module in self.model.named_modules():
            layer_idx = extract_layer_index(module_name)
            if layer_idx is None or layer_idx not in self.experts_by_layer:
                continue
            if not any(hasattr(module, attr) for attr in EXPERT_ATTRIBUTES):
                continue

            router = None
            for attr in ROUTER_ATTRIBUTES:
                if hasattr(module, attr):
                    router = getattr(module, attr)
                    break
            if router is None or id(router) in seen_router_ids:
                continue

            expert_indices = tuple(self.experts_by_layer[layer_idx])
            handle = router.register_forward_hook(self._make_hook(expert_indices))
            self.handles.append(handle)
            self.patched_layers.append(layer_idx)
            seen_router_ids.add(id(router))

        if self.verbose:
            print(f"[+] Domain steering hooks applied to {len(self.handles)} layer(s): {sorted(self.patched_layers)}")
        if not self.handles and self.verbose:
            print("[!] No router hooks were applied. Check model architecture and selected expert layers.")

    def remove(self) -> None:
        """Remove all registered steering hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, expert_indices: Sequence[int]):
        def hook(_module, _inputs, output):
            return shift_router_output(output, expert_indices, self.logit_delta)

        return hook


def format_prompt(example: Dict[str, Any]) -> str:
    """Format one MCQA prompt."""
    prompt = example["question"].strip() + "\n\nOptions:\n"
    for idx, option in enumerate(example["options"]):
        prompt += f"{LETTERS[idx]}. {option}\n"
    valid_letters = ", ".join(LETTERS[:len(example["options"])])
    prompt += f"\nAnswer ({valid_letters}):"
    return prompt


def get_input_device(model) -> torch.device:
    """Return the device expected for input IDs."""
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    max_sequence_length: int,
) -> float:
    """Score a candidate continuation by summed log probability."""
    continuation_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids
    prompt_max_length = max(1, max_sequence_length - continuation_ids.shape[1])
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
        return_tensors="pt",
    ).input_ids
    input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)

    device = get_input_device(model)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    continuation_start = prompt_ids.shape[1] - 1
    log_probs = torch.log_softmax(logits[:, continuation_start:, :], dim=-1)
    target_ids = targets[:, continuation_start:]
    token_scores = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_scores.sum().item())


def predict_mcqa(
    model,
    tokenizer,
    example: Dict[str, Any],
    max_sequence_length: int,
) -> Tuple[str, Dict[str, float]]:
    """Predict an MCQA answer by scoring answer-letter continuations."""
    prompt = format_prompt(example)
    scores = {}
    for idx in range(len(example["options"])):
        letter = LETTERS[idx]
        scores[letter] = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=f" {letter}",
            max_sequence_length=max_sequence_length,
        )
    prediction = max(scores.items(), key=lambda item: item[1])[0]
    return prediction, scores


def evaluate_domain(
    model,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    domain: str,
    max_sequence_length: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate MCQA accuracy for one domain."""
    model.eval()
    correct = 0
    predictions = []

    iterator: Iterable[Dict[str, Any]] = examples
    if verbose:
        iterator = tqdm(examples, desc=f"Evaluating {domain}", leave=False)

    for idx, example in enumerate(iterator):
        predicted, scores = predict_mcqa(model, tokenizer, example, max_sequence_length)
        is_correct = predicted == example["answer"]
        correct += int(is_correct)
        predictions.append({
            "example_idx": idx,
            "prediction": predicted,
            "answer": example["answer"],
            "correct": is_correct,
            "scores": scores,
        })

    total = len(examples)
    accuracy = correct / total if total else 0.0
    return {
        "domain": domain,
        "num_examples": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def save_domain_steering_results(results: Dict[str, Any], model_name: str, output_dir: str) -> str:
    """Save domain-steering results."""
    model_output_dir = ensure_dir(os.path.join(output_dir, safe_model_name(model_name)))
    output_file = os.path.join(model_output_dir, "domain_steering_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_file


def run_domain_steering(config: DomainSteeringConfig) -> Dict[str, Any]:
    """Run baseline and domain-steered evaluation."""
    if not config.data_path:
        raise ValueError("Domain steering requires data_path with question/options/answer records")

    if config.verbose:
        print("=" * 70)
        print("DOMAIN STEERING EVALUATION")
        print("=" * 70)
        print(f"Model: {config.model_name}")
        print(f"Domains: {', '.join(config.domains)}")
        print(f"Expert scores: {config.expert_scores_path}")
        print(f"Data: {config.data_path}")
        print(f"Steering: {config.steering_mode}, strength={config.steering_strength}")
        print(f"Top experts per domain: {config.top_k_experts}")
        print("=" * 70)

    examples_by_domain = {}
    for domain in config.domains:
        examples = load_domain_examples(
            data_path=config.data_path,
            domain=domain,
            sample_percentage=config.sample_percentage,
            max_samples=config.max_samples_per_domain,
            random_seed=config.random_seed,
        )
        if examples:
            examples_by_domain[domain] = examples
            if config.verbose:
                print(f"[+] {domain}: loaded {len(examples)} evaluation examples")
        elif config.verbose:
            print(f"[!] {domain}: no valid evaluation examples found")

    if not examples_by_domain:
        raise ValueError("No evaluation examples were loaded")

    model, tokenizer = load_model_and_tokenizer(config.model_name, config.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_results = {}
    steered_results = {}
    selected_experts = {}
    score_files = {}

    try:
        if config.verbose:
            print("\n[1/2] Baseline evaluation")
        for domain, examples in examples_by_domain.items():
            baseline_results[domain] = evaluate_domain(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                domain=domain,
                max_sequence_length=config.max_sequence_length,
                verbose=config.verbose,
            )
            if config.verbose:
                acc = baseline_results[domain]["accuracy"]
                print(f"  {domain}: baseline accuracy = {acc:.2%}")

        if config.verbose:
            print("\n[2/2] Domain-steered evaluation")
        for domain, examples in examples_by_domain.items():
            scores_data = load_expert_scores_for_domain(
                path_or_dir=config.expert_scores_path,
                model_name=config.model_name,
                domain=domain,
            )
            experts_by_layer = select_top_experts_from_scores(
                scores_data=scores_data,
                top_k_experts=config.top_k_experts,
            )
            score_files[domain] = scores_data["_scores_file"]
            selected_experts[domain] = flatten_selected_experts(experts_by_layer, scores_data)
            total_experts = sum(len(experts) for experts in experts_by_layer.values())

            if config.verbose:
                print(f"\n  {domain}: loaded scores from {scores_data['_scores_file']}")
                print(f"  {domain}: steering top {total_experts} expert(s)")

            if total_experts == 0:
                steered_results[domain] = {
                    **baseline_results[domain],
                    "note": "No top experts selected; steered result equals baseline.",
                }
                continue

            with DomainSteering(
                model=model,
                experts_by_layer=experts_by_layer,
                steering_strength=config.steering_strength,
                steering_mode=config.steering_mode,
                verbose=config.verbose,
            ):
                steered_results[domain] = evaluate_domain(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    domain=domain,
                    max_sequence_length=config.max_sequence_length,
                    verbose=config.verbose,
                )

            if config.verbose:
                base_acc = baseline_results[domain]["accuracy"]
                steer_acc = steered_results[domain]["accuracy"]
                print(f"  {domain}: steered accuracy = {steer_acc:.2%} (delta {steer_acc - base_acc:+.2%})")

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    deltas = {
        domain: steered_results[domain]["accuracy"] - baseline_results[domain]["accuracy"]
        for domain in steered_results
    }
    summary = {
        "baseline_average": sum(item["accuracy"] for item in baseline_results.values()) / len(baseline_results),
        "steered_average": sum(item["accuracy"] for item in steered_results.values()) / len(steered_results),
        "average_delta": sum(deltas.values()) / len(deltas) if deltas else 0.0,
    }

    results = {
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "score_files": score_files,
        "selected_experts": selected_experts,
        "baseline": baseline_results,
        "steered": steered_results,
        "deltas": deltas,
        "summary": summary,
    }

    output_file = save_domain_steering_results(results, config.model_name, config.output_dir)
    if config.verbose:
        print("\n" + "=" * 70)
        print(f"Domain steering results saved to: {output_file}")
        print(f"Average baseline: {summary['baseline_average']:.2%}")
        print(f"Average steered:  {summary['steered_average']:.2%}")
        print(f"Average delta:    {summary['average_delta']:+.2%}")
        print("=" * 70)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate domain steering with top-N experts from saved expert scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--expert_scores_path", type=str, default="output/expert_discovery")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--domains", type=str, nargs="+", default=["biology"])
    parser.add_argument("--sample_percentage", type=float, default=100.0)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--steering_strength", type=float, default=0.10)
    parser.add_argument("--steering_mode", type=str, default="multiplicative", choices=["multiplicative", "additive"])
    parser.add_argument("--top_k_experts", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="output/domain_steering")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config = DomainSteeringConfig.from_yaml(args.config)
    else:
        config = DomainSteeringConfig(
            model_name=args.model,
            device=args.device,
            expert_scores_path=args.expert_scores_path,
            data_path=args.data_path,
            domains=args.domains,
            sample_percentage=args.sample_percentage,
            max_samples_per_domain=args.max_samples,
            random_seed=args.seed,
            max_sequence_length=args.max_length,
            steering_strength=args.steering_strength,
            steering_mode=args.steering_mode,
            top_k_experts=args.top_k_experts,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    run_domain_steering(config)


if __name__ == "__main__":
    main()
