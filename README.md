# Domain-specific Experts

Code for the paper "Do Domain-specific Experts Exist in MoE-based LLMs?"

This repository contains the expert specialization analysis pipeline: token
importance scoring, MoE router activation capture, expert importance scoring,
and domain steering.

## Repository Layout

```text
analysis_specialize/      Python package for the analysis pipeline
configs/sample_config.yaml
configs/domain_steering_config.yaml
run_sample.sh             Full sample run
data/sample_mcqa.jsonl    Tiny MCQA smoke-test data
requirements.txt          Python dependencies
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default
`pip install torch` is not appropriate for your machine.

## Data Format

For public runs, pass a JSON or JSONL file through `data_path` in the config or
through `--data_path`. Each record should contain:

```json
{"domain": "biology", "question": "Question text?", "options": ["A", "B", "C", "D"]}
```

Accepted aliases are `category`, `subject`, or `field` for `domain`, and
`choices` for `options`. If `data_path` is not set, the code first tries the
local `src.data` loader used in internal experiments. If that is unavailable,
it uses tiny placeholder examples for smoke testing only.

## Domain Steering

```bash
bash run_sample.sh
```

This runs expert discovery first, then runs domain steering with the top-N
domain experts selected from the saved expert scores. Equivalent direct
commands:

```bash
python -m analysis_specialize.main --config configs/sample_config.yaml
python -m analysis_specialize.domain_steering --config configs/domain_steering_config.yaml
```

The sample config uses a small MoE model and two placeholder examples when no
dataset is configured. Outputs are written to `output/sample_run/` and
`output/domain_steering/`.


## Run With Your Data

Edit `configs/sample_config.yaml` or pass overrides from the command line:

```bash
python -m analysis_specialize.main \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --data_path /path/to/examples.jsonl \
  --domains biology physics chemistry \
  --sample_percentage 10 \
  --max_samples 100 \
  --output_dir output/qwen_moe_example
```

Large MoE models require sufficient GPU memory and may need Hugging Face access
permissions depending on the model.


## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{do2026domainspecificexpertsexistmoebased,
      title={Do Domain-specific Experts exist in MoE-based LLMs?}, 
      author={Giang Do and Hung Le and Truyen Tran},
      year={2026},
      eprint={2604.05267},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05267}, 
}
```
