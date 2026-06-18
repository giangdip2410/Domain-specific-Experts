#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m analysis_specialize.main --config configs/sample_config.yaml
python -m analysis_specialize.domain_steering --config configs/domain_steering_config.yaml
