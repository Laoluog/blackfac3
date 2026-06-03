#!/usr/bin/env bash
# One-time setup on a fresh DigitalOcean GPU droplet (Ubuntu 22.04 + NVIDIA/CUDA image).
# Usage: bash scripts/setup_droplet.sh
set -euo pipefail

echo "==> GPU check"
nvidia-smi

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y git python3.10 python3.10-venv python3.10-dev build-essential

echo "==> AdaFace repo (backbone + head)"
sudo mkdir -p /models
if [ ! -d /models/AdaFace ]; then
  sudo git clone https://github.com/mk-minchul/AdaFace /models/AdaFace
fi

echo "==> Python venv (3.10 for mxnet compatibility)"
python3.10 -m venv "$HOME/bf-venv"
# shellcheck disable=SC1091
source "$HOME/bf-venv/bin/activate"
pip install --upgrade pip

echo "==> Install training + eval deps"
pip install -r requirements_train.txt
pip install boto3 matplotlib pytest   # eval extras

echo "==> Sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'/models/AdaFace'); import net, head; print('AdaFace net/head import OK')"

cat <<'EOF'

Setup complete. Next:
  source ~/bf-venv/bin/activate
  # extract data to /data (see runbook), then:
  bash scripts/run_train.sh
EOF
