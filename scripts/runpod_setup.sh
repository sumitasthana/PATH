#!/usr/bin/env bash
# Prepare a RunPod pod to continue the experiments that were running on Colab.
#
# The notebooks already have a non-Colab path: they find the checkout by
# walking up from the working directory, read secrets from the environment,
# and keep every result in Cloudflare R2. So a pod needs the checkout, the
# packages, and the three R2 variables. Nothing else.
#
# Run it from the pod's shell. The R2 values go in the pod template's
# environment variables, not in this file.
#
#   bash runpod_setup.sh
#   cd /workspace/PATH && jupyter lab --allow-root --ip 0.0.0.0 --no-browser
#
# Start Jupyter from inside the checkout. The notebooks locate the repository
# relative to where they are opened.
set -euo pipefail

REPO=${REPO:-/workspace/PATH}

for name in R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
  if [ -z "${!name:-}" ]; then
    echo "Missing $name. Set the three R2 variables in the pod template." >&2
    exit 1
  fi
done

# /workspace is the only directory a RunPod network volume keeps between
# pods. Anything written elsewhere disappears when the pod stops.
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only
else
  git clone https://github.com/sumitasthana/PATH.git "$REPO"
fi

# torch and torchvision come with the RunPod PyTorch image. boto3 is what
# talks to R2; the notebooks refuse to install it themselves outside Colab.
pip install --quiet boto3 tqdm matplotlib

cd "$REPO"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# The image data downloads itself on first use (build_tasks(download=True)),
# into data/tiny-imagenet-200 inside the checkout. Keeping the checkout on
# /workspace means the next pod does not download it again.
echo "Ready. Start Jupyter from $REPO."
