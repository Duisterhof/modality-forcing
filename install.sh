#!/usr/bin/env bash
# Robust installer: detect the GPU driver's CUDA version, install a matching
# PyTorch build FIRST (so `accelerate` et al. don't drag in a mismatched one),
# then the remaining dependencies. Run inside your activated virtualenv:
#
#   python -m venv .venv && source .venv/bin/activate
#   bash install.sh
#
set -euo pipefail

# May be multi-word, e.g. PIP="uv pip" — hence the unquoted expansions below.
PIP="${PIP:-pip}"

pick_cuda_index() {
  # Echo the best-matching PyTorch wheel index for this machine's driver.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"; return
  fi
  # "CUDA Version: 12.8" -> 12.8 (the max CUDA the driver supports).
  local cuda
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
  if [ -z "$cuda" ]; then echo "cpu"; return; fi
  local major="${cuda%%.*}" minor="${cuda#*.}"
  # Pick the newest PyTorch index <= the driver's CUDA. A cu12x wheel runs on
  # any >= that 12.x driver (CUDA minor-version compatibility), so an exact
  # match isn't required — just don't exceed the driver's major.minor.
  if   [ "$major" -ge 13 ]; then echo "cu130"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then echo "cu128"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then echo "cu126"
  elif [ "$major" -eq 12 ]; then echo "cu121"
  else echo "cu118"; fi
}

IDX="$(pick_cuda_index)"
if [ "$IDX" = "cpu" ]; then
  echo "==> No usable NVIDIA driver detected (no nvidia-smi, or no CUDA version reported) — installing CPU-only PyTorch."
  $PIP install torch --index-url "https://download.pytorch.org/whl/cpu"
else
  echo "==> Detected CUDA $IDX — installing matching PyTorch from the official index."
  $PIP install torch --index-url "https://download.pytorch.org/whl/${IDX}"
fi

echo "==> Installing the remaining dependencies."
$PIP install -r requirements.txt

echo "==> Verifying the install."
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. If you have a GPU, the torch build "
          "does not match your driver — re-run with the right index or check "
          "`nvidia-smi`.")
PY
echo "==> Done."
