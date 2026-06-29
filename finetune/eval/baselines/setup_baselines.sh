#!/usr/bin/env bash
# Set up the UniK3D + Depth-Any-Camera baselines *inside this repo*.
#
# Clones both upstream repos into <repo>/third_party/ — the location
# run_baselines.py looks in by default, so no --unik3d-root/--dac-root or
# env vars are needed afterwards — and installs the one(s) you ask for into the
# CURRENT Python environment.
#
# The two models have conflicting dependencies (UniK3D: python>=3.11, torch>=2.4,
# numpy>=2 ; DAC: numpy<2 + an older torch), so install each inside its OWN env:
#
#   bash finetune/eval/baselines/setup_baselines.sh unik3d   # run in your UniK3D env
#   bash finetune/eval/baselines/setup_baselines.sh dac      # run in your DAC env (also fetches weights)
#   bash finetune/eval/baselines/setup_baselines.sh clone    # just clone both, no pip
#   bash finetune/eval/baselines/setup_baselines.sh both     # clone + install both (only if one env supports both)
#
# Re-runnable: existing clones are pulled, not re-cloned.
set -euo pipefail

TARGET="${1:-both}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # baselines -> eval -> finetune -> vggt-omega
THIRD_PARTY="$REPO_ROOT/third_party"

UNIK3D_URL="https://github.com/lpiccinelli-eth/UniK3D.git"
DAC_URL="https://github.com/yuliangguo/depth_any_camera.git"
UNIK3D_DIR="$THIRD_PARTY/UniK3D"
DAC_DIR="$THIRD_PARTY/depth_any_camera"

echo "[setup] repo root:    $REPO_ROOT"
echo "[setup] third_party:  $THIRD_PARTY"
echo "[setup] target:       $TARGET"
mkdir -p "$THIRD_PARTY"

clone_repo () {  # url dest
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "[setup] $(basename "$dest") already cloned — pulling"
    git -C "$dest" pull --ff-only || echo "[setup]   (pull skipped)"
  else
    echo "[setup] cloning $(basename "$dest")"
    git clone --depth 1 "$url" "$dest"
  fi
}

install_unik3d () {
  echo "[setup] pip install -e UniK3D  (editable — code stays under the repo)"
  pip install -e "$UNIK3D_DIR"
  echo "[setup] UniK3D ready (weights auto-download from HF on first run)"
}

install_dac () {
  echo "[setup] pip install DAC requirements"
  pip install -r "$DAC_DIR/requirements.txt"
  echo "[setup] building MultiScaleDeformableAttention CUDA extension"
  ( cd "$DAC_DIR/dac/models/ops" && python setup.py build install )
  echo "[setup] downloading DAC weights -> $REPO_ROOT/checkpoints/"
  ( cd "$REPO_ROOT" && python -m finetune.eval.baselines.download_weights --variant dac_swinl_indoor )
}

# Always clone both so either model — and `--mode official` — can import them.
clone_repo "$UNIK3D_URL" "$UNIK3D_DIR"
clone_repo "$DAC_URL"    "$DAC_DIR"

case "$TARGET" in
  clone)  echo "[setup] clone-only; skipping pip install" ;;
  unik3d) install_unik3d ;;
  dac)    install_dac ;;
  both)
    echo "[setup] WARNING: UniK3D and DAC have conflicting deps (numpy>=2 vs <2,"
    echo "[setup]          torch>=2.4 vs older) — installing both into ONE env may"
    echo "[setup]          break it. Prefer running this script once per env as"
    echo "[setup]          'unik3d' and 'dac'."
    install_unik3d
    install_dac
    ;;
  *) echo "[setup] usage: $0 [unik3d|dac|both|clone]"; exit 2 ;;
esac

case "$TARGET" in
  unik3d) HINT_MODELS="unik3d" ;;
  dac)    HINT_MODELS="dac" ;;
  *)      HINT_MODELS="dac      # or unik3d, each run in its own env" ;;
esac

echo
echo "[setup] done. Repos under $THIRD_PARTY — run_baselines finds them automatically."
echo "[setup] e.g.: python -m finetune.eval.baselines.run_baselines --mode official \\"
echo "[setup]            --dac-config checkpoints/dac_swinl_indoor.json \\"
echo "[setup]            --dac-weights checkpoints/dac_swinl_indoor.pt --models ${HINT_MODELS}"
