#!/usr/bin/env bash
# Rebuild the space-container run environment. IDEMPOTENT -- safe to re-run, and
# it has to be re-run after the pod is recreated.
#
# WHY THIS SCRIPT EXISTS
#
# The pod is recreated on its own schedule with a new name and a new address
# (run1150238 on 2026-08-06, run1151839 on 2026-08-11, run1154132 on 2026-08-17),
# and everything outside /group-volume goes with it: $HOME, /usr, /opt/mlp-conda,
# and therefore every pip install. So the environment lives on /group-volume,
# which is the one persistent mount, and this script rebuilds it from nothing if
# the image itself has moved.
#
#   ssh space-container 'bash -lc "/group-volume/Fengjia/projects/vggt-omega-023/space_container_bootstrap.sh"'
#
# Note the `bash -lc`. The HTTP proxy is exported from ~/.bash_profile, which
# bash sources only for LOGIN shells, so a plain `ssh host 'cmd'` gets no proxy
# and every pip install dies of a connect timeout that looks like an air-gapped
# pod. This is the single most expensive mistake available on this box.
#
# WHAT IT DELIBERATELY DOES NOT DO
#
# It does not touch /opt/mlp-conda, even though that base env is writable. That
# is the pod image's shared python and other work of the user's runs against it.
# A venv under /group-volume is reversible by `rm -rf`; a mutated base env is not.
#
# It installs no system packages, because none are missing -- cv2 imports (so
# libGL is present) and the `av` wheels carry their own ffmpeg. It could if it
# needed to: `sudo -n -l` shows apt-get, apt, aptitude and apt-key are all
# NOPASSWD for this user, and the sudoers Defaults keep the proxy variables
# through env_reset, so `sudo apt-get install <pkg>` works unattended. (Testing
# this with `sudo -n true` says "a password is required" and means nothing --
# /usr/bin/true is not in the rule. Test the actual command.) Anything installed
# that way is as ephemeral as the venv would be outside /group-volume, so it
# belongs in this script rather than in a one-off shell.
set -euo pipefail

VENV=/group-volume/Fengjia/envs/vggt360-py312
BASE_PYTHON=/usr/bin/python3.12   # the image's python, not $HOME/.local's

# The stack is pinned to lambda_63's, NOT to the newest wheels, and that is the
# whole point of the pins. fovbench and slambench numbers are only readable if
# every cell in a table came off one stack: run C is read against run A, and the
# ticket 023 self-check demands bit-identity against a reference produced on
# lambda_63. The pod's own base env ships torch 2.12.0+cu130 against lambda_63's
# 2.11.0+cu128, so using it would mix a numerics term into every comparison.
#
# CUDA 12.8 on this pod is also the safer of the two: the driver here is
# 535.183.01, and CUDA 12.x runs on it under minor-version compatibility while
# CUDA 13 wants r580+.
#
# torch comes from download.pytorch.org and NOT from the site mirror, which is
# the opposite of what is convenient. The site's Artifactory mirrors PyPI, and
# PyPI's default linux wheel for torch 2.11.0 is `+cu130` -- installing it pulls
# nvidia-cublas-13.x, cudnn-cu13, nccl-cu13, and reports
# `torch.version.cuda == 13.0`. It runs here (a 2048^2 matmul is fine, despite
# the driver reporting 535.183.01), so nothing announces the problem: it is
# simply a different numerical stack from the one every published number in this
# repo came off. download.pytorch.org is reachable through the proxy -- HTTP 200
# with it, refused without it -- and the `+cu128` local version is only served
# there.
TORCH_CU128_INDEX=https://download.pytorch.org/whl/cu128
TORCH_REQS=(torch==2.11.0+cu128 torchvision==0.26.0+cu128)
REQS=(
  numpy==1.26.4
  transformers==5.15.0
  omegaconf addict einops
  projectaria_tools av opencv-python-headless
  pytest safetensors huggingface_hub pillow scipy matplotlib pandas tqdm
)

# A pip CONSTRAINTS file, not just pins in REQS, and the difference is the whole
# point. Installing `numpy==1.26.4` early does not keep it: opencv and scipy both
# declare `numpy>=2`, so a later line in REQS silently upgraded it to 2.5.2 and
# the verify step caught it. numpy 2 changed the scalar type-promotion rules, so
# that is not a cosmetic version difference -- it is a different arithmetic under
# every metric in both benchmarks. Constraints apply to the whole resolve, so
# nothing can raise it behind our back; if a package genuinely cannot live with
# numpy 1.26 the install fails loudly here instead of producing numbers later.
#
# Versions are lambda_63's, read off the raytun3r env on 2026-08-17.
CONSTRAINTS="$VENV/lambda63-constraints.txt"
write_constraints() {
  cat > "$CONSTRAINTS" <<'EOF'
numpy==1.26.4
scipy==1.17.1
opencv-python-headless==4.11.0.86
pillow==11.3.0
transformers==5.15.0
huggingface_hub==1.26.0
safetensors==0.8.0
torch==2.11.0+cu128
torchvision==0.26.0+cu128
EOF
}

say() { printf '\n=== %s\n' "$*"; }

# The site's pip index is `bart.sec.samsung.net`, which is inside `no_proxy`, so
# it is reached directly -- and it drops large transfers under load. torch is a
# 530 MB wheel and its nvidia-* dependencies are another ~2.5 GB, so a one-shot
# `pip install` reliably exhausts pip's own five retries on
# `RemoteDisconnected('Remote end closed connection without response')` and leaves
# the venv with no torch at all. pip's --retries does not cover this: the failure
# is mid-body, not at connect.
#
# So retry at the shell level, and keep every wheel that does land in a cache on
# the persistent mount. Each attempt therefore starts further along than the last
# and a rebuild after the pod is recreated is nearly free.
WHEELS=/group-volume/Fengjia/envs/wheels
pip_retry() {  # pip_retry <attempts> <pip args...>
  local n="$1"; shift
  local i
  for i in $(seq 1 "$n"); do
    if "$PIP" install --retries 5 --timeout 120 \
         -c "$CONSTRAINTS" --find-links "$WHEELS" "$@"; then
      return 0
    fi
    echo "--- attempt $i/$n failed; caching what landed and retrying" >&2
    "$PIP" download --retries 5 --timeout 120 -c "$CONSTRAINTS" -d "$WHEELS" "$@" || true
  done
  return 1
}

# Same retry, against the pytorch index. Kept separate so the cu128 pin can never
# silently resolve against the site mirror, which would hand back a cu130 build.
pip_retry_torch() {
  local n="$1"; shift
  local i
  for i in $(seq 1 "$n"); do
    if "$PIP" install --retries 5 --timeout 300 \
         --index-url "$TORCH_CU128_INDEX" -c "$CONSTRAINTS" \
         --find-links "$WHEELS" "$@"; then
      return 0
    fi
    echo "--- torch attempt $i/$n failed; caching what landed and retrying" >&2
    "$PIP" download --retries 5 --timeout 300 \
      --index-url "$TORCH_CU128_INDEX" -d "$WHEELS" "$@" || true
  done
  return 1
}

say "proxy check"
if [ -z "${http_proxy:-}" ]; then
  echo "http_proxy is unset -- you are not in a login shell. Re-run as:" >&2
  echo "  ssh space-container 'bash -lc \"$0\"'" >&2
  exit 2
fi
echo "http_proxy=$http_proxy"

say "base python"
"$BASE_PYTHON" -V

if [ ! -x "$VENV/bin/python" ]; then
  say "creating venv at $VENV"
  mkdir -p "$(dirname "$VENV")"
  "$BASE_PYTHON" -m venv "$VENV"
else
  say "venv already present at $VENV"
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
mkdir -p "$WHEELS"
write_constraints
"$PIP" install --retries 5 --timeout 120 --upgrade pip wheel

# torch first and from its own index. If a cu130 torch is already installed --
# which is what a previous run against the site mirror leaves behind -- take it
# out first, so pip cannot decide the requirement is already satisfied.
say "torch, from ${TORCH_CU128_INDEX}"
if "$PY" -c 'import torch,sys; sys.exit(0 if torch.version.cuda != "12.8" else 1)' \
     2>/dev/null; then
  echo "removing a non-cu128 torch left by an earlier attempt"
  "$PIP" uninstall -y -q torch torchvision || true
fi
pip_retry_torch 8 "${TORCH_REQS[@]}"

# One package at a time. The point is not tidiness: a single `pip install` of the
# whole list restarts the entire resolve when any one transfer drops, so torch
# gets re-fetched because pandas failed. Per-package, the retry is scoped to the
# package that actually failed.
for req in "${REQS[@]}"; do
  say "$req"
  pip_retry 6 "$req"
done

# depth_anything_3 resolves a torch of its own choosing if allowed to, which would
# silently undo the pin above -- hence --no-deps. Its three real imports
# (omegaconf, addict, einops) are in REQS. This is what GPU_EXPERIMENTS.md
# section 0 prescribes.
say "depth_anything_3 (--no-deps, so it cannot move torch)"
pip_retry 6 --no-deps depth-anything-3

say "verify"
"$PY" - <<'EOF'
import importlib, sys
want = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
        "transformers": "5.15.0", "numpy": "1.26.4"}
bad = []
for m in ("torch", "torchvision", "transformers", "numpy", "depth_anything_3",
          "projectaria_tools", "av", "cv2", "scipy", "pytest"):
    try:
        mod = importlib.import_module(m)
        got = getattr(mod, "__version__", "?")
    except Exception as e:
        print("%-18s MISSING (%s)" % (m, type(e).__name__)); bad.append(m); continue
    flag = ""
    if m in want and got != want[m]:
        flag = "  <-- WANTED %s" % want[m]; bad.append(m)
    print("%-18s %s%s" % (m, got, flag))
import torch
print("cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
# The pin above is on the release, so assert the CUDA build separately -- this is
# the half that has to agree with lambda_63 for a number to be comparable.
print("torch.version.cuda:", torch.version.cuda)
if torch.version.cuda != "12.8":
    print("  <-- WANTED 12.8, to match lambda_63. Numbers from this env are NOT"
          " comparable to the published tables until this agrees.")
    bad.append("torch.version.cuda")
sys.exit(1 if bad else 0)
EOF

say "done -- activate with:  source $VENV/bin/activate"
