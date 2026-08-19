# space-container: connect, set up, run

The A100-80GB pod. Read this **before** trying to reach it — every section below
is a mistake that has already been made once and cost a wrong diagnosis.

This file lives in the repo, not in one machine's notes, because it has to be
readable from `lambda_63`, from the pod itself, and from either Claude account.

**When it is the right box:** `slambench`'s `vggt360` arm, which OOMs `lambda_63`
(peaks ~22 GB while another user holds ~26 GB of card 0's 47.4 GB — capacity, not
fragmentation, so it will recur). It also holds
`VGGT-Omega-1B-512/model.pt`, which `lambda_63` does not.
**When it is the wrong box:** `fovbench`. The pod's ADT export has **1** usable
sequence (`depth_npy` + `videos_synthetic` + `videos_rgb`) against `lambda_63`'s
**6**, so it builds a different split and a different digest; the five missing
sequences are 346 GB. Count *usable* sequences, never directory entries — the pod
lists 18 and 17 of them are unusable.

---

## 1. Connect

```bash
ssh space-container 'bash -lc "<command>"'
```

`~/.ssh/config` on the Mac already chains the path, so no hand-built `-J` string
is needed:

```
Mac --> lambda_63 --> space-jumping-host (jumping-host.n6.sr-cloud.com:3307) --> space-container (pod)
```

**`bash -lc` is not optional.** The HTTP proxy is exported from the pod's
`~/.bash_profile`, which bash sources only for **login** shells. A plain
`ssh space-container 'git pull'` gets no proxy and dies of connect timeouts while
DNS still resolves — which reads exactly like an air-gapped box. It is not
air-gapped; this is the single most expensive mistake available here. Corollary:
`env | grep proxy` over a non-login shell is not evidence of anything.

Two harmless things that look like failures on every connect:
`X11 forwarding request failed on channel 0` (from `ForwardX11 yes` in the config
— ignore it), and a first connect that takes **2–4 minutes** to return. Launch any
probe detached rather than blocking a turn on it. Also: **macOS has no `timeout`**,
so do not wrap `ssh` in it — use
`-o ConnectTimeout=25 -o ServerAliveInterval=10 -o ServerAliveCountMax=6`.

### Health check — one command, run it before anything else

This answers "am I on the pod, is the proxy live, is the env intact, is the GPU
free" in a single round trip. Verified working 2026-08-19:

```bash
ssh -o ConnectTimeout=25 -o ServerAliveInterval=10 space-container 'bash -lc "
  hostname
  echo PROXY=\$http_proxy
  /group-volume/Fengjia/envs/vggt360-py312/bin/python -c \"import torch;print(torch.__version__, torch.version.cuda)\"
  nproc
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"'
```

Expected, and what each line is actually gating:

```
run<NNNNNNN>-vggt-omega-360        <- which pod; record it in the run's meta.json
PROXY=http://75.17.107.42:8080     <- non-empty, else you forgot `bash -lc` (§1)
2.11.0+cu128 12.8                  <- MUST say 12.8, not 13.0 (§2)
247                                <- why OMP_NUM_THREADS=16 is mandatory (§4)
NVIDIA A100-SXM4-80GB, 0 MiB, 81920 MiB   x2
```

If the torch line is missing or says `13.0`, or the venv path does not exist, go
to §2 and re-run the bootstrap — that is the whole recovery, and it is idempotent.

### When it will not connect

In this order, and do not skip to 3:

1. **Retry 2–3 times.** The chain drops intermittently — `Connection timed out
   during banner exchange`, `channel 0: open failed: connect failed: Device or
   resource busy` — and clears on its own within a couple of minutes. This is
   the default explanation, not a dead pod.
2. **Distinguish "chain flaky" from "pod recreated"** — over plain retries the
   two are identical. From the jump host, open a bare TCP connection to the
   pod's port 22:
   ```bash
   cat < /dev/null > /dev/tcp/<pod-ip>/22
   ```
   Refused or timed out there means the address is stale, not that the link
   flapped. Note `lambda_63` can reach the jump host itself but **only on port
   3307** — `ssh -J` without `-p 3307` fails with the same banner-exchange
   message as case 1, which has already cost one round of misdiagnosis.
3. **A stale address needs the user, not a retry loop.** The pod is recreated
   with a new name *and* address on some schedule (`run1150238` 2026-08-06,
   `run1151839` 2026-08-11, `run1154132` 2026-08-17, `run1155074` 2026-08-19
   morning, `run1155098` 2026-08-19 evening — that last gap was **hours, not
   weeks**: see the idle-kill warning below). Assume it has recycled since this
   line was written. The
   `space-container` entry in `~/.ssh/config` is kept pointing at the live one;
   `_1`…`_4` are the dead history. Recovering the new address
   needs `space login --region n6`, which is interactive and whose token expires
   after ~10 days — **only the user can do it.** Do not cycle through
   `space-container_1`…`_4` in `~/.ssh/config`: those are confirmed-dead earlier
   pods, kept as history, and trying them proves nothing.

**Never build a long command over this link.** Two launches were lost on
2026-08-17 to the connection closing *while the `tmux` command was being sent*,
leaving no session and no error anyone would notice. `scp` a launcher to
`/group-volume` first, then `ssh` a one-word invocation of it.

### The pod gets killed for sitting idle — this is now confirmed, not theoretical

2026-08-19: a pod (`run1155074`) that had been connected to and left with **0%
GPU utilization while its session spent turns on diagnostics** was gone within
hours — replaced by `run1155098`, same day. That is not the "roughly weekly"
recreation cadence case 3 above describes; it is a direct, fast consequence of
connecting without promptly putting real work on the GPUs. **Once connected,
treat "GPUs at 0%" as a clock running out**, not as room to explore:
run the health check, then go straight to launching an actual ticket's job —
do not spend multiple turns re-deriving environment facts that are already
answered in §2 below, and do not leave a connection open "to look around."
If you catch yourself running read-only diagnostics for more than a couple of
minutes with both GPUs idle, stop and launch something.

One unresolved, lower-confidence observation from the same day: several
`ssh -o BatchMode=yes -o ConnectTimeout=N space-container '...'` attempts in a
row all failed with `Connection timed out during banner exchange` against a pod
whose port 22 was confirmed open from the jump host (case 2 above), and a bare
`ssh space-container '...'` with no extra flags succeeded immediately
afterward. It is not established whether `BatchMode=yes` itself is the problem
(vs. coincidental timing of the chain recovering) — but if flagged connection
attempts keep failing banner exchange while the port is confirmed open, try the
plain form before concluding the pod is down.

---

## 2. Set up the environment

**Nothing outside `/group-volume` survives a pod recreation** — `$HOME`, `/usr`,
`/opt/mlp-conda`, and therefore every `pip install`. So the env lives on
`/group-volume` and is rebuilt per pod by a committed, idempotent script:

```bash
ssh space-container 'bash -lc "/group-volume/Fengjia/projects/vggt-omega-023/space_container_bootstrap.sh"'
```

It is already staged on the pod at that path, and **the env normally survives a
recreation** because it is on `/group-volume` — confirmed on `run1155074`
(2026-08-19), where the health check above passed with no rebuild at all. So run
the health check first and only bootstrap if it fails; the script is idempotent
either way, but a clean rebuild costs ~3 GB of downloads over a flaky mirror.

That script is [`space_container_bootstrap.sh`](../../space_container_bootstrap.sh)
at the repo root; its header comments carry the full reasoning. Do **not**
hand-install packages instead. Four things it exists to get right:

* **The site pip mirror serves a torch that silently breaks comparability.**
  `/etc/pip.conf` points at `bart.sec.samsung.net`, whose `torch==2.11.0` is
  PyPI's default wheel — **`+cu130`**, pulling `nvidia-cublas-13.x` / `cudnn-cu13`
  and reporting `torch.version.cuda == "13.0"`. It *runs*, so nothing announces
  the problem; it is simply a different numerical stack from the one every
  published number in this repo came off. `lambda_63` is `2.11.0+cu128` /
  `torchvision 0.26.0+cu128` / `transformers 5.15.0` / `numpy 1.26.4`, and
  `+cu128` is served only by `https://download.pytorch.org/whl/cu128` (reachable
  through the proxy: HTTP 200 with it, refused without). **Assert
  `torch.version.cuda == "12.8"`, never trust the version string alone.**
* **Pin with a pip constraints file, not versions in the install list.**
  `pip install numpy==1.26.4` does not keep numpy there — opencv and scipy both
  declare `numpy>=2` and a later line silently upgraded it to 2.5.2. numpy 2
  changed scalar type-promotion, i.e. different arithmetic under every metric in
  both benchmarks, and nothing fails. The script writes
  `$VENV/lambda63-constraints.txt` and passes `-c` on every install.
* **That mirror drops large transfers mid-body and pip's `--retries` does not
  cover it.** A one-shot install of torch (530 MB + ~2.5 GB of nvidia deps)
  exhausts pip's five retries on `RemoteDisconnected` and leaves the venv with
  no torch at all. Retry at the shell level, one package per call, with a wheel
  cache on `/group-volume/Fengjia/envs/wheels` so each attempt starts further
  along.
* **`sudo apt-get` is NOPASSWD** (`sudo -n -l` lists apt-get/apt/aptitude/apt-key,
  and sudoers keeps the proxy vars through `env_reset`). Testing with
  `sudo -n true` says "a password is required" and means nothing — `/usr/bin/true`
  is not in the rule. Test the actual command. As of 2026-08-17 nothing
  system-level is missing anyway.

Env path: `/group-volume/Fengjia/envs/vggt360-py312`, built on `/usr/bin/python3.12`
(the image's python — `$HOME/.local/bin/python` is ephemeral). Activate with
`source /group-volume/Fengjia/envs/vggt360-py312/bin/activate`. The script
deliberately leaves `/opt/mlp-conda` alone: a venv is reversible with `rm -rf`, a
mutated shared base env is not.

**Put `HF_HOME` on the persistent mount too** — `export HF_HOME=/group-volume/Fengjia/hf-cache`.
The default `~/.cache/huggingface` dies with the pod, and VGGT-1B is a 9.4 GB
re-download each time. The stock image cache has Prior-Depth-Anything, moge-2,
bert-base and DA-V2-Small but **not** VGGT-1B.

---

## 3. Move data — do not use SSH for it

`file://groups/SR-TORAIC-IVU/Fengjia/...` **is** the pod's
`/group-volume/Fengjia/...` — the same network volume. So `space storage`, run
**from `lambda_63`**, moves bulk data in either direction **without the pod being
up at all**, sidestepping every flakiness and staleness problem in §1.

```bash
# lambda_63 -> pod
space storage upload   file <local-path> file://groups/SR-TORAIC-IVU/Fengjia/... --recursive
# pod's /group-volume -> lambda_63  (also how to recover results after a pod dies)
space storage download file file://groups/SR-TORAIC-IVU/Fengjia/... <dest> --recursive --match <RE2>
```

Worked examples in the repo: [`stage_egosynth_to_pod.sh`](../../stage_egosynth_to_pod.sh),
[`stage_adt024_to_pod.sh`](../../stage_adt024_to_pod.sh).

Mechanics that `--help` does not usefully explain:

* **`--match` is a Go (RE2) regexp, not a glob.** `*foo*` dies with "missing
  argument to repetition operator". No negative lookahead, so exclude by matching
  the file extension rather than the directory name.
* It applies to the whole key. **Top-level files need no leading `/` while nested
  ones effectively do**, so a pattern anchored `/meta\.json$` silently skips the
  take's own `meta.json`. Always check the member list actually landed.
* **Downloading a *directory* puts its contents flat into the destination.** Give
  each source directory its own destination directory — four arms downloaded into
  one dest silently overwrite each other into a single plausible-looking result.
* Startup is ~1 s, so **one invocation per unit of work** is cheap and buys
  per-unit resumability. Throughput: 8 workers ≈ 17 MB/s, **16 workers ≈ 38 MB/s**
  downloading; upload is far slower at ~3.6 MB/s with
  `--max-request-processes 16`. Budget accordingly.
* **Gate on file counts against the producer's manifest, never on exit codes**, or
  a truncated take passes silently.

---

## 4. Run experiments

**Run benchmark work from `/group-volume/Fengjia/projects/vggt-omega-023`, not
`vggt-omega-organized`.** See "Two repo hazards on the pod" at the end of this
file for why the latter is unsafe to `reset`/rebase against — this is worth
knowing before you `cd` anywhere, not just at the end. `vggt-omega-023` is a
clean checkout kept in sync with `organized`; `git fetch && git merge
origin/organized` there is safe (watch for untracked local copies of scripts
that later got committed upstream — `git status --short` first, move any
conflicting untracked files aside with `mv`, never `git checkout -f`/`clean`).

**Always set `OMP_NUM_THREADS=16` (plus `MKL_`/`OPENBLAS_`). This is worth more
than any other setting here.** The pod is a two-socket AMD EPYC 7742 with **247
visible threads**; `lambda_63` has 64. Unlimited, numpy/OpenMP opens a pool across
all 247 and both NUMA sockets for ops far too small to pay for it. Measured on one
take, three frames, vggt360 arm: **default 1m59s wall / 83m09s CPU** vs
**`OMP_NUM_THREADS=16` 1m15s wall / 1m34s CPU**. The first #023 slambench attempt
ran unlimited at 1.1 take-frames/min against lambda's 8.1 — **a 7x slowdown on the
faster GPU.** The diagnosis is the memorable part: the GPU was idle ~90% of the
time at full clock and `/group-volume` read 198 MB in 0.4 s, so both obvious
suspects were innocent.
It **does not move the numbers** — those two runs are bit-identical on every
metric at the same split digest. Note the opposite caution holds on `lambda_63`,
where `OMP_NUM_THREADS=1` was deliberately refused as a `fovbench` deadlock fix,
since BLAS thread count *can* move a `scale_shift` fit going through `lstsq`.
That retirement is measured for `slambench` only; re-measure before assuming it
for `fovbench`.

**Gate on the split digest before the forward passes, not after.** A wrong pool
is the one failure that makes every number in an arm unpublishable, and the check
costs seconds against hours. Verified 2026-08-17: the pod prints
`split 61195914f090`, identical to #020's published split, clip counts matching
exactly (aea 251, nymeria 256). Check it anyway — it is one grep and it is the
only thing that licenses tabulating a pod number beside a lambda one.

**Launch in a named tmux session**, per the tmux rule in global `CLAUDE.md`, and
if a background waiter greps the log for a completion marker, **wrap the command
and the marker in a subshell before the pipe**:

```bash
( CMD; echo MARKER_$? ) 2>&1 | tee logfile
```

`CMD 2>&1 | tee log; echo EXIT=$?` is broken twice over: the `echo`'s stdout is
not part of the piped stream so the marker never reaches the *file* (only the
pane), and without `pipefail` that `$?` is `tee`'s status, not the job's — a
crashed job reports 0. Both were hit on 2026-08-18.

**A blocking waiter must reconnect per probe** rather than hold one long `ssh`;
otherwise a dropped link and a finished job are indistinguishable, and one waiter
did exit early on `client_loop: send disconnect: Broken pipe`. Have it treat SSH
failure as its own reported state, and check `space storage list` before
diagnosing a hang.

Worked launcher examples, with their reasoning in the headers:
[`run_slam_023_pod.sh`](../../run_slam_023_pod.sh),
[`run_fov024_B_pod.sh`](../../run_fov024_B_pod.sh).

### If the pod dies mid-run, the results are not lost

On 2026-08-18 four `fovbench` arms all exited 0 and the pod stopped answering SSH
shortly after — port 22 closed on **all five** pod IPs in `~/.ssh/config` while
the jump host answered normally (§1 case 3's signature, not case 1's). Everything
survived, because the repo and its `eval_out` live on `/group-volume`. Recover
with the `space storage download` path in §3, from `lambda_63`. The pod can die
in the window between a run finishing and a poller noticing, so a finished half
looks exactly like a stuck one.

### Two repo hazards on the pod

* `/group-volume/Fengjia/projects/vggt-omega-organized` carries **`ef924bc
  "results"`, authored by the user 2026-07-30 and on no remote** — seven commits
  diverged, 85 behind. **Never `reset --hard` that tree.** Benchmark runs use the
  separate clean checkout `vggt-omega-023` instead.
* The pod's git identity is misconfigured as `Tristan A.A.
  <tristan.a@samsung.com>`, so commits made there are misattributed (they are the
  user's own work). Outbound `git push` from the pod is unreliable even with the
  proxy — the working pattern is to move results off with §3 and push from the
  Mac.
