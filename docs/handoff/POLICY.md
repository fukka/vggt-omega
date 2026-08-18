# Two-Claude split policy

One Claude account is burning tokens doing work that never needed the GPU box.
This splits the work across two accounts that share nothing but a GitHub repo.

| | **GPU-Claude** | **CPU-Claude** |
|---|---|---|
| Runs on | the machine with SSH access to `lambda_63` | any machine, no access to `lambda_63` |
| Owns | launching + babysitting GPU jobs | everything else |
| Sees | datasets, weights, GPUs, conda envs | the repo, and whatever GPU-Claude commits |
| Transfers via | `results` branch + GitHub issues | `organized` branch + GitHub issues |

Repo: `github.com/fukka/vggt-omega`. Both accounts need `gh auth login`.

---

## The rule

**GPU-Claude does only what physically requires the GPU box.** Everything else is
a ticket for CPU-Claude.

Requires the box (GPU-Claude):
* `raytun3r.train` / `raytun3r.eval` and any run touching real weights
* anything reading ADT (`/user/f.zhang2/Documents/...`) or ScanNet++
  (`/netapp/datasets/...`) — those paths exist on no other machine
* conda env construction, weight downloads, CUDA/driver debugging
* diagnostics that need real data (baseline-vs-stride profiling, UFM matching)

Does **not** require the box (CPU-Claude):
* reading the paper; extracting equations, protocol, hyperparameters
* paper↔code fidelity review — this is pure code reading
* writing and refactoring code
* running `pytest raytun3r/tests` and `python raytun3r/smoke_test.py` — **the whole
  suite is CPU-only and finishes in under a minute** (73 tests + 33 checks, no
  weights, no data, no GPU; one test skips without `depth_anything_3`). This is the
  single biggest win: nearly all code work can be written *and verified* without
  the box. On this Mac the interpreter that has torch is
  `~/opt/anaconda3/bin/python` (3.8.3 / torch 2.2.2 / cv2 4.4.0) — **not** any
  `python3` on `PATH`, which is 3.7 with torch 1.0.1.
* interpreting `results.json` once it is on the `results` branch
* README, docs, plots, tables

Grey area → CPU-Claude. If it turns out to need the box, it becomes a `gpu` ticket.

---

## Transfer protocol

Two channels only. No shared filesystem exists.

### Code: `organized` branch

CPU-Claude commits code and pushes. GPU-Claude pulls before every run:

```bash
git -C /user/f.zhang2/projects/vggt-omega-organized pull --ff-only origin organized
```

CPU-Claude never edits a file while a GPU job is mid-run against it; GPU-Claude
pins the commit it ran by recording `git rev-parse HEAD` in the result JSON.

### Results: `results` branch

GPU-Claude pushes run artifacts there and nothing else. Keep it small — JSON and
trimmed logs, never checkpoints or images:

```bash
gh issue comment <N> --body "run done, results branch @ $(git rev-parse --short HEAD)"
```

Layout:

```
results/<run-id>/results.json     eval output
results/<run-id>/train_log.json   loss history
results/<run-id>/meta.json        commit, GPU, matcher, env, wall-clock
```

### Tickets: GitHub issues

Labels `gpu` and `cpu` say who owns it. One owner at a time; the label *is* the
lock. To hand back, relabel and comment. Ticket bodies follow
[`tickets/TEMPLATE.md`](tickets/TEMPLATE.md).

```bash
gh issue create --label cpu --title "..." --body-file docs/handoff/tickets/<name>.md
gh issue list --label gpu --state open
```

---

## Working concurrently without collisions

* **Disjoint file ownership per ticket.** A ticket names the files it may touch.
  Two open tickets must not name the same file. This is what makes parallel work
  safe without either side seeing the other's diff.
* **CPU-Claude never pushes to `results`; GPU-Claude never pushes code to
  `organized`** except a one-line config or a revert to unblock a run.
* **Long runs are fire-and-forget.** GPU-Claude launches detached
  (`setsid nohup ... &`), writes the run-id to the ticket, and stops. It does not
  sit and poll.

---

## Token rules for GPU-Claude

These are where this session actually leaked tokens:

1. **Never poll in a loop from the model.** One blocking waiter script that exits
   on completion, or a scheduled check — not a turn per 30 s.
2. **Filter logs on the remote.** `ssh ... 'grep -aE "^\[eval\]|^\[fit\] done" log'`,
   never `cat` a log into context.
3. **Never read a source file to understand it.** That is a `cpu` ticket. Read
   only the few lines needed to launch or unblock a run.
4. **Batch SSH.** One `ssh` doing five things beats five round-trips.
5. **Don't re-derive what is already committed.** If a number is in
   `results/*.json`, cite it; do not re-run.

## Token rules for CPU-Claude

1. Verify with the local test suite before handing back — it is ~10 s and it
   catches shape, convention, and no-op-adapter regressions.
2. Never ask GPU-Claude for something derivable from the repo.
3. One ticket = one commit = one push. Small diffs review cheaply.
