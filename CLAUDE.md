# Project instructions

This repo is worked on by two separate Claude accounts under a GPU/CPU split.
The full protocol, and everything else project-specific, lives in
`docs/handoff/` — not here, so it stays readable from every machine (this repo
is the only channel both accounts share).

**Read these before doing anything else, in this order:**

1. [`docs/handoff/POLICY.md`](docs/handoff/POLICY.md) — the GPU/CPU split rule.
   Determines which account you are and what you're allowed to touch.
2. If you have SSH access to `lambda_63`, you are GPU-Claude — also read
   [`docs/handoff/tickets/`](docs/handoff/tickets/) for the current ticket.
   If you don't, you are CPU-Claude — read
   [`docs/handoff/CPU_CLAUDE.md`](docs/handoff/CPU_CLAUDE.md).
3. **Before running any command against `space-container`
   (the A100-80GB pod) — every time, not just the first time — read
   [`docs/handoff/SPACE_CONTAINER.md`](docs/handoff/SPACE_CONTAINER.md) in
   full.** This is not optional and not a one-time read: connecting, environment
   setup, data staging and job launching on that box each have non-obvious
   failure modes that have already cost wrong diagnoses more than once. Do not
   improvise a python path, a repo checkout, or a thread-count setting instead
   of reading it — every one of those has silently produced wrong numbers or a
   7x slowdown before. **If the pod's GPUs sit idle for more than a few
   minutes after you connect, the pod gets reclaimed** — read the doc, then
   move straight to launching real work; don't spend many turns on ad hoc
   diagnostics with the GPUs sitting at 0%.

Do not re-derive any of the above from scratch by exploring the codebase —
it has already been worked out and written down specifically so that doesn't
need to happen again.
