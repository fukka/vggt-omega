# <title>

**Owner:** cpu | gpu
**Files I may touch:** `path/a.py`, `path/b.py`   ← must not overlap another open ticket
**Blocked by:** #<issue> | none

## Goal

One or two sentences. What is true when this is done.

## Context

What is already known, with links to `results/<run-id>/results.json` or a file:line.
Do not restate the repo; link it.

## Steps

1. ...
2. ...

## Done when

- [ ] `python -m pytest raytun3r/tests -q` passes
- [ ] `python raytun3r/smoke_test.py` passes
- [ ] <ticket-specific check>
- [ ] pushed to `organized`, issue commented with the sha

## Needs a GPU run afterwards?

yes → relabel `gpu` and say exactly which command to run
no  → close it
