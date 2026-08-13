# Copyright (c) 2026.
"""The comparability token: what makes two evaluation runs the same experiment.

Both experiments freeze a frame list before they score anything, and both answer
"are these two runs comparable?" the same way — **their digests match**. This
module owns that rule, and the even-spacing rule that decides which frames go
into the list. Nothing else.

What is deliberately *not* here
-------------------------------
``fovbench.split.Split`` and ``slambench.split.Split`` stay separate classes, and
they should. They are not near-duplicates that someone failed to merge:

    fovbench    Frame(seq, frame_id, depth, rgb-per-stream); a context window
                over the frame pool; streams; ``PROTOCOL = "adt-fov-v1"``
    slambench   Frame(dataset, take, clip, index, npz, video, clip_frames); a
                context window that must stay inside its clip; grouping by clip
                so each mp4 is decoded once; ``PROTOCOL = "egosynth-slam-v1"``

Merging them would put the two experiments' comparability tokens in one place,
which is the coupling `tests/test_experiment_separation.py` exists to prevent.
Each package keeps its own ``PROTOCOL`` precisely so that a digest from one can
never be mistaken for a digest from the other.

What *is* here is the part where duplication is dangerous rather than merely
repetitive. The digest rule was written twice, identically, and it is the
definition of "these two runs measured the same thing". Two copies of that can
drift silently and then two digests mean different things while still looking
like digests — a failure with no symptom. One rule, imported twice, cannot.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

#: Hex characters kept from the SHA-1. Twelve is 48 bits — far past collision
#: risk for the hundreds of splits this repo will ever produce, and short enough
#: to paste into a ticket, which is what it is for.
DIGEST_CHARS = 12


def digest(protocol: str, keys: Iterable[str]) -> str:
    """The split's identity: a hash over its protocol and its ordered frame keys.

    **Order is part of the identity, not an implementation detail.** Two runs
    over the same frames in a different order pool their per-frame numbers in a
    different sequence, and floating-point addition is not associative — so they
    are not guaranteed to produce the same digits and must not claim to be the
    same split.

    The protocol token is hashed *first* and separately from the keys, so a
    change of protocol moves every digest even when the frame list is untouched.
    That is what stops a run of one experiment being compared against a run of
    the other on the strength of a matching frame list.
    """
    h = hashlib.sha1(protocol.encode())
    for k in keys:
        h.update(k.encode())
    return h.hexdigest()[:DIGEST_CHARS]


def evenly_spaced(items: Sequence, n: int) -> List:
    """``n`` items spread across ``items``, endpoints included, order kept.

    A prefix would sample one continuous stretch of a recording, which for
    egocentric video is one room and one activity; spreading is what makes a
    25-frame split a sample of the take rather than of its first six seconds.

    Deduplicated by index, so asking for more than the sequence holds returns it
    whole rather than repeating entries — a repeated frame would be scored twice
    and would weight itself double in every pooled number.
    """
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / float(n - 1)
    return [items[i] for i in sorted({int(round(i * step)) for i in range(n)})]
