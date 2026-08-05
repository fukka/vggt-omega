# Copyright (c) 2026.
"""Ray-Aware Global Alignment -- paper Sec. 3.3, Eq. 12 and supp. Eq. S4.

Two stages turn a bag of two-view predictions into one globally consistent
reconstruction.

**Scene-graph pruning.**  Edges are dropped when
(a) a reciprocal pair disagrees -- ``R_j->i`` must be ``R_i->j^T`` to within
``tau_rot``, and the translation directions must be opposite to within
``tau_tra``; or (b) the two point sets barely overlap, measured as the fraction
of mutual-nearest-neighbour matches within a radius (paper: reject below 20%).
This is what removes "doppelganger" pairs -- similar-looking, geometrically
incompatible views.

**Global optimization.**  Eq. 12 minimizes, over global poses ``P_i`` and
per-view scales ``s_i``,

    sum_{(i,j) in E} sum_u  sigma_ij(u) || P_i(s_i x_i(u)) - P_j(s_j x_j(u)) ||^2

The "ray-aware" part is Eq. S4: a point is ``exp(log d + log s) * r`` with the
ray direction ``r`` **frozen**, so points may only slide along their own optical
ray.  Ordinary bundle adjustment lets points move freely, which on a fisheye
quietly re-optimizes the reconstruction toward a pinhole-consistent one -- the
ablation in the paper attributes a 55.5 -> 82.6 mAA jump on 360Loc to this
constraint.

The gauge is fixed by pinning view 0 to identity with unit scale; the objective
is otherwise invariant to a global similarity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from cam3r.geometry import geodesic_angle, make_se3, rot6d_to_matrix


# --------------------------------------------------------------------------- #
# Eq. S4 -- ray conditioning
# --------------------------------------------------------------------------- #

def ray_conditioned_points(
    rays: torch.Tensor, radial: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """``exp(log d + log s) * r`` -- scale a point cloud strictly along its rays.

    ``rays`` (N, 3) unit directions, ``radial`` (N,) distances, ``log_scale`` a
    scalar.  Working in log space keeps the scale positive without a clamp.
    """
    return (torch.exp(torch.log(radial.clamp(min=1e-8)) + log_scale)).unsqueeze(-1) * rays


def split_ray_and_radius(points: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) points -> ``(unit rays, radial distances)``."""
    radial = points.norm(dim=-1).clamp(min=eps)
    return points / radial.unsqueeze(-1), radial


# --------------------------------------------------------------------------- #
# Scene graph
# --------------------------------------------------------------------------- #

@dataclass
class PairwiseEdge:
    """One two-view prediction, as consumed by global alignment.

    ``points_i``/``points_j`` are corresponding 3D points (N, 3) in the *own
    camera frame* of view ``i`` and view ``j`` respectively -- row ``k`` of one
    is the same scene point as row ``k`` of the other.  ``conf`` is the per-
    correspondence sigma from Eq. 12.  ``R``/``t`` are the network's relative
    pose for this edge, used by reciprocal-consistency pruning.  ``overlap`` is
    the MNN match fraction measured when the edge was built from pointmaps.

    Use :meth:`from_pointmaps` to build one from raw network output; the direct
    constructor is for callers that already hold correspondences.
    """

    i: int
    j: int
    points_i: torch.Tensor
    points_j: torch.Tensor
    conf: torch.Tensor
    R: Optional[torch.Tensor] = None
    t: Optional[torch.Tensor] = None
    overlap: Optional[float] = None

    def __post_init__(self) -> None:
        if self.points_i.shape != self.points_j.shape:
            raise ValueError(
                f"edge ({self.i},{self.j}): correspondence counts differ "
                f"{tuple(self.points_i.shape)} vs {tuple(self.points_j.shape)}"
            )

    @classmethod
    def from_pointmaps(
        cls,
        i: int,
        j: int,
        X_ii: torch.Tensor,
        X_ji: torch.Tensor,
        X_jj: torch.Tensor,
        conf_i: Optional[torch.Tensor] = None,
        conf_j: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        radius: float = 0.1,
    ) -> "PairwiseEdge":
        """Build an edge from a two-view prediction, via MNN in the common frame.

        ``X_ii`` (N, 3) is view ``i``'s pointmap in its own frame; ``X_ji``
        (M, 3) is view ``j``'s pointmap expressed in frame ``i`` (Eq. 4); and
        ``X_jj`` (M, 3) is view ``j``'s pointmap in its own frame.  Matching
        happens between ``X_ii`` and ``X_ji`` -- both in frame ``i``, which is
        the only place the two views are comparable -- and the surviving
        correspondences are then stored in each view's own frame, which is what
        Eq. 12 optimizes over.

        ``conf_i`` is indexed on view ``i``'s grid and ``conf_j`` on view
        ``j``'s; they are kept as separate arguments because the two are
        gathered with *different* index vectors.  Passing one map and letting it
        be indexed by the other view's indices silently permutes the weights --
        the arrays are the same length, so nothing complains.  When both are
        given, Eq. 12's sigma_ij is their elementwise minimum: a correspondence
        is only as trustworthy as its least certain endpoint.
        """
        idx_a, idx_b, dist = mutual_nearest_neighbors(X_ii, X_ji)
        keep = dist < radius
        idx_a, idx_b = idx_a[keep], idx_b[keep]
        overlap = float(keep.sum()) / max(int(X_ii.shape[0]), 1)

        parts = []
        if conf_i is not None:
            parts.append(conf_i[idx_a])
        if conf_j is not None:
            parts.append(conf_j[idx_b])
        if parts:
            c = parts[0] if len(parts) == 1 else torch.minimum(parts[0], parts[1])
        else:
            c = torch.ones(len(idx_a), dtype=X_ii.dtype, device=X_ii.device)
        return cls(i, j, X_ii[idx_a], X_jj[idx_b], c, R=R, t=t, overlap=overlap)


def _cdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise distances, forced onto the exact (non-matmul) kernel.

    ``torch.cdist``'s default expands ``|a-b|^2 = |a|^2 + |b|^2 - 2a.b``, which
    catastrophically cancels for near-coincident points: identical float32
    clouds come back with self-distances up to 1e-3 rather than 0, which is
    enough to break a mutual-nearest-neighbour test at a tight radius.
    """
    return torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist")


def mutual_nearest_neighbors(
    A: torch.Tensor, B: torch.Tensor, chunk: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual nearest neighbours between two (N, 3)/(M, 3) point sets.

    Returns ``(idx_a, idx_b, dist)`` for pairs that are each other's nearest.
    """
    if A.numel() == 0 or B.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=A.device)
        return empty, empty, torch.empty(0, device=A.device)

    nn_ab, d_ab = [], []
    for s in range(0, A.shape[0], chunk):
        d = _cdist(A[s : s + chunk], B)
        v, i = d.min(dim=1)
        nn_ab.append(i)
        d_ab.append(v)
    nn_ab = torch.cat(nn_ab)
    d_ab = torch.cat(d_ab)

    nn_ba = []
    for s in range(0, B.shape[0], chunk):
        nn_ba.append(_cdist(B[s : s + chunk], A).min(dim=1).indices)
    nn_ba = torch.cat(nn_ba)

    idx_a = torch.arange(A.shape[0], device=A.device)
    keep = nn_ba[nn_ab] == idx_a
    return idx_a[keep], nn_ab[keep], d_ab[keep]


def overlap_ratio(A: torch.Tensor, B: torch.Tensor, radius: float = 0.1) -> float:
    """Fraction of ``A`` with a mutual nearest neighbour in ``B`` within ``radius``."""
    if A.shape[0] == 0:
        return 0.0
    _, _, dist = mutual_nearest_neighbors(A, B)
    return float((dist < radius).sum()) / float(A.shape[0])


def prune_scene_graph(
    edges: Sequence[PairwiseEdge],
    min_overlap: float = 0.2,
    overlap_radius: float = 0.1,
    tau_rot_deg: float = 15.0,
    tau_tra_deg: float = 30.0,
    gate: str = "quantile",
) -> List[PairwiseEdge]:
    """Drop geometrically inconsistent or barely-overlapping edges.

    The overlap criterion is an **adaptive quantile gate**, per supp. Sec. C.1:
    *"Rather than using a fixed threshold, we apply an adaptive quantile gate:
    edges whose n_e falls below the 20th percentile of the scene-wide match
    distribution are removed."*  The main text's "<20% of the pixel count" reads
    like a fixed cut, but the supplementary overrides it -- and the difference
    matters: a fixed cut is all-or-nothing across a scene whose absolute match
    counts depend on texture and resolution, so it tends to keep everything or
    nothing.  ``gate="fixed"`` selects the main-text reading instead, in which
    case ``min_overlap`` is an absolute ratio.

    ``tau_rot``/``tau_tra`` are named but not given numerically in the paper;
    15 deg / 30 deg here, matching the RRA@15 operating point of the two-view
    tables.

    The overlap test uses ``edge.overlap`` as recorded by
    :meth:`PairwiseEdge.from_pointmaps`.  Edges built directly from
    correspondences carry no overlap and skip that test -- overlap is only
    meaningful between pointmaps in a common frame, which raw ``points_i`` /
    ``points_j`` (each in its own camera frame) are not.
    """
    if gate not in ("quantile", "fixed"):
        raise ValueError(f"gate must be 'quantile' or 'fixed' (got {gate!r})")

    by_pair: Dict[Tuple[int, int], PairwiseEdge] = {(e.i, e.j): e for e in edges}
    kept: List[PairwiseEdge] = []

    cutoff = min_overlap
    if gate == "quantile" and min_overlap > 0.0:
        measured = [e.overlap for e in edges if e.overlap is not None]
        if len(measured) >= 2:
            cutoff = float(torch.tensor(measured, dtype=torch.float64).quantile(min_overlap))
        else:
            cutoff = 0.0        # a single edge has no distribution to gate against

    for e in edges:
        # (a) symmetric pose consistency, only checkable when both directions exist
        rev = by_pair.get((e.j, e.i))
        if rev is not None and e.R is not None and rev.R is not None:
            rot_gap = torch.rad2deg(
                geodesic_angle(e.R.unsqueeze(0), rev.R.transpose(-1, -2).unsqueeze(0))
            )
            if float(rot_gap) > tau_rot_deg:
                continue
            if e.t is not None and rev.t is not None:
                # t_j->i should point opposite to R_i->j^T applied to t_i->j.
                pred = -(rev.R.transpose(-1, -2) @ rev.t)
                cos = torch.nn.functional.cosine_similarity(
                    e.t.unsqueeze(0), pred.unsqueeze(0), dim=-1
                ).clamp(-1.0, 1.0)
                if float(torch.rad2deg(torch.arccos(cos))) > tau_tra_deg:
                    continue

        # (b) geometric overlap via MNN, as measured when the edge was built
        if min_overlap > 0.0 and e.overlap is not None and e.overlap < cutoff:
            continue
        kept.append(e)

    return kept


# --------------------------------------------------------------------------- #
# Eq. 12 -- global optimization
# --------------------------------------------------------------------------- #

@dataclass
class AlignmentResult:
    poses: torch.Tensor       # (N, 4, 4) world-from-camera
    scales: torch.Tensor      # (N,)
    history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")


def _objective(
    edges: Sequence[PairwiseEdge],
    poses: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    total = poses.new_zeros(())
    weight = poses.new_zeros(())
    for e in edges:
        r_i, d_i = split_ray_and_radius(e.points_i)
        r_j, d_j = split_ray_and_radius(e.points_j)
        p_i = ray_conditioned_points(r_i, d_i, log_scales[e.i])
        p_j = ray_conditioned_points(r_j, d_j, log_scales[e.j])

        w_i = p_i @ poses[e.i, :3, :3].T + poses[e.i, :3, 3]
        w_j = p_j @ poses[e.j, :3, :3].T + poses[e.j, :3, 3]

        sigma = e.conf.to(p_i.dtype)
        total = total + (sigma * ((w_i - w_j) ** 2).sum(-1)).sum()
        weight = weight + sigma.sum()
    return total / weight.clamp(min=1e-8)


def _spanning_tree_init(
    n_views: int, edges: Sequence[PairwiseEdge], dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Seed poses and log-scales by chaining closed-form sim(3) fits.

    For an edge ``(i, j)``, weighted Umeyama on the correspondences yields
    exactly the relative similarity ``p_i = s R p_j + t``.  Walking a BFS tree
    from view 0 and composing those gives a solution that is already near-exact
    on clean data, so the gradient stage only has to reconcile loop closures.
    Starting instead from all-identity leaves an 8x scale spread for Adam to
    discover, which it does slowly and imprecisely.

    Views in a component disconnected from view 0 keep identity/unit scale.
    """
    from cam3r.metrics import umeyama_alignment

    adj: Dict[int, List[Tuple[int, PairwiseEdge, bool]]] = {v: [] for v in range(n_views)}
    for e in edges:
        adj[e.i].append((e.j, e, False))   # traverse i -> j
        adj[e.j].append((e.i, e, True))    # traverse j -> i

    R_w = torch.eye(3, dtype=dtype, device=device).repeat(n_views, 1, 1)
    t_w = torch.zeros(n_views, 3, dtype=dtype, device=device)
    log_s = torch.zeros(n_views, dtype=dtype, device=device)

    seen = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for nxt, e, reverse in adj[cur]:
            if nxt in seen:
                continue
            # Fit the similarity carrying `nxt`'s frame into `cur`'s frame.
            src = e.points_i if reverse else e.points_j
            dst = e.points_j if reverse else e.points_i
            try:
                s_rel, R_rel, t_rel = umeyama_alignment(src, dst, weights=e.conf)
            except Exception:
                continue
            if not (math.isfinite(s_rel) and s_rel > 0):
                continue
            R_rel = R_rel.to(dtype=dtype, device=device)
            t_rel = t_rel.to(dtype=dtype, device=device)

            s_cur = torch.exp(log_s[cur])
            R_w[nxt] = R_w[cur] @ R_rel
            t_w[nxt] = R_w[cur] @ (s_cur * t_rel) + t_w[cur]
            log_s[nxt] = log_s[cur] + math.log(s_rel)
            seen.add(nxt)
            queue.append(nxt)

    return make_se3(R_w, t_w), log_s


def ray_aware_global_alignment(
    n_views: int,
    edges: Sequence[PairwiseEdge],
    iters: int = 500,
    lr: float = 0.01,
    verbose: bool = False,
    dtype: torch.dtype = torch.float64,
    init: str = "spanning_tree",
) -> AlignmentResult:
    """Minimize Eq. 12 over global poses and per-view scales.

    Follows the paper's alternating schedule -- poses, then scales, then a joint
    refinement -- each taking a third of ``iters``.  View 0 is pinned (identity
    pose, unit scale) to fix the global similarity gauge.  ``init`` selects the
    starting point: ``"spanning_tree"`` (closed-form, default) or
    ``"identity"``.
    """
    if n_views < 1:
        raise ValueError("need at least one view")
    if not edges:
        raise ValueError("no edges left to align (scene graph fully pruned?)")
    for e in edges:
        if not (0 <= e.i < n_views and 0 <= e.j < n_views):
            raise ValueError(f"edge ({e.i},{e.j}) out of range for {n_views} views")

    device = edges[0].points_i.device
    edges = [
        PairwiseEdge(e.i, e.j, e.points_i.to(dtype), e.points_j.to(dtype), e.conf.to(dtype),
                     e.R, e.t, e.overlap)
        for e in edges
    ]

    # This routine runs its own optimization, but callers reach it from an
    # inference path that is normally wrapped in torch.no_grad().  Re-enable
    # gradients locally rather than requiring every caller to remember.
    with torch.enable_grad():
        return _run_alignment(n_views, edges, iters, lr, verbose, dtype, init, device)


def _run_alignment(n_views, edges, iters, lr, verbose, dtype, init, device) -> AlignmentResult:
    if init == "spanning_tree":
        T0, log_s0 = _spanning_tree_init(n_views, edges, dtype, device)
    elif init == "identity":
        T0 = torch.eye(4, dtype=dtype, device=device).repeat(n_views, 1, 1)
        log_s0 = torch.zeros(n_views, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown init {init!r}")

    # Free parameters for views 1.. ; view 0 is the fixed gauge.
    rot6d = torch.cat([T0[1:, :3, 0], T0[1:, :3, 1]], dim=-1).clone().requires_grad_()
    trans = T0[1:, :3, 3].clone().requires_grad_()
    log_s = (log_s0[1:] - log_s0[0]).clone().requires_grad_()

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    zero3 = torch.zeros(1, 3, dtype=dtype, device=device)
    zero1 = torch.zeros(1, dtype=dtype, device=device)

    def assemble():
        R = torch.cat([eye, rot6d_to_matrix(rot6d)], dim=0)
        t = torch.cat([zero3, trans], dim=0)
        return make_se3(R, t), torch.cat([zero1, log_s], dim=0)

    phases = [
        ([rot6d, trans], max(1, iters // 3)),      # poses
        ([log_s], max(1, iters // 3)),             # scales
        ([rot6d, trans, log_s], iters - 2 * max(1, iters // 3)),   # joint
    ]

    history: List[float] = []
    # Keep the best iterate rather than the last.  The spanning-tree seed can
    # already be optimal (on clean input its residual is ~1e-29), and a
    # fixed-step optimizer will happily walk away from it; returning the last
    # iterate would then be strictly worse than doing no refinement at all.
    with torch.no_grad():
        poses0, log_scales0 = assemble()
        best_loss = float(_objective(edges, poses0, log_scales0))
    best_state = (rot6d.detach().clone(), trans.detach().clone(), log_s.detach().clone())

    for params, n_steps in phases:
        opt = torch.optim.Adam(params, lr=lr)
        for step in range(n_steps):
            opt.zero_grad()
            poses, log_scales = assemble()
            loss = _objective(edges, poses, log_scales)
            loss.backward()
            opt.step()

            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_state = (rot6d.detach().clone(), trans.detach().clone(), log_s.detach().clone())
            if verbose and step % 100 == 0:
                print(f"  [RAGA] {len(history):5d}  loss {value:.6e}")

    with torch.no_grad():
        rot6d.copy_(best_state[0])
        trans.copy_(best_state[1])
        log_s.copy_(best_state[2])
        poses, log_scales = assemble()
    return AlignmentResult(poses.detach(), torch.exp(log_scales.detach()), history, best_loss)
