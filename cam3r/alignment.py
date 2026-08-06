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
    rays: torch.Tensor,
    radial: torch.Tensor,
    log_scale: torch.Tensor,
    log_depth_residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``exp(log d + log s) * r`` -- scale a point cloud strictly along its rays.

    ``rays`` (N, 3) unit directions, ``radial`` (N,) distances, ``log_scale`` a
    scalar.  Working in log space keeps the scale positive without a clamp.

    ``log_depth_residual`` (N,) is the free per-pixel ``log d`` of supp. Eq. S4:
    Sec. C.3 lists log-depths among the optimisation variables, while the main
    paper's Eq. 12 freezes them (see ``PAPER.md`` erratum 5).  Passing ``None``
    is the Eq. 12 reading and is the default.
    """
    log_d = torch.log(radial.clamp(min=1e-8))
    if log_depth_residual is not None:
        log_d = log_d + log_depth_residual
    return torch.exp(log_d + log_scale).unsqueeze(-1) * rays


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

    ``idx_i``/``idx_j`` are the *pixel* indices the correspondences were drawn
    from, in each view's own flattened grid.  They are what lets several edges
    incident on the same image be fused into one per-image field by
    :func:`consensus_fields`; without them an edge is a bag of points with no
    way to tell that its 400th correspondence and another edge's 12th are the
    same pixel.

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
    idx_i: Optional[torch.Tensor] = None
    idx_j: Optional[torch.Tensor] = None

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
        return cls(i, j, X_ii[idx_a], X_jj[idx_b], c, R=R, t=t, overlap=overlap,
                   idx_i=idx_a, idx_j=idx_b)


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


def _largest_connected_component(edges: Sequence[PairwiseEdge]) -> List[PairwiseEdge]:
    """Supp. C.1 stage 3: keep only the edges of the largest component.

    "After pruning, the scene graph may fragment. We therefore retain only the
    largest component, ensuring that global optimization is performed on a
    single consistent camera graph."  Without this, a fragment that shares no
    edge with the anchor never gets a pose: the spanning-tree seed leaves it at
    identity and the objective has no term touching it, so it silently reports
    a camera at the origin instead of an error.  Ties go to the component
    holding the lowest view id, so the result is deterministic.
    """
    if not edges:
        return []
    parent: Dict[int, int] = {}

    def find(v: int) -> int:
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e in edges:
        a, b = find(e.i), find(e.j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    members: Dict[int, List[PairwiseEdge]] = {}
    for e in edges:
        members.setdefault(find(e.i), []).append(e)
    # Rank by number of *views* spanned (the paper's "largest component"), then
    # by edge count, then by lowest root for determinism.
    def size(root: int) -> Tuple[int, int, int]:
        group = members[root]
        return (len({v for e in group for v in (e.i, e.j)}), len(group), -root)

    return members[max(members, key=size)]


def prune_scene_graph(
    edges: Sequence[PairwiseEdge],
    min_overlap: float = 0.2,
    overlap_radius: float = 0.1,
    tau_rot_deg: float = 15.0,
    tau_tra_deg: float = 30.0,
    gate: str = "quantile",
    strict_symmetry: bool = True,
    largest_component: bool = True,
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

    ``strict_symmetry`` is supp. C.1's *"if e_ij passes but e_ji fails, both
    edges are discarded"*, which stops an asymmetric outlier from entering the
    graph in the one direction that happened to look plausible.  It only has
    anything to act on when both directions were actually built; see
    ``eval_adt._multi_view``, which runs a forward pass per direction.

    ``largest_component`` is supp. C.1's third stage.
    """
    if gate not in ("quantile", "fixed"):
        raise ValueError(f"gate must be 'quantile' or 'fixed' (got {gate!r})")

    by_pair: Dict[Tuple[int, int], PairwiseEdge] = {(e.i, e.j): e for e in edges}

    cutoff = min_overlap
    if gate == "quantile" and min_overlap > 0.0:
        measured = [e.overlap for e in edges if e.overlap is not None]
        if len(measured) >= 2:
            cutoff = float(torch.tensor(measured, dtype=torch.float64).quantile(min_overlap))
        else:
            cutoff = 0.0        # a single edge has no distribution to gate against

    def survives(e: PairwiseEdge) -> bool:
        # (1) symmetric pose consistency, only checkable when both directions exist
        rev = by_pair.get((e.j, e.i))
        if rev is not None and e.R is not None and rev.R is not None:
            rot_gap = torch.rad2deg(
                geodesic_angle(e.R.unsqueeze(0), rev.R.transpose(-1, -2).unsqueeze(0))
            )
            if float(rot_gap) > tau_rot_deg:
                return False
            if e.t is not None and rev.t is not None:
                # t_j->i should point opposite to R_i->j^T applied to t_i->j.
                pred = -(rev.R.transpose(-1, -2) @ rev.t)
                cos = torch.nn.functional.cosine_similarity(
                    e.t.unsqueeze(0), pred.unsqueeze(0), dim=-1
                ).clamp(-1.0, 1.0)
                if float(torch.rad2deg(torch.arccos(cos))) > tau_tra_deg:
                    return False

        # (2) geometric overlap via MNN, as measured when the edge was built
        if min_overlap > 0.0 and e.overlap is not None and e.overlap < cutoff:
            return False
        return True

    # Per *edge*, not per (i, j) key: two edges may share a key, and one of them
    # failing must not condemn the other by association.
    survived = [survives(e) for e in edges]

    if strict_symmetry:
        # An ordered pair passes only if every edge carrying it passed.
        ok_key: Dict[Tuple[int, int], bool] = {}
        for e, s in zip(edges, survived):
            ok_key[(e.i, e.j)] = ok_key.get((e.i, e.j), True) and s
        # Drop e_ij whenever its reciprocal exists and failed.
        survived = [
            s and ok_key.get((e.j, e.i), True)
            for e, s in zip(edges, survived)
        ]

    kept = [e for e, s in zip(edges, survived) if s]
    return _largest_connected_component(kept) if largest_component else kept


# --------------------------------------------------------------------------- #
# Sec. 3.3 -- global consensus fields
# --------------------------------------------------------------------------- #

@dataclass
class ViewField:
    """One image's fused per-pixel prior ``x_i(u) = R_i(u) D_i(u)``.

    ``rays`` is the consensus ray field ``D_i``, ``radial`` the global radial
    field ``R_i``, both indexed by flattened pixel.  ``seen`` marks the pixels
    any incident edge actually observed -- the rest are filler and carry zero
    confidence.
    """

    rays: torch.Tensor        # (P, 3) unit
    radial: torch.Tensor      # (P,)
    conf: torch.Tensor        # (P,)
    seen: torch.Tensor        # (P,) bool

    def points(self) -> torch.Tensor:
        return self.rays * self.radial.unsqueeze(-1)


def consensus_fields(
    n_views: int, edges: Sequence[PairwiseEdge], n_pixels: int, eps: float = 1e-8
) -> List[ViewField]:
    """Sec. 3.3 "Global Consensus and Optimization": fuse edges into per-image fields.

    The Ray Module runs once per *edge*, so an image incident on ``k`` surviving
    edges carries ``k`` predictions of the same pixel.  The paper collapses them
    before global alignment:

    1. **Consensus ray** ``D_i`` -- "a confidence-weighted average of all
       normalized rays predicted across its valid incident edges", renormalised
       back onto ``S^2``.
    2. **Global radial field** ``R_i``, in three steps -- align the pairwise
       radial distances *along* the consensus rays (project each observation
       onto ``D_i``), resolve relative scale by "robust median-based
       alignment" against the most confident observation, then fuse by
       confidence-weighted averaging.

    Requires ``idx_i``/``idx_j`` on the edges, i.e. edges built by
    :meth:`PairwiseEdge.from_pointmaps`.  Raises otherwise, because silently
    treating every edge's points as different pixels would defeat the point.

    Note what this step *cannot* do here: CAM3R's Ray Module is per-image, so
    every incident edge feeds it the identical image and it returns the
    identical ray field.  Step 1 therefore averages ``k`` copies of one vector
    and is a no-op on ray directions -- see ``PAPER.md`` erratum 15.  Step 2 is
    not a no-op: radial distance comes from the Cross-view Module, which does
    see the partner image, so the ``k`` radial predictions genuinely differ.
    """
    if n_pixels <= 0:
        raise ValueError(f"n_pixels must be positive (got {n_pixels})")
    for e in edges:
        if e.idx_i is None or e.idx_j is None:
            raise ValueError(
                f"edge ({e.i},{e.j}) has no pixel indices; build edges with "
                "PairwiseEdge.from_pointmaps so observations can be fused per pixel"
            )

    dtype = edges[0].points_i.dtype if edges else torch.float64
    device = edges[0].points_i.device if edges else torch.device("cpu")

    # Group observations by view: (pixel index, points, confidence).
    obs: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        v: [] for v in range(n_views)
    }
    for e in edges:
        obs[e.i].append((e.idx_i, e.points_i.to(dtype), e.conf.to(dtype)))
        obs[e.j].append((e.idx_j, e.points_j.to(dtype), e.conf.to(dtype)))

    fields: List[ViewField] = []
    for v in range(n_views):
        rays = torch.zeros(n_pixels, 3, dtype=dtype, device=device)
        rays[:, 2] = 1.0
        radial = torch.ones(n_pixels, dtype=dtype, device=device)
        conf = torch.zeros(n_pixels, dtype=dtype, device=device)
        seen = torch.zeros(n_pixels, dtype=torch.bool, device=device)
        if not obs[v]:
            fields.append(ViewField(rays, radial, conf, seen))
            continue

        # (1) confidence-weighted average of normalized rays -> D_i
        acc = torch.zeros(n_pixels, 3, dtype=dtype, device=device)
        wsum = torch.zeros(n_pixels, dtype=dtype, device=device)
        for idx, pts, c in obs[v]:
            unit, _ = split_ray_and_radius(pts)
            acc.index_add_(0, idx, unit * c.unsqueeze(-1))
            wsum.index_add_(0, idx, c)
            seen[idx] = True
        hit = wsum > eps
        rays[hit] = torch.nn.functional.normalize(acc[hit], dim=-1, eps=eps)
        conf = wsum

        # (2a) align each observation's radial distance along the consensus ray
        projected: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx, pts, c in obs[v]:
            r = (pts * rays[idx]).sum(-1).clamp(min=eps)
            projected.append((idx, r, c))

        # (2b) robust median-based relative-scale alignment against the most
        # confident observation.  A ratio of medians over the pixels the two
        # share is insensitive to the outliers a single bad edge contributes.
        ref = max(range(len(projected)), key=lambda k: float(projected[k][2].sum()))
        ref_r = torch.zeros(n_pixels, dtype=dtype, device=device)
        ref_has = torch.zeros(n_pixels, dtype=torch.bool, device=device)
        ref_r[projected[ref][0]] = projected[ref][1]
        ref_has[projected[ref][0]] = True

        # (2c) fuse by confidence-weighted averaging
        num = torch.zeros(n_pixels, dtype=dtype, device=device)
        den = torch.zeros(n_pixels, dtype=dtype, device=device)
        for k, (idx, r, c) in enumerate(projected):
            factor = 1.0
            if k != ref:
                shared = ref_has[idx]
                if bool(shared.any()):
                    factor = float(torch.median(ref_r[idx][shared] / r[shared].clamp(min=eps)))
                    if not (math.isfinite(factor) and factor > 0):
                        factor = 1.0
            num.index_add_(0, idx, c * r * factor)
            den.index_add_(0, idx, c)
        ok = den > eps
        radial[ok] = (num[ok] / den[ok]).clamp(min=eps)
        fields.append(ViewField(rays, radial, conf, seen))

    return fields


# --------------------------------------------------------------------------- #
# Eq. 12 -- global optimization
# --------------------------------------------------------------------------- #

@dataclass
class AlignmentResult:
    poses: torch.Tensor       # (N, 4, 4) world-from-camera
    scales: torch.Tensor      # (N,)
    history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")
    anchor: int = 0


def anchor_node(n_views: int, edges: Sequence[PairwiseEdge]) -> int:
    """Supp. C.2: the node with the highest degree in the pruned graph.

    Its pose is fixed to ``I`` as the global reference.  Ties break toward the
    lowest view id so a run is reproducible.  With no edges at all, view 0.
    """
    degree = [0] * max(n_views, 1)
    for e in edges:
        if 0 <= e.i < n_views:
            degree[e.i] += 1
        if 0 <= e.j < n_views:
            degree[e.j] += 1
    return int(max(range(max(n_views, 1)), key=lambda v: (degree[v], -v)))


def _objective(
    edges: Sequence[PairwiseEdge],
    poses: torch.Tensor,
    log_scales: torch.Tensor,
    views: Optional[Sequence[ViewField]] = None,
    log_depth: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    """Eq. 12, optionally over the per-image consensus fields of Sec. 3.3."""
    total = poses.new_zeros(())
    weight = poses.new_zeros(())
    for e in edges:
        if views is None:
            r_i, d_i = split_ray_and_radius(e.points_i)
            r_j, d_j = split_ray_and_radius(e.points_j)
            res_i = res_j = None
            sigma = e.conf.to(poses.dtype)
        else:
            fi, fj = views[e.i], views[e.j]
            r_i, d_i = fi.rays[e.idx_i], fi.radial[e.idx_i]
            r_j, d_j = fj.rays[e.idx_j], fj.radial[e.idx_j]
            res_i = None if log_depth is None else log_depth[e.i][e.idx_i]
            res_j = None if log_depth is None else log_depth[e.j][e.idx_j]
            sigma = torch.minimum(fi.conf[e.idx_i], fj.conf[e.idx_j]).to(poses.dtype)

        p_i = ray_conditioned_points(r_i, d_i, log_scales[e.i], res_i)
        p_j = ray_conditioned_points(r_j, d_j, log_scales[e.j], res_j)

        w_i = p_i @ poses[e.i, :3, :3].T + poses[e.i, :3, 3]
        w_j = p_j @ poses[e.j, :3, :3].T + poses[e.j, :3, 3]

        total = total + (sigma * ((w_i - w_j) ** 2).sum(-1)).sum()
        weight = weight + sigma.sum()
    return total / weight.clamp(min=1e-8)


def _spanning_tree_init(
    n_views: int, edges: Sequence[PairwiseEdge], dtype: torch.dtype, device: torch.device,
    root: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Seed poses and log-scales by chaining closed-form sim(3) fits.

    For an edge ``(i, j)``, weighted Umeyama on the correspondences yields
    exactly the relative similarity ``p_i = s R p_j + t``.  Walking a BFS tree
    from view 0 and composing those gives a solution that is already near-exact
    on clean data, so the gradient stage only has to reconcile loop closures.
    Starting instead from all-identity leaves an 8x scale spread for Adam to
    discover, which it does slowly and imprecisely.

    Views in a component disconnected from ``root`` keep identity/unit scale --
    which is why :func:`prune_scene_graph` extracts the largest connected
    component first (supp. C.1 stage 3): otherwise those views are reported as
    cameras sitting at the origin rather than as missing.
    """
    from cam3r.metrics import umeyama_alignment

    adj: Dict[int, List[Tuple[int, PairwiseEdge, bool]]] = {v: [] for v in range(n_views)}
    for e in edges:
        adj[e.i].append((e.j, e, False))   # traverse i -> j
        adj[e.j].append((e.i, e, True))    # traverse j -> i

    R_w = torch.eye(3, dtype=dtype, device=device).repeat(n_views, 1, 1)
    t_w = torch.zeros(n_views, 3, dtype=dtype, device=device)
    log_s = torch.zeros(n_views, dtype=dtype, device=device)

    seen = {root}
    queue = [root]
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
    views: Optional[Sequence[ViewField]] = None,
    refine_log_depth: bool = False,
    anchor: Optional[int] = None,
) -> AlignmentResult:
    """Minimize Eq. 12 over global poses and per-view scales.

    Follows the paper's alternating schedule -- poses, then scales, then a joint
    refinement -- each taking a third of ``iters``.  ``init`` selects the
    starting point: ``"spanning_tree"`` (closed-form, default) or
    ``"identity"``.

    The gauge is fixed by pinning one camera to identity with unit scale.  Supp.
    C.2 picks *"the node with the highest degree in the pruned graph"*; that is
    the default, and ``anchor`` overrides it.  The choice does not change
    Umeyama-aligned ATE, but it does change which camera the returned poses are
    expressed relative to.

    ``views`` supplies the per-image consensus fields of Sec. 3.3 (see
    :func:`consensus_fields`); the edges then index into them instead of
    carrying their own copies of the geometry, which is what makes the
    per-pixel ``log d`` of supp. Eq. S4 a well-defined shared variable.
    ``refine_log_depth`` turns that variable on -- Sec. C.3's reading of what is
    optimised, against Eq. 12's; see ``PAPER.md`` erratum 5.
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
                     e.R, e.t, e.overlap, e.idx_i, e.idx_j)
        for e in edges
    ]
    if anchor is None:
        anchor = anchor_node(n_views, edges)
    if not 0 <= anchor < n_views:
        raise ValueError(f"anchor {anchor} out of range for {n_views} views")
    if refine_log_depth and views is None:
        raise ValueError("refine_log_depth needs `views` (see consensus_fields)")
    if views is not None:
        views = [
            ViewField(v.rays.to(dtype), v.radial.to(dtype), v.conf.to(dtype), v.seen)
            for v in views
        ]
        if any(e.idx_i is None or e.idx_j is None for e in edges):
            raise ValueError("`views` requires edges carrying pixel indices")

    # This routine runs its own optimization, but callers reach it from an
    # inference path that is normally wrapped in torch.no_grad().  Re-enable
    # gradients locally rather than requiring every caller to remember.
    with torch.enable_grad():
        return _run_alignment(n_views, edges, iters, lr, verbose, dtype, init, device,
                              views, refine_log_depth, anchor)


def _run_alignment(n_views, edges, iters, lr, verbose, dtype, init, device,
                   views, refine_log_depth, anchor) -> AlignmentResult:
    if init == "spanning_tree":
        T0, log_s0 = _spanning_tree_init(n_views, edges, dtype, device, root=anchor)
    elif init == "identity":
        T0 = torch.eye(4, dtype=dtype, device=device).repeat(n_views, 1, 1)
        log_s0 = torch.zeros(n_views, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown init {init!r}")

    # Free parameters for every view but the anchor, which is the fixed gauge.
    keep = [v for v in range(n_views) if v != anchor]
    T_free, s_free = T0[keep], log_s0[keep] - log_s0[anchor]
    rot6d = torch.cat([T_free[:, :3, 0], T_free[:, :3, 1]], dim=-1).clone().requires_grad_()
    trans = T_free[:, :3, 3].clone().requires_grad_()
    log_s = s_free.clone().requires_grad_()

    log_depth: Optional[List[torch.Tensor]] = None
    if refine_log_depth:
        log_depth = [
            torch.zeros_like(v.radial).requires_grad_() for v in views      # type: ignore[union-attr]
        ]

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    zero3 = torch.zeros(1, 3, dtype=dtype, device=device)
    zero1 = torch.zeros(1, dtype=dtype, device=device)

    def assemble():
        # `keep` is ascending, so splicing the anchor's fixed row back in at its
        # own index restores the original view order.
        Rf = rot6d_to_matrix(rot6d)
        R = torch.cat([Rf[:anchor], eye, Rf[anchor:]], dim=0)
        t = torch.cat([trans[:anchor], zero3, trans[anchor:]], dim=0)
        s = torch.cat([log_s[:anchor], zero1, log_s[anchor:]], dim=0)
        return make_se3(R, t), s

    def evaluate():
        poses, log_scales = assemble()
        return _objective(edges, poses, log_scales, views, log_depth)

    depth_params = list(log_depth) if log_depth is not None else []
    phases = [
        ([rot6d, trans], max(1, iters // 3)),                   # poses
        ([log_s] + depth_params, max(1, iters // 3)),           # scales (and depths)
        ([rot6d, trans, log_s] + depth_params,
         iters - 2 * max(1, iters // 3)),                       # joint
    ]

    history: List[float] = []
    # Keep the best iterate rather than the last.  The spanning-tree seed can
    # already be optimal (on clean input its residual is ~1e-29), and a
    # fixed-step optimizer will happily walk away from it; returning the last
    # iterate would then be strictly worse than doing no refinement at all.
    tracked = [rot6d, trans, log_s] + depth_params

    def snapshot():
        return [p.detach().clone() for p in tracked]

    with torch.no_grad():
        best_loss = float(evaluate())
    best_state = snapshot()

    for params, n_steps in phases:
        opt = torch.optim.Adam(params, lr=lr)
        for step in range(n_steps):
            opt.zero_grad()
            loss = evaluate()
            loss.backward()
            opt.step()

            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_state = snapshot()
            if verbose and step % 100 == 0:
                print(f"  [RAGA] {len(history):5d}  loss {value:.6e}")

    with torch.no_grad():
        for param, saved in zip(tracked, best_state):
            param.copy_(saved)
        poses, log_scales = assemble()
    return AlignmentResult(poses.detach(), torch.exp(log_scales.detach()), history,
                           best_loss, anchor)
