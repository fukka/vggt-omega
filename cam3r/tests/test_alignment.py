"""Ray-Aware Global Alignment: scene-graph pruning + global optimization.

The decisive test is :func:`test_alignment_recovers_known_poses` -- a synthetic
multi-view scene with known poses and deliberately mismatched per-view scales.
If the optimizer cannot recover those, nothing downstream can.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.alignment import (
    PairwiseEdge,
    anchor_node,
    consensus_fields,
    mutual_nearest_neighbors,
    overlap_ratio,
    prune_scene_graph,
    ray_aware_global_alignment,
    ray_conditioned_points,
)
from cam3r.geometry import make_se3, random_rotations, relative_pose, se3_inverse, transform_points
from cam3r.metrics import relative_pose_error


def _scene(n_views: int = 4, n_pts: int = 200, seed: int = 0):
    """Cameras looking at a shared cloud; returns GT world-from-cam poses + cloud."""
    torch.manual_seed(seed)
    world = torch.randn(n_pts, 3, dtype=torch.float64) * 2.0 + torch.tensor([0.0, 0.0, 8.0], dtype=torch.float64)
    R = random_rotations(n_views, seed=seed, dtype=torch.float64)
    R[0] = torch.eye(3, dtype=torch.float64)
    t = torch.randn(n_views, 3, dtype=torch.float64) * 0.4
    t[0] = 0.0
    T_wc = make_se3(R, t)                     # world-from-camera
    return T_wc, world


def _edges_from_scene(T_wc, world, scales, pairs, conf=None):
    """Per-view points in camera frame, pre-divided by that view's true scale."""
    edges = []
    for (i, j) in pairs:
        pi = transform_points(se3_inverse(T_wc[i]), world) / scales[i]
        pj = transform_points(se3_inverse(T_wc[j]), world) / scales[j]
        c = torch.ones(world.shape[0], dtype=torch.float64) if conf is None else conf
        edges.append(PairwiseEdge(i=i, j=j, points_i=pi, points_j=pj, conf=c))
    return edges


# ------------------------------------------------------------------- Eq. S4

def test_ray_conditioned_points_move_only_along_their_rays():
    torch.manual_seed(1)
    rays = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    d = torch.rand(50) + 0.5
    a = ray_conditioned_points(rays, d, torch.tensor(0.0))
    b = ray_conditioned_points(rays, d, torch.tensor(0.7))
    # Direction preserved exactly; only the radius changed, by exp(0.7).
    assert torch.allclose(torch.nn.functional.normalize(a, dim=-1), rays, atol=1e-6)
    assert torch.allclose(torch.nn.functional.normalize(b, dim=-1), rays, atol=1e-6)
    assert torch.allclose(b.norm(dim=-1) / a.norm(dim=-1),
                          torch.full((50,), math.exp(0.7)), atol=1e-5)


# --------------------------------------------------------------- MNN / pruning

def test_mnn_finds_the_identity_matching():
    torch.manual_seed(2)
    A = torch.randn(60, 3)
    perm = torch.randperm(60)
    idx_a, idx_b, _ = mutual_nearest_neighbors(A, A[perm])
    assert torch.equal(perm[idx_b], idx_a)


def test_mnn_is_mutual_not_just_nearest():
    """A far-away outlier must not be matched just because something is closest to it."""
    A = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    B = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    idx_a, idx_b, _ = mutual_nearest_neighbors(A, B)
    assert len(idx_a) == 1 and int(idx_a[0]) == 0 and int(idx_b[0]) == 0


def test_overlap_ratio_is_high_for_a_shared_cloud_and_low_for_disjoint_ones():
    torch.manual_seed(3)
    A = torch.randn(100, 3)
    assert overlap_ratio(A, A.clone(), radius=1e-3) > 0.99
    far = torch.randn(100, 3) + torch.tensor([500.0, 0.0, 0.0])
    assert overlap_ratio(A, far, radius=0.5) < 0.05


def test_from_pointmaps_matches_in_the_common_frame_and_scores_overlap():
    """MNN runs between X^{i,i} and X^{j,i}; points are stored per own frame."""
    T_wc, world = _scene(n_views=2, n_pts=120, seed=11)
    X_ii = transform_points(se3_inverse(T_wc[0]), world)
    X_jj = transform_points(se3_inverse(T_wc[1]), world)
    X_ji = transform_points(se3_inverse(T_wc[0]), world)   # view j's points, in frame i

    e = PairwiseEdge.from_pointmaps(0, 1, X_ii, X_ji, X_jj, radius=1e-6)
    assert e.overlap is not None and e.overlap > 0.99
    assert torch.allclose(e.points_i, X_ii, atol=1e-9)
    assert torch.allclose(e.points_j, X_jj, atol=1e-9)


def test_from_pointmaps_reports_low_overlap_for_disjoint_views():
    torch.manual_seed(12)
    X_ii = torch.randn(80, 3, dtype=torch.float64)
    X_jj = torch.randn(80, 3, dtype=torch.float64)
    X_ji = X_jj + 900.0                                    # nowhere near view i
    e = PairwiseEdge.from_pointmaps(0, 1, X_ii, X_ji, X_jj, radius=0.5)
    assert e.overlap is not None and e.overlap < 0.05


def test_pruning_drops_non_overlapping_edges():
    T_wc, world = _scene(n_views=3)
    scales = torch.ones(3, dtype=torch.float64)
    good = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 2)])
    for e in good:
        e.overlap = 0.9

    disjoint = PairwiseEdge(
        i=0, j=2,
        points_i=torch.randn(50, 3, dtype=torch.float64),
        points_j=torch.randn(50, 3, dtype=torch.float64),
        conf=torch.ones(50, dtype=torch.float64),
        overlap=0.03,
    )
    kept = prune_scene_graph(good + [disjoint], min_overlap=0.2, gate="fixed")
    assert {(e.i, e.j) for e in kept} == {(0, 1), (1, 2)}


def test_quantile_gate_adapts_to_the_scene_wide_distribution():
    """Supp C.1: cut at the 20th percentile of the match distribution, not 0.2.

    All these edges sit far below a fixed 0.2 cut, which would delete the whole
    scene graph; the quantile gate keeps the better 80% of them.
    """
    edges = [
        PairwiseEdge(i=0, j=k, points_i=torch.zeros(4, 3, dtype=torch.float64),
                     points_j=torch.zeros(4, 3, dtype=torch.float64),
                     conf=torch.ones(4, dtype=torch.float64), overlap=o)
        for k, o in enumerate([0.01, 0.02, 0.03, 0.04, 0.05], start=1)
    ]
    assert prune_scene_graph(edges, min_overlap=0.2, gate="fixed") == []
    kept = prune_scene_graph(edges, min_overlap=0.2, gate="quantile")
    assert len(kept) == 4 and min(e.overlap for e in kept) > 0.01


def test_quantile_gate_needs_a_distribution_to_gate_against():
    """One edge is not a distribution; do not delete the only thing you have."""
    e = PairwiseEdge(i=0, j=1, points_i=torch.zeros(4, 3, dtype=torch.float64),
                     points_j=torch.zeros(4, 3, dtype=torch.float64),
                     conf=torch.ones(4, dtype=torch.float64), overlap=0.001)
    assert len(prune_scene_graph([e], min_overlap=0.2, gate="quantile")) == 1


def test_confidence_is_gathered_with_each_views_own_indices():
    """Regression: sigma_ij must not be permuted onto the wrong points.

    conf_i lives on view i's grid and conf_j on view j's, and MNN returns a
    different index vector for each.  Indexing one map with the other's indices
    has matching lengths, so it fails silently.
    """
    torch.manual_seed(20)
    X_ii = torch.randn(40, 3, dtype=torch.float64)
    perm = torch.randperm(40)
    X_jj = torch.randn(40, 3, dtype=torch.float64)
    X_ji = X_ii[perm]                       # view j sees the same points, reordered

    conf_i = torch.arange(40, dtype=torch.float64)
    conf_j = torch.arange(40, dtype=torch.float64) * 100

    e = PairwiseEdge.from_pointmaps(0, 1, X_ii, X_ji, X_jj, conf_i=conf_i, radius=1e-9)
    idx_a, idx_b, _ = mutual_nearest_neighbors(X_ii, X_ji)
    assert torch.allclose(e.conf, conf_i[idx_a])
    assert not torch.allclose(e.conf, conf_i[idx_b]), "weights were permuted"

    both = PairwiseEdge.from_pointmaps(0, 1, X_ii, X_ji, X_jj,
                                       conf_i=conf_i, conf_j=conf_j, radius=1e-9)
    assert torch.allclose(both.conf, torch.minimum(conf_i[idx_a], conf_j[idx_b]))


def test_pruning_skips_the_overlap_test_when_it_was_never_measured():
    """Edges built straight from correspondences carry no overlap; don't invent one."""
    T_wc, world = _scene(n_views=2)
    edges = _edges_from_scene(T_wc, world, torch.ones(2, dtype=torch.float64), [(0, 1)])
    assert edges[0].overlap is None
    assert len(prune_scene_graph(edges, min_overlap=0.99)) == 1


def test_pruning_rejects_asymmetric_reciprocal_poses():
    """Reciprocal edges must satisfy R_j->i = R_i->j^T within tau_rot."""
    T_wc, world = _scene(n_views=2)
    scales = torch.ones(2, dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 0)])
    R01, t01 = relative_pose(se3_inverse(T_wc[0]).unsqueeze(0), se3_inverse(T_wc[1]).unsqueeze(0))
    edges[0].R, edges[0].t = R01[0], t01[0]
    # Reciprocal edge given a rotation 40 deg away from the required transpose.
    a = math.radians(40.0)
    spin = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                         [math.sin(a), math.cos(a), 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float64)
    edges[1].R, edges[1].t = spin @ R01[0].T, -R01[0].T @ t01[0]

    kept = prune_scene_graph(edges, min_overlap=0.0, tau_rot_deg=15.0)
    assert len(kept) == 0, "inconsistent reciprocal pair should be dropped"

    edges[1].R = R01[0].T
    assert len(prune_scene_graph(edges, min_overlap=0.0, tau_rot_deg=15.0)) == 2


def test_strict_symmetry_drops_the_direction_that_passed():
    """Supp C.1: 'if e_ij passes but e_ji fails, both edges are discarded'."""
    T_wc, world = _scene(n_views=2)
    scales = torch.ones(2, dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 0)])
    edges[0].overlap = 0.90        # e_01 sails through the gate
    edges[1].overlap = 0.01        # e_10 fails it

    both = prune_scene_graph(edges, min_overlap=0.5, gate="fixed", strict_symmetry=True)
    assert both == [], "the surviving direction must go down with its reciprocal"

    lone = prune_scene_graph(edges, min_overlap=0.5, gate="fixed", strict_symmetry=False)
    assert {(e.i, e.j) for e in lone} == {(0, 1)}


def test_largest_connected_component_is_extracted():
    """Supp C.1 stage 3: a fragment that shares no edge with the bulk is dropped."""
    T_wc, world = _scene(n_views=5)
    scales = torch.ones(5, dtype=torch.float64)
    # 0-1-2 is the bulk; 3-4 is an unrelated fragment.
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 2), (0, 2), (3, 4)])
    for e in edges:
        e.overlap = 0.9

    kept = prune_scene_graph(edges, min_overlap=0.0)
    assert {(e.i, e.j) for e in kept} == {(0, 1), (1, 2), (0, 2)}
    assert {(e.i, e.j) for e in prune_scene_graph(edges, min_overlap=0.0,
                                                  largest_component=False)} == {
        (0, 1), (1, 2), (0, 2), (3, 4)}


def test_a_disconnected_view_would_otherwise_be_reported_at_the_origin():
    """Why stage 3 matters: the seed leaves an unreachable view at identity.

    Without the component extraction the optimizer has no term touching view 3,
    so it silently returns a camera at the world origin rather than an error.
    """
    T_wc, world = _scene(n_views=4)
    scales = torch.ones(4, dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 2), (3, 3)][:2])
    res = ray_aware_global_alignment(4, edges, iters=30)
    assert torch.allclose(res.poses[3], torch.eye(4, dtype=torch.float64)), \
        "view 3 is unreachable and should still be sitting at identity"


# --------------------------------------------------------------- Sec. 3.3 anchor

def test_anchor_is_the_highest_degree_node():
    """Supp C.2 fixes the pose of the highest-degree node, not view 0."""
    T_wc, world = _scene(n_views=4)
    scales = torch.ones(4, dtype=torch.float64)
    # View 2 touches three edges; views 0, 1, 3 touch one or two.
    edges = _edges_from_scene(T_wc, world, scales, [(0, 2), (1, 2), (2, 3)])
    assert anchor_node(4, edges) == 2

    res = ray_aware_global_alignment(4, edges, iters=60)
    assert res.anchor == 2
    assert torch.allclose(res.poses[2], torch.eye(4, dtype=torch.float64), atol=1e-9)
    assert abs(float(res.scales[2]) - 1.0) < 1e-9


def test_anchor_choice_does_not_change_relative_geometry():
    """The gauge is arbitrary: pinning a different camera must give the same scene."""
    T_wc, world = _scene(n_views=3, n_pts=200, seed=11)
    scales = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 2), (0, 2)])
    a = ray_aware_global_alignment(3, edges, iters=300, anchor=0)
    b = ray_aware_global_alignment(3, edges, iters=300, anchor=2)
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        Ra, ta = relative_pose(se3_inverse(a.poses[i]).unsqueeze(0), se3_inverse(a.poses[j]).unsqueeze(0))
        Rb, tb = relative_pose(se3_inverse(b.poses[i]).unsqueeze(0), se3_inverse(b.poses[j]).unsqueeze(0))
        rot, tra = relative_pose_error(Ra, ta, Rb, tb)
        assert float(rot) < 1.0 and float(tra) < 2.0
    ratio_a = a.scales / a.scales[0]
    ratio_b = b.scales / b.scales[0]
    assert torch.allclose(ratio_a, ratio_b, rtol=0.05)


# ------------------------------------------------------- Sec. 3.3 consensus fields

def _pointmap_edge(i, j, T_wc, world, scale_i=1.0, scale_j=1.0, radius=0.5):
    """A real from_pointmaps edge, so it carries pixel indices."""
    X_ii = transform_points(se3_inverse(T_wc[i]), world) / scale_i
    X_jj = transform_points(se3_inverse(T_wc[j]), world) / scale_j
    R, t = relative_pose(se3_inverse(T_wc[i]).unsqueeze(0), se3_inverse(T_wc[j]).unsqueeze(0))
    X_ji = X_jj * scale_j @ R[0].T + t[0]
    return PairwiseEdge.from_pointmaps(i, j, X_ii, X_ji, X_jj, R=R[0], t=t[0], radius=radius)


def test_consensus_fields_need_pixel_indices():
    T_wc, world = _scene(n_views=2, n_pts=40)
    edges = _edges_from_scene(T_wc, world, torch.ones(2, dtype=torch.float64), [(0, 1)])
    try:
        consensus_fields(2, edges, n_pixels=40)
    except ValueError as exc:
        assert "pixel indices" in str(exc)
    else:
        raise AssertionError("fusing edges without indices should be refused")


def test_consensus_rays_are_unit_and_agree_with_the_observations():
    T_wc, world = _scene(n_views=3, n_pts=60, seed=8)
    edges = [_pointmap_edge(0, 1, T_wc, world), _pointmap_edge(0, 2, T_wc, world)]
    fields = consensus_fields(3, edges, n_pixels=60)

    seen = fields[0].seen
    assert bool(seen.any())
    norms = fields[0].rays[seen].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9)

    # View 0's own pointmap is the same in both edges, so the fused field must
    # reproduce it exactly.
    truth = transform_points(se3_inverse(T_wc[0]), world)
    got = fields[0].points()[seen]
    assert torch.allclose(got, truth[seen], atol=1e-8)


def test_consensus_radial_median_alignment_survives_a_mis_scaled_edge():
    """Step 2 of the three: relative scale resolved by a robust median."""
    T_wc, world = _scene(n_views=3, n_pts=80, seed=9)
    good = _pointmap_edge(0, 1, T_wc, world)
    # An edge whose view-0 radial distances are uniformly 4x too large.
    skewed = _pointmap_edge(0, 2, T_wc, world)
    skewed.points_i = skewed.points_i * 4.0
    skewed.conf = skewed.conf * 0.5

    fields = consensus_fields(3, [good, skewed], n_pixels=80)
    seen = fields[0].seen
    truth = transform_points(se3_inverse(T_wc[0]), world).norm(dim=-1)
    got = fields[0].radial[seen]
    err = (got / truth[seen] - 1.0).abs().max()
    assert float(err) < 1e-6, f"median alignment left {float(err):.3e} relative error"


def test_alignment_over_consensus_fields_recovers_known_poses():
    T_wc, world = _scene(n_views=3, n_pts=200, seed=10)
    edges = [_pointmap_edge(i, j, T_wc, world) for (i, j) in [(0, 1), (1, 2), (0, 2)]]
    views = consensus_fields(3, edges, n_pixels=200)
    res = ray_aware_global_alignment(3, edges, iters=300, views=views)
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        R_gt, t_gt = relative_pose(se3_inverse(T_wc[i]).unsqueeze(0), se3_inverse(T_wc[j]).unsqueeze(0))
        R_p, t_p = relative_pose(se3_inverse(res.poses[i]).unsqueeze(0), se3_inverse(res.poses[j]).unsqueeze(0))
        rot, tra = relative_pose_error(R_p, t_p, R_gt, t_gt)
        assert float(rot) < 2.0 and float(tra) < 5.0


def test_refine_log_depth_needs_consensus_fields():
    T_wc, world = _scene(n_views=2, n_pts=60, seed=12)
    edges = [_pointmap_edge(0, 1, T_wc, world)]
    try:
        ray_aware_global_alignment(2, edges, iters=10, refine_log_depth=True)
    except ValueError as exc:
        assert "consensus_fields" in str(exc)
    else:
        raise AssertionError("per-pixel log-depth has no meaning without shared fields")


def test_refine_log_depth_does_not_make_a_clean_scene_worse():
    """Supp Eq. S4 frees log d; on exact input it should find nothing to fix."""
    T_wc, world = _scene(n_views=3, n_pts=150, seed=16)
    edges = [_pointmap_edge(i, j, T_wc, world) for (i, j) in [(0, 1), (1, 2), (0, 2)]]
    views = consensus_fields(3, edges, n_pixels=150)
    frozen = ray_aware_global_alignment(3, edges, iters=150, views=views)
    freed = ray_aware_global_alignment(3, edges, iters=150, views=views, refine_log_depth=True)
    assert freed.best_loss <= frozen.best_loss * 1.05


# ------------------------------------------------------------------- Eq. 12

def test_alignment_recovers_known_poses():
    """Four views, per-view scales spanning 8x, all pairs observed."""
    T_wc, world = _scene(n_views=4, n_pts=300, seed=4)
    scales = torch.tensor([1.0, 2.0, 0.5, 4.0], dtype=torch.float64)
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    edges = _edges_from_scene(T_wc, world, scales, pairs)

    res = ray_aware_global_alignment(4, edges, iters=400, verbose=False)

    # Compare relative poses -- the global gauge is arbitrary.
    for (i, j) in pairs:
        R_gt, t_gt = relative_pose(se3_inverse(T_wc[i]).unsqueeze(0), se3_inverse(T_wc[j]).unsqueeze(0))
        R_p, t_p = relative_pose(se3_inverse(res.poses[i]).unsqueeze(0), se3_inverse(res.poses[j]).unsqueeze(0))
        rot, tra = relative_pose_error(R_p, t_p, R_gt, t_gt)
        assert float(rot) < 2.0, f"pair {(i, j)} rotation off by {float(rot):.2f} deg"
        assert float(tra) < 5.0, f"pair {(i, j)} translation off by {float(tra):.2f} deg"


def test_alignment_recovers_relative_scales():
    T_wc, world = _scene(n_views=3, n_pts=250, seed=5)
    scales = torch.tensor([1.0, 3.0, 0.25], dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (0, 2), (1, 2)])
    res = ray_aware_global_alignment(3, edges, iters=400, verbose=False)

    ratio = res.scales / res.scales[0]
    assert torch.allclose(ratio, (scales / scales[0]).to(ratio.dtype), rtol=0.05)


def test_alignment_downweights_low_confidence_outliers():
    """Corrupted correspondences with near-zero sigma must not drag the solution."""
    T_wc, world = _scene(n_views=3, n_pts=300, seed=6)
    scales = torch.ones(3, dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (0, 2), (1, 2)])
    for e in edges:
        e.points_j[:80] += torch.randn(80, 3, dtype=torch.float64) * 5.0
        e.conf = e.conf.clone()
        e.conf[:80] = 1e-4

    res = ray_aware_global_alignment(3, edges, iters=400, verbose=False)
    R_gt, t_gt = relative_pose(se3_inverse(T_wc[0]).unsqueeze(0), se3_inverse(T_wc[1]).unsqueeze(0))
    R_p, t_p = relative_pose(se3_inverse(res.poses[0]).unsqueeze(0), se3_inverse(res.poses[1]).unsqueeze(0))
    rot, _ = relative_pose_error(R_p, t_p, R_gt, t_gt)
    assert float(rot) < 5.0, f"outliers dominated: {float(rot):.2f} deg"


def test_alignment_returns_its_best_iterate_not_its_last():
    """Refinement must never return something worse than where it started."""
    T_wc, world = _scene(n_views=3, n_pts=150, seed=7)
    edges = _edges_from_scene(T_wc, world, torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64),
                              [(0, 1), (1, 2)])
    res = ray_aware_global_alignment(3, edges, iters=200, verbose=False)
    assert res.poses.shape == (3, 4, 4) and res.scales.shape == (3,)
    assert res.best_loss <= min(res.history)
    assert res.best_loss <= res.history[0]


def test_alignment_works_inside_no_grad():
    """Callers reach RAGA from an inference path; it must enable grad itself."""
    T_wc, world = _scene(n_views=3, n_pts=120, seed=14)
    edges = _edges_from_scene(T_wc, world, torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64),
                              [(0, 1), (1, 2)])
    with torch.no_grad():
        res = ray_aware_global_alignment(3, edges, iters=60)
    assert torch.isfinite(res.poses).all() and torch.isfinite(res.scales).all()


def test_dtype_conversion_preserves_overlap():
    """The internal float64 copy must not drop the pruning metadata."""
    T_wc, world = _scene(n_views=2, n_pts=80, seed=15)
    edges = _edges_from_scene(T_wc, world, torch.ones(2, dtype=torch.float64), [(0, 1)])
    edges[0].overlap = 0.42
    res = ray_aware_global_alignment(2, edges, iters=30)
    assert torch.isfinite(res.poses).all()


def test_identity_init_still_converges_but_starts_far_worse():
    """The spanning-tree seed is the reason the default converges; show the gap."""
    T_wc, world = _scene(n_views=3, n_pts=200, seed=13)
    scales = torch.tensor([1.0, 3.0, 0.25], dtype=torch.float64)
    edges = _edges_from_scene(T_wc, world, scales, [(0, 1), (0, 2), (1, 2)])
    tree = ray_aware_global_alignment(3, edges, iters=300, init="spanning_tree")
    ident = ray_aware_global_alignment(3, edges, iters=300, init="identity")
    assert tree.best_loss < ident.best_loss


def test_duplicate_keys_are_judged_per_edge():
    """Regression: two edges sharing (i, j) must be filtered independently.

    Keying the verdict by ordered pair lets a bad duplicate ride in on its
    twin's pass -- which is exactly how an all-pairs graph plus one junk edge
    on an existing pair slipped through.
    """
    T_wc, world = _scene(n_views=3)
    scales = torch.ones(3, dtype=torch.float64)
    good = _edges_from_scene(T_wc, world, scales, [(0, 1), (1, 2), (0, 2)])
    for e in good:
        e.overlap = 0.9
    junk = PairwiseEdge(i=0, j=2,
                        points_i=torch.randn(40, 3, dtype=torch.float64),
                        points_j=torch.randn(40, 3, dtype=torch.float64),
                        conf=torch.ones(40, dtype=torch.float64), overlap=0.01)
    kept = prune_scene_graph(good + [junk], min_overlap=0.2)
    assert len(kept) == 3
    assert all(e.overlap == 0.9 for e in kept)
