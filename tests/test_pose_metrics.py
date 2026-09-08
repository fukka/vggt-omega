# Copyright (c) 2026.
"""Pin the window pose metrics of exp_rendered against poses whose answer is known.

A pose metric that silently measures the wrong thing looks exactly like a real
result, so each invariance the docstring claims is checked on synthetic cameras:
a global Sim(3) (VGGT's free frame and scale) must cost nothing, a known
relative rotation must read as its own angle, and a collapsed prediction must
not score as a perfect one.
"""
import numpy as np
import pytest

from finetune.eval.exp_rendered import (A_ROT, _rot_angle_deg, _vec_angle_deg,
                                        window_pose_metrics)


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _trajectory(n=8, seed=0):
    """Cam-from-world (n,3,4) along a gently curving path with real baseline."""
    rng = np.random.default_rng(seed)
    E = []
    for i in range(n):
        R = _rot([0, 1, 0], 6.0 * i) @ _rot([1, 0, 0], 2.0 * np.sin(i))
        c = np.array([0.3 * i, 0.05 * i * i, 0.1 * np.cos(i)]) + 0.01 * rng.standard_normal(3)
        E.append(np.concatenate([R, (-R @ c)[:, None]], 1))
    return np.stack(E)


def test_identical_poses_are_perfect():
    E = _trajectory()
    m = window_pose_metrics(E, E)
    assert m["rot_err_deg"] == pytest.approx(0.0, abs=1e-4)   # arccos near 1
    assert m["trans_err_deg"] == pytest.approx(0.0, abs=1e-4)
    assert m["auc30"] == 1.0 and m["rra15"] == 1.0 and m["rta15"] == 1.0
    assert m["ate_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["sim3_scale"] == pytest.approx(1.0)
    assert m["n_pairs"] == 28


def test_global_sim3_costs_nothing():
    """VGGT anchors camera 0 and predicts translation up to scale: neither may score."""
    E = _trajectory()
    s, Rs, t = 0.37, _rot([1, 2, 3], 71.0), np.array([5.0, -2.0, 9.0])
    # Move the world by the similarity x' = s Rs x + t: centres follow it, the
    # camera orientations rotate with it, and the extrinsic stays rigid.
    Ep = []
    for R, tt in zip(E[:, :, :3], E[:, :, 3]):
        c = -R.T @ tt
        Rp, cp = R @ Rs.T, s * Rs @ c + t
        Ep.append(np.concatenate([Rp, (-Rp @ cp)[:, None]], 1))
    Ep = np.stack(Ep)
    # Re-anchor on camera 0 as VGGT does.
    E0 = np.linalg.inv(np.vstack([Ep[0], [0, 0, 0, 1]]))
    Ep = np.einsum("nij,jk->nik", np.concatenate(
        [Ep, np.tile([[[0, 0, 0, 1]]], (len(Ep), 1, 1))], 1), E0)[:, :3]
    m = window_pose_metrics(Ep, E)
    assert m["rot_err_deg"] == pytest.approx(0.0, abs=1e-4)
    assert m["trans_err_deg"] == pytest.approx(0.0, abs=1e-4)
    assert m["ate_m"] == pytest.approx(0.0, abs=1e-7)
    assert m["sim3_scale"] == pytest.approx(1 / s, rel=1e-6)


def test_known_rotation_error_reads_as_its_angle():
    E = _trajectory(n=2)
    Ep = E.copy()
    Ep[1, :, :3] = _rot([0, 0, 1], 12.0) @ E[1, :, :3]
    m = window_pose_metrics(Ep, E)
    assert m["rot_err_deg"] == pytest.approx(12.0, abs=1e-6)
    assert m["rra5"] == 0.0 and m["rra15"] == 1.0


def test_collapsed_prediction_is_not_perfect():
    """All predicted centres at one point: no Sim(3) may rescue it."""
    E = _trajectory()
    Ep = E.copy()
    Ep[:, :, 3] = 0.0
    m = window_pose_metrics(Ep, E)
    assert m["ate_m"] > 0.3
    assert np.isnan(m["sim3_scale"])
    assert m["rta15"] < 1.0


def test_primitives():
    assert _rot_angle_deg(_rot([1, 1, 0], 33.0)) == pytest.approx(33.0)
    assert _vec_angle_deg(np.array([1, 0, 0]), np.array([0, 1, 0])) == pytest.approx(90.0)
    assert _vec_angle_deg(np.zeros(3), np.zeros(3)) == 0.0
    assert _vec_angle_deg(np.zeros(3), np.ones(3)) == 90.0
    # raw = A @ rotated: the rotated frame's +x (image right) is the raw +y... and
    # the rotated frame's +y (image down) is the raw -x.
    assert np.allclose(A_ROT @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(A_ROT @ [0, 1, 0], [1, 0, 0])
    assert np.allclose(A_ROT @ A_ROT.T, np.eye(3))
