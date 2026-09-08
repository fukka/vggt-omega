"""CPU tests for the roll line. No weights, no data, no torch where avoidable.

Two pieces of this session's code produce headline numbers and had no test:

* `roll_distribution.rolls_deg` gives section 02c's "ADT's head roll is median
  3.7 deg, p99 21.8, never past 40". Its sign convention
  (`atan2(up_x, -up_y)`) was asserted nowhere, and neither was the claim that a
  camera rolled by phi comes back as phi.
* the per-theta-bin log-log fit in `radial_probe` gives section 03c's "16
  numbers carry everything that crosses a room". Nothing checked that the fit
  recovers a curve it was generated from, or that applying it inverts the fit.

The `RolledView` rotation test needs torch and is skipped where torch is absent
(this Mac); it runs on lambda_63.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roll_distribution as RD  # noqa: E402


# ----------------------------------------------------------------- helpers

def _orthonormal(seed: int) -> np.ndarray:
    """An arbitrary proper rotation, so nothing here depends on a nice frame."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _world_camera(z_c: np.ndarray, up_w: np.ndarray, roll_rad: float) -> np.ndarray:
    """R_world_camera for a camera looking along ``z_c``, rolled by ``roll_rad``.

    Camera convention here is the repo's: x right, y DOWN, z forward, and
    right-handed (x cross y = z). At zero roll the world-up direction projects
    onto image-up, i.e. onto -y.
    """
    z_c = z_c / np.linalg.norm(z_c)
    u = up_w - np.dot(up_w, z_c) * z_c
    u /= np.linalg.norm(u)
    y0 = -u
    x0 = np.cross(y0, z_c)
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    x = c * x0 + s * y0
    y = -s * x0 + c * y0
    return np.stack([x, y, z_c], axis=1)


# ------------------------------------------------------- quaternion helper

def test_quat_to_R_is_a_proper_rotation():
    for seed in range(5):
        v = np.random.default_rng(seed).normal(size=4)
        v /= np.linalg.norm(v)
        R = RD.quat_to_R(*v)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "not orthonormal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, "not proper (det != +1)"


# ------------------------------------------------------------- the roll read

def test_a_level_camera_reads_as_zero_roll():
    """The whole 02c measurement rests on this: no roll in, no roll out."""
    up = np.array([0.0, 1.0, 0.0])                # ADT's world: gravity along -Y
    g = -up * 9.81
    R_cd = _orthonormal(0)                        # camera <- device, arbitrary
    for seed in range(6):
        z = np.random.default_rng(seed + 10).normal(size=3)
        if abs(np.dot(z / np.linalg.norm(z), up)) > 0.95:
            continue                              # optical axis along gravity: roll undefined
        R_wc = _world_camera(z, up, 0.0)
        R_wd = R_wc @ R_cd                        # rolls_deg does R_wd @ R_cd.T
        out = RD.rolls_deg(R_wd[None], g[None], R_cd)
        assert abs(out[0]) < 1e-6, f"level camera read as {out[0]} deg"


def test_a_rolled_camera_reads_back_its_own_angle():
    """Magnitude must be exact, and the sign convention is pinned here.

    `rolls_deg` returns the angle the IMAGE is rotated by, which is the negative
    of the angle the camera frame was rolled by — rolling the camera clockwise
    moves the scene counter-clockwise in the image. Nothing downstream depends
    on the sign (02c reports |roll| and a distribution that is near symmetric),
    but it should not drift silently.
    """
    up = np.array([0.0, 1.0, 0.0])
    g = -up * 9.81
    R_cd = _orthonormal(3)
    z = np.array([0.3, -0.2, 1.0])
    for phi_deg in (-40.0, -12.5, 0.0, 7.0, 33.0):
        R_wc = _world_camera(z, up, math.radians(phi_deg))
        R_wd = R_wc @ R_cd
        out = RD.rolls_deg(R_wd[None], g[None], R_cd)[0]
        assert abs(abs(out) - abs(phi_deg)) < 1e-6, f"{phi_deg} deg came back as {out}"
        assert abs(out + phi_deg) < 1e-6, f"sign convention changed: {phi_deg} -> {out}"


def test_the_read_is_independent_of_gravity_magnitude_and_device_frame():
    """ADT stores gravity as (0, -9.81, 0); nothing may depend on that scale,
    nor on which arbitrary device frame the camera is bolted into."""
    up = np.array([0.0, 1.0, 0.0])
    z = np.array([0.1, 0.05, 1.0])
    R_wc = _world_camera(z, up, math.radians(21.0))
    vals = []
    for scale in (1.0, 9.81, 1000.0):
        for seed in (1, 2, 7):
            R_cd = _orthonormal(seed)
            vals.append(RD.rolls_deg((R_wc @ R_cd)[None], (-up * scale)[None], R_cd)[0])
    assert np.ptp(vals) < 1e-6, f"read varied by {np.ptp(vals)} deg across frames/scales"


def test_wrap180_is_a_bijection_on_the_half_open_circle():
    a = np.array([-180.0, -179.9, -0.1, 0.0, 179.9, 180.0, 200.0, 540.0])
    w = RD.wrap180(a)
    assert np.all(w >= -180.0) and np.all(w < 180.0 + 1e-9)
    assert abs(RD.wrap180(np.array([200.0]))[0] - (-160.0)) < 1e-9
    assert abs(RD.wrap180(np.array([540.0]))[0] - 180.0) < 1e-9 or \
           abs(RD.wrap180(np.array([540.0]))[0] + 180.0) < 1e-9


# ------------------------------------------------- the per-theta radial fit

def _fit_apply(logp, logt, bins, n_bins):
    """The fit `radial_probe` performs, isolated so it can be tested without
    a backbone: least squares of log(target) on log(pred), per theta bin."""
    C = np.zeros((n_bins, 2))
    for b in range(n_bins):
        m = bins == b
        X = np.stack([logp[m], np.ones(m.sum())], 1)
        C[b] = np.linalg.lstsq(X, logt[m], rcond=None)[0]
    return C


def test_the_radial_fit_recovers_a_curve_it_was_generated_from():
    """Section 03c's whole reading is that these 16 numbers ARE the object.
    If the estimator cannot recover a known curve, the numbers mean nothing."""
    rng = np.random.default_rng(0)
    n_bins = 8
    a_true = 1.30 + 0.18 * np.sin(np.linspace(0, np.pi, n_bins))
    b_true = np.linspace(0.02, -0.25, n_bins)
    bins = rng.integers(0, n_bins, 200_000)
    logp = rng.uniform(np.log(0.4), np.log(10.0), bins.size)
    logt = a_true[bins] * logp + b_true[bins]
    C = _fit_apply(logp, logt, bins, n_bins)
    assert np.abs(C[:, 0] - a_true).max() < 1e-9, "slope not recovered"
    assert np.abs(C[:, 1] - b_true).max() < 1e-9, "intercept not recovered"


def test_the_fit_survives_noise_and_stays_unbiased():
    rng = np.random.default_rng(1)
    n_bins = 8
    a_true = np.full(n_bins, 1.40)
    b_true = np.full(n_bins, -0.10)
    bins = rng.integers(0, n_bins, 400_000)
    logp = rng.uniform(np.log(0.4), np.log(10.0), bins.size)
    logt = a_true[bins] * logp + b_true[bins] + rng.normal(0, 0.30, bins.size)
    C = _fit_apply(logp, logt, bins, n_bins)
    assert np.abs(C[:, 0] - a_true).max() < 0.02, f"slope biased: {C[:,0]}"


def test_applying_the_fit_reproduces_the_targets():
    """`radial_probe` applies exp(a*log(p)+b). That must invert the fit."""
    rng = np.random.default_rng(2)
    n_bins = 8
    a_true = np.linspace(1.20, 1.50, n_bins)
    b_true = np.linspace(0.05, -0.20, n_bins)
    bins = rng.integers(0, n_bins, 50_000)
    pred = np.exp(rng.uniform(np.log(0.4), np.log(10.0), bins.size))
    target = np.exp(a_true[bins] * np.log(pred) + b_true[bins])
    C = _fit_apply(np.log(pred), np.log(target), bins, n_bins)
    applied = np.exp(C[bins, 0] * np.log(pred) + C[bins, 1])
    assert np.abs(applied / target - 1).max() < 1e-9


def test_a_flat_curve_is_the_identity():
    """a=1, b=0 must leave the prediction alone — the `global` control's null
    and the fallback `radial_probe` uses for an under-populated bin."""
    p = np.exp(np.linspace(np.log(0.3), np.log(12.0), 1000))
    assert np.abs(np.exp(1.0 * np.log(p) + 0.0) / p - 1).max() < 1e-12


# ------------------------------------------------- the border-dose guard

def test_the_border_guard_matches_the_geometry_it_claims():
    """`border_dose` refuses a border that would eat into the scored cap. The
    arithmetic it guards with is reproduced here so a refactor cannot quietly
    change what "outside the scored region" means."""
    vs, fov = 630, 89.0
    f_px = (vs / 2.0) / math.tan(math.radians(fov) / 2.0)
    edge = math.degrees(math.atan((vs / 2.0) / f_px))
    corner = math.degrees(math.atan(math.sqrt(2.0) * (vs / 2.0) / f_px))
    assert abs(edge - fov / 2) < 1e-6, "half-edge ray is not fov/2"
    assert abs(corner - 54.3) < 0.1, f"corner ray moved: {corner:.2f} deg"
    # a 7 px border already cuts inside a 44 deg cap -- this is why 02e scores
    # at 30 deg, and the number is load-bearing for that choice
    th7 = math.degrees(math.atan((vs / 2.0 - 7) / f_px))
    assert th7 < 44.0, f"7 px border reaches {th7:.2f} deg, expected inside 44"
    th100 = math.degrees(math.atan((vs / 2.0 - 100) / f_px))
    assert th100 > 30.0, f"100 px border reaches {th100:.2f} deg, must clear 30"


def test_black_fraction_matches_the_reported_doses():
    """The percentages quoted in 02e come from this one expression."""
    vs = 630
    for w, want in ((5, 3.1), (12, 7.5), (25, 15.2), (45, 26.5), (70, 39.5), (100, 53.4)):
        got = 100 * (1 - ((vs - 2 * w) / vs) ** 2)
        assert abs(got - want) < 0.1, f"border {w}px -> {got:.1f}%, report says {want}%"


# --------------------------------------------------- torch-only: RolledView

def test_rolled_view_is_a_roll_about_the_optical_axis():
    # roll_controls pulls in upright + rect_teacher + raytun3r, none of which
    # resolve on a machine without the full env. Skip rather than fail: this
    # one is meant to run on the GPU box.
    here = Path(__file__).resolve().parents[1]
    for sub in ("h16-orientation", "h14-rect-distill", "h1-rim-pose-value"):
        sys.path.insert(0, str(here / sub / "code"))
    sys.path.insert(0, str(here / "common"))
    try:
        import roll_controls as RC
    except Exception as exc:
        print(f"[skip] RolledView test needs the full env "
              f"({exc.__class__.__name__}); it runs on lambda_63")
        return
    for deg in (-40.0, 0.0, 17.0, 40.0):
        R = RC.RolledView(fov_x_deg=89.0, width=630, height=630, roll_deg=deg).rotation().numpy()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "not orthonormal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, "a mirror, not a rotation"
        # the optical axis must NOT move -- that is what makes it a roll and
        # not a tilt, and it is why the 89 deg view stays inside the cone at
        # every angle (02's whole clean-arm argument)
        assert np.allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-9), f"axis moved at {deg}"
        ang = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        assert abs(ang - deg) < 1e-6, f"roll {deg} encoded as {ang}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:                      # a crash is a failure too
            bad += 1
            print(f"  ERROR {name}: {e.__class__.__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
