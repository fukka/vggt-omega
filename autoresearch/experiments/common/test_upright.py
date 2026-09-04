"""CPU tests for the upright boundary. No weights, no data."""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upright as U


def test_to_and_from_are_inverses():
    for shape in ((3, 8, 8), (8, 8), (2, 1, 3, 8, 8)):
        x = torch.randn(*shape)
        assert torch.equal(U.from_model(U.to_model(x)), x)


def test_the_turn_actually_turns():
    x = torch.zeros(1, 4, 4); x[0, 0, 0] = 1.0          # top-left
    y = U.to_model(x)
    assert float(y[0, 0, 0]) == 0.0, "k=0 would be a silent no-op"
    assert float(y[0].sum()) == 1.0


def test_unroll_preserves_the_angle_and_moves_the_axis():
    """A conjugation cannot change a rotation's angle -- which is why the naive
    reading ('the angle is invariant, so it does not matter') is wrong: the
    error metric compares against a GT expressed in the STORED frame, so it is
    the axis that is being measured."""
    a = math.radians(20.0)
    R = torch.tensor([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)],
                      [0, math.sin(a), math.cos(a)]])
    Ru = U.unroll_R(R)
    ang = lambda M: math.degrees(math.acos(min(1.0, max(-1.0, (float(M.trace()) - 1) / 2))))
    assert ang(Ru) == pytest.approx(ang(R), abs=1e-4)
    assert not torch.allclose(Ru, R, atol=1e-3), "k=3 must move the axis"


def test_unroll_is_invertible():
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    if float(torch.det(R)) < 0:
        R = R @ torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    back = U.unroll_R(U.unroll_R(R, sign=1.0), sign=-1.0)
    assert torch.allclose(back, R, atol=1e-5)


def test_forward_z_refuses_a_range_install():
    """The whole reason forward_z exists: converting inside forward() would
    apply the camera's cos map to a prediction sitting in the rotated frame."""
    class FakeBB:
        depth_convention = "range"
        def forward(self, x): raise AssertionError("must not be reached")
    with pytest.raises(RuntimeError, match="depth_convention='z'"):
        U.forward_z(FakeBB(), torch.zeros(3, 8, 8))
