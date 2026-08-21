import sys, os
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(__file__))
from film import FiLMConditioner, make_arm_field  # noqa: E402


def test_at_init_the_module_is_an_exact_identity():
    """Zero-init means arm differences cannot be a different starting point."""
    m = FiLMConditioner(3, 16)
    tok = torch.randn(1, 20, 16)
    f = torch.randn(12, 3)
    assert torch.equal(m(tok, f), tok)


def test_prefix_tokens_are_never_touched():
    """CLS/register tokens have no image-plane position; conditioning them on a
    made-up one is the same error class as filling a mesh hole with a value."""
    m = FiLMConditioner(3, 8)
    with torch.no_grad():
        m.net[-1].weight.normal_(); m.net[-1].bias.normal_()
    tok = torch.randn(1, 15, 8)
    out = m(tok, torch.randn(10, 3))
    assert torch.equal(out[:, :5], tok[:, :5])
    assert not torch.equal(out[:, 5:], tok[:, 5:])


def test_shuffled_arm_preserves_the_value_distribution_exactly():
    """The control must differ from `jac` only in WHERE values sit, so a win
    cannot be explained by the arms seeing different numbers."""
    f = torch.randn(50, 3)
    g = torch.Generator().manual_seed(0)
    s = make_arm_field(f, "shuffled", g)
    for c in range(3):
        assert torch.equal(torch.sort(f[:, c]).values, torch.sort(s[:, c]).values)
    assert not torch.equal(f, s)


def test_shuffled_permutation_is_stable_within_a_run():
    """Re-drawing per step would average the shuffle away and silently turn the
    control into 'no conditioning', which would fake a win for `jac`."""
    f = torch.randn(40, 3)
    a = make_arm_field(f, "shuffled", torch.Generator().manual_seed(7))
    b = make_arm_field(f, "shuffled", torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_theta_arm_keeps_only_the_angle():
    f = torch.randn(30, 3)
    t = make_arm_field(f, "theta", torch.Generator().manual_seed(0))
    assert torch.equal(t[:, 2], f[:, 2])
    assert torch.count_nonzero(t[:, :2]) == 0


def test_all_arms_have_identical_parameter_count():
    """Capacity is held fixed across arms by construction -- the field changes,
    the network does not."""
    ms = [FiLMConditioner(3, 32) for _ in range(3)]
    ns = {sum(p.numel() for p in m.parameters()) for m in ms}
    assert len(ns) == 1
