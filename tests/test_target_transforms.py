import numpy as np
from recsyslib.target_transforms import (
    interact_to_confidence,
    loginteract_to_confidence,
    scale_zero_one,
)


def test_interact_to_confidence():
    d = np.arange(1, 10, 1)
    expected = d * 10
    c = interact_to_confidence(d, 10)
    assert (c == expected).all()
    expected1 = d * 40
    c1 = interact_to_confidence(d, 40)
    assert (c1 == expected1).all()


def test_loginteract_to_confidence():
    d = np.geomspace(1, 1000, num=4)
    expected = 1.0 + 10 * np.log1p(d / 1e-5)
    c = loginteract_to_confidence(d, 10)
    assert np.allclose(c, expected)


def test_scale_zero_one():
    d = np.random.normal(size=10)
    s = scale_zero_one(d)
    assert np.max(d) == 1.0
    assert np.min(d) == 0.0
