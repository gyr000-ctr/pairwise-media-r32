import math

import pytest

import exposure

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.1, 0.2], [0.3, 0.4]]

CASES = [
    ("gain_positive", lambda: exposure.AutoExposure(0.5, 8.0).estimate(IMG) > 0.0),
    ("bounded", lambda: exposure.AutoExposure(0.5, 2.0).estimate(IMG) <= 2.0),
    ("apply_shape", lambda: len(exposure.AutoExposure(0.5, 8.0).apply(IMG)) == 2),
    ("bad_target", lambda: _raises(lambda: exposure.AutoExposure(0.0, 8.0))),
    ("bad_gain", lambda: _raises(lambda: exposure.AutoExposure(0.5, 0.0))),
    ("empty", lambda: _raises(lambda: exposure.AutoExposure(0.5, 8.0).estimate([]))),
    ("type", lambda: isinstance(exposure.AutoExposure(0.5, 8.0).estimate(IMG), float)),
    ("det", lambda: exposure.AutoExposure(0.5, 8.0).estimate(IMG) == exposure.AutoExposure(0.5, 8.0).estimate(IMG)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in exposure.AutoExposure(0.5, 8.0).apply(IMG) for v in row)),
    ("nan", lambda: _raises(lambda: exposure.AutoExposure(0.5, 8.0).estimate([[float('nan')]]))),
    ("no_mut", lambda: exposure.AutoExposure(0.5, 8.0).apply([r[:] for r in IMG]) is not None),
    ("target_effect", lambda: exposure.AutoExposure(0.8, 8.0).estimate(IMG) >= exposure.AutoExposure(0.2, 8.0).estimate(IMG)),
    ("float", lambda: isinstance(exposure.AutoExposure(0.5, 8.0).apply(IMG)[0][0], float)),
    ("quantile", lambda: exposure.AutoExposure(0.5, 8.0).quantile(IMG, 0.5) >= 0.0),
    ("bad_quantile", lambda: _raises(lambda: exposure.AutoExposure(0.5, 8.0).quantile(IMG, 1.5))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
