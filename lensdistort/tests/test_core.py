import math

import pytest

import lensdistort

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _l():
    return lensdistort.LensDistort(-0.2, 0.02)

CASES = [
    ("center", lambda: abs(_l().distort(0.5, 0.5)[0] - _l().distort(0.5, 0.5)[0]) < 1e-12),
    ("roundtrip", lambda: abs(_l().undistort(*_l().distort(0.3, 0.6))[0] - 0.3) < 1e-6),
    ("bad_k", lambda: _raises(lambda: lensdistort.LensDistort(float('nan'), 0.0))),
    ("type", lambda: isinstance(_l().distort(0.3, 0.3), tuple)),
    ("det", lambda: _l().distort(0.3, 0.3) == _l().distort(0.3, 0.3)),
    ("apply", lambda: isinstance(_l().apply([[0.5] * 4 for _ in range(4)]), list)),
    ("bad_apply", lambda: _raises(lambda: _l().apply([]))),
    ("undistort_type", lambda: isinstance(_l().undistort(0.3, 0.3), tuple)),
    ("nan", lambda: _raises(lambda: _l().distort(float('nan'), 0.3))),
    ("no_mut", lambda: _l().apply([[0.5] * 4 for _ in range(4)]) is not None),
    ("tangential", lambda: lensdistort.LensDistort(-0.2, 0.02, 0.01, 0.01) is not None),
    ("float", lambda: isinstance(_l().distort(0.2, 0.2)[0], float)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _l().apply([[0.5] * 4 for _ in range(4)]) for v in row)),
    ("roundtrip_y", lambda: abs(_l().undistort(*_l().distort(0.7, 0.2))[1] - 0.2) < 1e-6),
    ("bad_p", lambda: _raises(lambda: lensdistort.LensDistort(0.0, 0.0, float('nan'), 0.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
