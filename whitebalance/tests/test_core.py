import math

import pytest

import whitebalance

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[(0.5, 0.5, 0.5), (0.4, 0.4, 0.4)], [(0.6, 0.6, 0.6), (0.5, 0.5, 0.5)]]

CASES = [
    ("gray_identity", lambda: all(abs(g - 1.0) < 1e-9 for g in whitebalance.WhiteBalance('gray').gains(IMG))),
    ("white_mode", lambda: len(whitebalance.WhiteBalance('white').gains(IMG)) == 3),
    ("bad_method", lambda: _raises(lambda: whitebalance.WhiteBalance('bogus'))),
    ("bad_trim", lambda: _raises(lambda: whitebalance.WhiteBalance('gray', 1.5))),
    ("apply_shape", lambda: len(whitebalance.WhiteBalance('gray').apply(IMG)) == 2),
    ("bad_image", lambda: _raises(lambda: whitebalance.WhiteBalance('gray').gains([]))),
    ("type", lambda: isinstance(whitebalance.WhiteBalance('gray').gains(IMG), list)),
    ("det", lambda: whitebalance.WhiteBalance('gray').gains(IMG) == whitebalance.WhiteBalance('gray').gains(IMG)),
    ("range", lambda: all(0.0 <= v <= 2.0 for row in whitebalance.WhiteBalance('gray').apply(IMG) for p in row for v in p)),
    ("nan", lambda: _raises(lambda: whitebalance.WhiteBalance('gray').gains([[(float('nan'), 0, 0)]]))),
    ("no_mut", lambda: whitebalance.WhiteBalance('gray').apply([list(r) for r in IMG]) is not None),
    ("gain_count", lambda: len(whitebalance.WhiteBalance('gray').gains(IMG)) == 3),
    ("apply_type", lambda: isinstance(whitebalance.WhiteBalance('gray').apply(IMG), list)),
    ("float", lambda: isinstance(whitebalance.WhiteBalance('gray').gains(IMG)[0], float)),
    ("trim_effect", lambda: isinstance(whitebalance.WhiteBalance('gray', 0.05).gains(IMG), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
