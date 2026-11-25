import math

import pytest

import deblock

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.5] * 16 for _ in range(16)]

CASES = [
    ("identity", lambda: deblock.Deblock().apply(IMG, 0.0) == IMG),
    ("shape", lambda: len(deblock.Deblock().apply(IMG, 1.0)) == 16),
    ("mask", lambda: sum(sum(r) for r in deblock.Deblock().edge_mask(16, 16)) > 0),
    ("bad_strength", lambda: _raises(lambda: deblock.Deblock().apply(IMG, -1.0))),
    ("bad_image", lambda: _raises(lambda: deblock.Deblock().apply([], 1.0))),
    ("type", lambda: isinstance(deblock.Deblock().apply(IMG, 1.0), list)),
    ("det", lambda: deblock.Deblock().apply(IMG, 1.0) == deblock.Deblock().apply(IMG, 1.0)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in deblock.Deblock().apply(IMG, 1.0) for v in row)),
    ("mask_type", lambda: isinstance(deblock.Deblock().edge_mask(8, 8), list)),
    ("nan", lambda: _raises(lambda: deblock.Deblock().apply([[float('nan')]], 1.0))),
    ("no_mut", lambda: deblock.Deblock().apply([r[:] for r in IMG], 1.0) is not None),
    ("flat_preserved", lambda: all(abs(v - 0.5) < 1e-12 for row in deblock.Deblock().apply(IMG, 1.0) for v in row)),
    ("mask_bool", lambda: all(isinstance(v, bool) for row in deblock.Deblock().edge_mask(8, 8) for v in row)),
    ("bad_mask_size", lambda: _raises(lambda: deblock.Deblock().edge_mask(0, 8))),
    ("float", lambda: isinstance(deblock.Deblock().apply(IMG, 1.0)[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
