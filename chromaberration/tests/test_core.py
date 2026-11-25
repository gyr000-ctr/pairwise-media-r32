import math

import pytest

import chromaberration

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[(0.5, 0.5, 0.5)] * 4 for _ in range(4)]

CASES = [
    ("center_fixed", lambda: abs(chromaberration.ChromaAberration(1.0, 1.0).sample(0, 0.5, 0.5) - 0.5) < 1e-12),
    ("shape", lambda: len(chromaberration.ChromaAberration(1.0, 1.0).apply(IMG)) == 4),
    ("bad_k", lambda: _raises(lambda: chromaberration.ChromaAberration(0.0, 1.0))),
    ("bad_channel", lambda: _raises(lambda: chromaberration.ChromaAberration(1.0, 1.0).sample(9, 0.5, 0.5))),
    ("bad_image", lambda: _raises(lambda: chromaberration.ChromaAberration(1.0, 1.0).apply([]))),
    ("type", lambda: isinstance(chromaberration.ChromaAberration(1.0, 1.0).apply(IMG), list)),
    ("det", lambda: chromaberration.ChromaAberration(1.0, 1.0).apply(IMG) == chromaberration.ChromaAberration(1.0, 1.0).apply(IMG)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in chromaberration.ChromaAberration(1.0, 1.0).apply(IMG) for p in row for v in p)),
    ("sample_type", lambda: isinstance(chromaberration.ChromaAberration(1.0, 1.0).sample(0, 0.3, 0.3), float)),
    ("nan", lambda: _raises(lambda: chromaberration.ChromaAberration(1.0, 1.0).sample(0, float('nan'), 0.5))),
    ("no_mut", lambda: chromaberration.ChromaAberration(1.0, 1.0).apply([list(r) for r in IMG]) is not None),
    ("k_effect", lambda: chromaberration.ChromaAberration(1.1, 1.0).sample(0, 0.25, 0.5) != chromaberration.ChromaAberration(1.0, 1.0).sample(0, 0.25, 0.5)),
    ("apply_type", lambda: isinstance(chromaberration.ChromaAberration(1.0, 1.0).apply(IMG), list)),
    ("float", lambda: isinstance(chromaberration.ChromaAberration(1.0, 1.0).apply(IMG)[0][0][0], float)),
    ("clamp", lambda: isinstance(chromaberration.ChromaAberration(2.0, 2.0).sample(1, 0.01, 0.01), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
