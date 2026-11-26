"""约束级回归测试：对应题面中可度量的每一条要求。

这些用例独立于 tests/test_core.py 的冒烟用例，直接验证：
只在 8 像素块边界滤波、块内不模糊、与逐像素参考实现一致、强度单调、
非法/非有限输入分类报错、100 万像素性能上界与并发安全。
"""

import math
import threading
import time

import pytest

import deblock
from deblock import core


def _rand_image(h, w, seed, channels=None):
    """确定性伪随机图（不依赖 NumPy，行内用 LCG）。"""
    state = seed

    def gen():
        nonlocal state
        state = (1103515245 * state + 12345) % (1 << 31)
        return state / float(1 << 31)

    if channels is None:
        return [[gen() for _ in range(w)] for _ in range(h)]
    return [[[gen() for _ in range(channels)] for _ in range(w)] for _ in range(h)]


def _make_block_image(rows, cols, cell=8, values=(0.1, 0.55, 0.9, 0.2)):
    """由 8x8 常数块拼成的图：块内平坦，块间（两个方向）都有台阶。"""
    h, w = rows * cell, cols * cell
    n = len(values)
    return [
        [values[((y // cell) * n + (x // cell)) % n] for x in range(w)]
        for y in range(h)
    ]


def _may_change(x, y, w, h):
    """与 ``edge_mask`` 同定义的边界邻接判定：内部 8-seam 两侧一格。"""
    return (x > 0 and x % 8 in (0, 7)) or (y > 0 and y % 8 in (0, 7))


# --------------------------------------------------------------------------- #
# 块边界与细节保持
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strength", [0.25, 0.5, 1.0, 3.0])
def test_changes_only_near_block_edges(strength):
    """不在任何 seam 两侧的像素（块内部）必须逐位不变（不模糊细节）。"""
    rng = _rand_image(37, 41, seed=7)
    out = deblock.Deblock().apply(rng, strength)
    for y in range(37):
        for x in range(41):
            if not _may_change(x, y, 41, 37):
                assert out[y][x] == rng[y][x], (y, x)


def test_strict_interior_bit_identical():
    """最严格解释：块内第 1..6 列/行（离边界 >=2）逐位不变。"""
    rng = _rand_image(40, 40, seed=11)
    out = deblock.Deblock().apply(rng, 1.0)
    for y in range(40):
        for x in range(40):
            if 1 <= x % 8 <= 6 and 1 <= y % 8 <= 6:
                assert out[y][x] == rng[y][x], (y, x)


def test_single_block_untouched():
    """一张恰好 8x8 的图没有内部 seam，滤波必须恒等。"""
    rng = _rand_image(8, 8, seed=3)
    assert deblock.Deblock().apply(rng, 1.0) == rng


def test_flat_image_identity():
    flat = [[0.5] * 16 for _ in range(16)]
    for s in (0.0, 0.5, 1.0):
        out = deblock.Deblock().apply(flat, s)
        assert all(abs(v - 0.5) < 1e-12 for row in out for v in row)


def test_linear_ramp_not_filtered():
    """平滑线性渐变跨 seam 时局部活动 >= 跨边界阶跃，不应被平滑。"""
    ramp = [[x / 63.0 for x in range(64)] for _ in range(8)]
    out = deblock.Deblock().apply(ramp, 1.0)
    assert out == ramp


def test_interior_texture_preserved():
    """块内高频细节（不在 seam 两侧）完全保留。"""
    rng = _rand_image(24, 24, seed=5)
    out = deblock.Deblock().apply(rng, 1.0)
    for y in range(24):
        for x in range(24):
            if not _may_change(x, y, 24, 24):
                assert out[y][x] == rng[y][x]


# --------------------------------------------------------------------------- #
# 块效应确实被压制（强度自适应）
# --------------------------------------------------------------------------- #
def test_blocking_metric_decreases_with_strength():
    img = _make_block_image(4, 4)
    d = deblock.Deblock()

    def seam_jump(a):
        worst = 0.0
        for y in range(1, len(a) - 1):
            for x in range(8, len(a[0]), 8):
                worst = max(worst, abs(a[y][x] - a[y][x - 1]))
        return worst

    jumps = [seam_jump(d.apply(img, s)) for s in (0.0, 0.5, 1.0)]
    assert jumps[0] > 0  # 原始台阶存在
    assert jumps[0] > jumps[1] > jumps[2]
    assert jumps[2] <= jumps[0] * 0.5 + 1e-12


def test_weak_edge_adaptive_steering():
    """块内存在与之相当的真实纹理时，不把它当块效应抹平。"""
    img = [[0.2] * 24 for _ in range(8)]
    for y in range(8):
        # x=16 seam：跨边界阶跃 0.4，但两侧紧邻格点有更强的块内活动
        img[y][14] = 1.0
        img[y][16] = 0.6
        img[y][17] = 0.5
    out = deblock.Deblock().apply(img, 1.0)
    assert [row[14:] for row in out] == [row[14:] for row in img]


# --------------------------------------------------------------------------- #
# 与逐像素参考实现一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape,strength", [
    ((1, 1), 1.0), ((7, 9), 0.3), ((8, 8), 1.0), ((17, 19), 0.87),
    ((33, 34), 1.0), ((5, 70), 0.5), ((70, 5), 0.2),
])
def test_matches_reference_2d(shape, strength):
    img = _rand_image(*shape, seed=13)
    a = deblock.Deblock().apply(img, strength)
    b = deblock.Deblock().reference_apply(img, strength)
    err = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    assert err <= 1e-9


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_matches_reference_3d(channels):
    img = _rand_image(19, 23, seed=17, channels=channels)
    a = deblock.Deblock().apply(img, 0.6)
    b = deblock.Deblock().reference_apply(img, 0.6)
    err = max(
        abs(x - y)
        for ra, rb in zip(a, b)
        for pa, pb in zip(ra, rb)
        for x, y in zip(pa, pb)
    )
    assert err <= 1e-9


def test_reference_differs_from_input_where_expected():
    """参考实现不是空操作：确实改变了强块边界。"""
    img = _make_block_image(2, 2)
    out = deblock.Deblock().reference_apply(img, 1.0)
    assert out[0][7] != img[0][7]


# --------------------------------------------------------------------------- #
# 强度语义
# --------------------------------------------------------------------------- #
def test_zero_strength_exact_identity():
    for shape in ((1, 1), (8, 20), (23, 17)):
        img = _rand_image(*shape, seed=21)
        assert deblock.Deblock().apply(img, 0.0) == img
    int_img = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    assert deblock.Deblock().apply(int_img, 0.0) == int_img


def test_zero_strength_does_not_mutate_input():
    img = _make_block_image(2, 2)
    snapshot = [row[:] for row in img]
    deblock.Deblock().apply(img, 0.5)
    assert img == snapshot


def test_deterministic():
    img = _rand_image(30, 30, seed=23)
    d = deblock.Deblock()
    assert d.apply(img, 0.7) == d.apply(img, 0.7)
    assert d.apply(img, 0.7) == deblock.Deblock().apply(img, 0.7)


def test_output_range_preserved_for_unit_input():
    img = _rand_image(40, 40, seed=29)
    out = deblock.Deblock().apply(img, 1.0)
    assert min(min(r) for r in out) >= 0.0
    assert max(max(r) for r in out) <= 1.0


def test_apply_accepts_numpy_array_and_returns_list():
    np = pytest.importorskip("numpy")
    img = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    out = deblock.Deblock().apply(img, 1.0)
    assert isinstance(out, list) and isinstance(out[0][0], float)


# --------------------------------------------------------------------------- #
# 报错分类
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strength", [-1.0, -1e-9, float("nan"), float("inf"),
                                      float("-inf"), "1", None, [1.0]])
def test_invalid_strength_raises_value_error(strength):
    with pytest.raises(ValueError):
        deblock.Deblock().apply([[0.5]], strength)


def test_negative_and_nonfinite_strength_are_distinguishable():
    img = [[0.5]]
    with pytest.raises(core.InvalidStrengthError) as neg:
        deblock.Deblock().apply(img, -1.0)
    with pytest.raises(core.NonFiniteStrengthError) as nonfinite:
        deblock.Deblock().apply(img, float("nan"))
    assert type(neg.value) is not type(nonfinite.value)
    assert not isinstance(neg.value, core.NonFiniteInputError)


@pytest.mark.parametrize("image", [[], [[]], [["x"]], [[None]], [[0.0, 1.0], [0.5]],
                                  [[[0.0], [1.0]], [[2.0]]], "abc", 3])
def test_invalid_image_raises_value_error(image):
    with pytest.raises(ValueError):
        deblock.Deblock().apply(image, 1.0)


@pytest.mark.parametrize("image", [
    [[float("nan")]], [[float("inf")]], [[-float("inf")]],
    [[0.0, float("nan")], [0.0, 0.0]],
    [[[0.0, 1.0, float("inf")]]],
])
def test_nonfinite_image_is_classified(image):
    with pytest.raises(core.NonFiniteImageError) as err:
        deblock.Deblock().apply(image, 1.0)
    assert isinstance(err.value, ValueError)
    assert isinstance(err.value, core.InvalidImageError)


@pytest.mark.parametrize("w,h", [(0, 8), (-8, 8), (2.5, 8), (float("nan"), 8),
                                 (float("inf"), 8), ("8", 8), (None, 8)])
def test_bad_mask_size_raises(w, h):
    with pytest.raises(ValueError):
        deblock.Deblock().edge_mask(w, h)


def test_nan_is_not_conflated_with_bad_shape():
    """非有限输入必须能与普通非法图像区分开。"""
    with pytest.raises(core.NonFiniteImageError) as e_nonfinite:
        deblock.Deblock().apply([[float("nan")]], 1.0)
    with pytest.raises(core.InvalidImageError) as e_empty:
        deblock.Deblock().apply([], 1.0)
    assert not isinstance(e_empty.value, core.NonFiniteInputError)
    assert isinstance(e_nonfinite.value, core.InvalidImageError)


# --------------------------------------------------------------------------- #
# 掩码
# --------------------------------------------------------------------------- #
def test_mask_shape_and_types():
    mask = deblock.Deblock().edge_mask(19, 13)
    assert isinstance(mask, list) and len(mask) == 13
    assert all(len(row) == 19 for row in mask)
    assert all(isinstance(v, bool) for row in mask for v in row)


def test_mask_marks_seams_and_covers_filtered_pixels():
    """掩码必须覆盖所有可能被 apply 改变的像素（seam 两侧）。"""
    w = h = 32
    mask = deblock.Deblock().edge_mask(w, h)
    rng = _rand_image(h, w, seed=31)
    blocky = _make_block_image(4, 4)
    for img in (rng, blocky):
        out = deblock.Deblock().apply(img, 1.0)
        for y in range(h):
            for x in range(w):
                if out[y][x] != img[y][x]:
                    assert mask[y][x], (y, x)


def test_mask_has_boundary_rows():
    mask = deblock.Deblock().edge_mask(16, 16)
    assert any(mask[8]) and any(mask[7])
    assert not mask[4][4]  # 块内部


# --------------------------------------------------------------------------- #
# 并发安全
# --------------------------------------------------------------------------- #
def test_concurrent_apply_is_consistent():
    d = deblock.Deblock()
    img = _rand_image(96, 96, seed=37)
    expected = d.apply(img, 1.0)
    results = [None] * 12
    errors = []

    def worker(i):
        try:
            results[i] = d.apply(img, 1.0)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == expected for r in results)


def test_concurrent_mixed_calls():
    d = deblock.Deblock()
    img = _rand_image(64, 64, seed=41)
    outcomes = [None] * 8
    errors = []

    def worker(i):
        try:
            if i % 2:
                outcomes[i] = d.apply(img, 0.5)
            else:
                outcomes[i] = d.reference_apply(img, 0.5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    err = 0.0
    for a, b in zip(outcomes[0], outcomes[1]):
        for x, y in zip(a, b):
            err = max(err, abs(x - y))
    assert err <= 1e-9


# --------------------------------------------------------------------------- #
# 性能上界
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("channels", [None, 3])
def test_one_megapixel_under_budget(channels):
    d = deblock.Deblock()
    d.apply(_rand_image(256, 256, seed=42, channels=channels), 1.0)  # 预热

    side = 1000  # 1,000,000 像素
    img = _rand_image(side, side, seed=43, channels=channels)
    best = float("inf")
    for _ in range(3):  # 取最好一次，避免与其它负载并发导致抖动
        start = time.perf_counter()
        d.apply(img, 1.0)
        best = min(best, time.perf_counter() - start)
    assert best < 0.700, f"{best * 1000:.1f} ms > 700 ms"
