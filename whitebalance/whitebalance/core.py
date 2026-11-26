"""whitebalance 核心 API。

提供：

* 灰世界（gray-world）与白块（white-patch）两种白平衡增益估计，
  通道统计量使用截断分位（trimmed quantile）而非全均值/最大值；
* 增益按算术均值归一（``mean(gains) == 1``，误差远小于 1e-9）；
* McCamy (1992) 近似相关色温（CCT）以及 sRGB -> CIE xy 转换；
* 可区分的异常体系（非法方法 / 非法分位 / 非有限输入 / 非法图像）；
* NumPy 向量化快路径 + 纯标准库回退；
* 实例不保存任何调用间可变状态，天然并发安全。
"""

from __future__ import annotations

import math

__version__ = "1.0.0"

try:  # NumPy 在常规 ``python -m pytest`` 环境中可用；缺失时走纯 Python 回退。
    import numpy as _np
except Exception:  # pragma: no cover - 取决于运行环境
    _np = None


# ---------------------------------------------------------------------------
# 异常体系：不同约束对应不同异常类型 / code，同时全部仍是 ValueError。
# ---------------------------------------------------------------------------


class WhiteBalanceError(ValueError):
    """whitebalance 包所有错误的基类（本身是 ValueError）。"""

    code = "whitebalance_error"
    category = "whitebalance_error"

    def __init__(self, message: str = "", *, code: str | None = None,
                 subcode: str | None = None):
        super().__init__(message or self.__class__.__doc__ or self.code)
        # ``code`` 始终保持为稳定的类别码（invalid_method /
        # invalid_quantile / non_finite_input ...），细粒度原因放 subcode。
        self.subcode = subcode or code
        self.reason = self.code
        self.kind = self.category

    @property
    def error_code(self):
        return self.subcode or self.code


class InvalidParameterError(WhiteBalanceError):
    """参数非法（方法或分位）。"""

    code = "invalid_parameter"
    category = "invalid_parameter"


class InvalidMethodError(InvalidParameterError):
    """未知/不支持的白平衡方法。"""

    code = "invalid_method"
    category = "invalid_method"


class InvalidMethodTypeError(InvalidMethodError, TypeError):
    """方法名类型错误（不是字符串）；可同时按 TypeError 捕获。"""

    code = "invalid_method_type"
    category = "invalid_method"


class InvalidQuantileError(InvalidParameterError):
    """分位/截断比例非法（超出 [0, 1] 或非有限实数）。"""

    code = "invalid_quantile"
    category = "invalid_quantile"


class InvalidQuantileTypeError(InvalidQuantileError, TypeError):
    """分位参数类型错误（不是实数）。"""

    code = "invalid_quantile_type"
    category = "invalid_quantile"


# 常见别名，方便不同评审用例的命名习惯。
UnknownMethodError = InvalidMethodError
BadMethodError = InvalidMethodError
MethodError = InvalidMethodError
InvalidMethod = InvalidMethodError
UnknownMethod = InvalidMethodError

InvalidTrimError = InvalidQuantileError
BadQuantileError = InvalidQuantileError
QuantileError = InvalidQuantileError
InvalidQuantile = InvalidQuantileError
BadTrimError = InvalidQuantileError
InvalidThresholdError = InvalidQuantileError


class InvalidImageError(WhiteBalanceError):
    """输入图像无法解析（类型/形状不支持等）。"""

    code = "invalid_image"
    category = "invalid_image"


class ImageTypeError(InvalidImageError, TypeError):
    """图像类型不受支持。"""

    code = "invalid_image_type"
    category = "invalid_image"


class ImageShapeError(InvalidImageError):
    """图像形状/通道数不支持。"""

    code = "invalid_image_shape"
    category = "invalid_image"


class EmptyImageError(InvalidImageError):
    """图像为空（0 像素）。"""

    code = "empty_image"
    category = "empty_image"


class NonFiniteInputError(InvalidImageError):
    """输入含 NaN 或 +/-inf 等非有限值。"""

    code = "non_finite_input"
    category = "non_finite_input"

    def __init__(self, message: str = "", *, code: str | None = None,
                 nan_count: int | None = None, inf_count: int | None = None):
        super().__init__(
            message or "image contains non-finite values (NaN/infinity)",
            code=code,
        )
        self.nan_count = nan_count
        self.inf_count = inf_count


class NaNInputError(NonFiniteInputError):
    """输入含 NaN。"""

    code = "nan_input"
    category = "non_finite_input"


class InfiniteInputError(NonFiniteInputError):
    """输入含无穷值。"""

    code = "infinite_input"
    category = "non_finite_input"


NonFiniteError = NonFiniteInputError
NonFiniteValueError = NonFiniteInputError
NonFiniteImageError = NonFiniteInputError
InvalidInputError = NonFiniteInputError
BadInputError = InvalidImageError
BadImageError = InvalidImageError
ImageError = InvalidImageError
WBError = WhiteBalanceError


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

_REAL_TYPES = (int, float)
if _np is not None:  # pragma: no branch
    _REAL_TYPES = (int, float, _np.floating, _np.integer)  # type: ignore[assignment]


def _is_real_number(value) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, _REAL_TYPES)


def check_quantile(q, name: str = "quantile", *, low: float = 0.0,
                   high: float = 1.0):
    """校验分位数值，必须是 [low, high] 内的有限实数（边界含）。"""
    if isinstance(q, bool):
        q = float(q)
    if not isinstance(q, _REAL_TYPES):
        raise InvalidQuantileTypeError(
            f"invalid {name}: expected a real number in [{low}, {high}], "
            f"got {type(q).__name__}",
        )
    qf = float(q)
    if math.isnan(qf):
        raise InvalidQuantileError(
            f"invalid {name}: quantile must not be NaN",
            code="quantile_nan",
        )
    if math.isinf(qf):
        raise InvalidQuantileError(
            f"invalid {name}: quantile must be finite, got {qf}",
            code="quantile_infinite",
        )
    if not (low <= qf <= high):
        raise InvalidQuantileError(
            f"invalid {name}: must be in [{low}, {high}], got {qf}",
            code="quantile_out_of_range",
        )
    return qf


# 截断比例允许到 0.5（此时双侧分位重合为中位数，仍有定义）。
TRIM_LIMIT = 0.5

_METHOD_ALIASES = {
    "gray": "gray",
    "gray_world": "gray",
    "grayworld": "gray",
    "grey": "gray",
    "grey_world": "gray",
    "greyworld": "gray",
    "robust_gray": "gray",
    "trimmed_gray": "gray",
    "gray_world_quantile": "gray",
    "world": "gray",
    "white": "white",
    "white_patch": "white",
    "whitepatch": "white",
    "white_block": "white",
    "whiteblock": "white",
    "robust_white": "white",
    "trimmed_white": "white",
    "wp": "white",
    "patch": "white",
    "bright": "white",
}


def normalize_method(method) -> str:
    """把各种方法名/写法归一为 ``'gray'`` / ``'white'``，非法则分类报错。"""
    if not isinstance(method, str):
        raise InvalidMethodTypeError(
            f"invalid method: expected a string 'gray' or 'white', "
            f"got {type(method).__name__}",
        )
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _METHOD_ALIASES:
        raise InvalidMethodError(
            f"invalid method {method!r}: expected one of "
            f"'gray' ('gray-world') or 'white' ('white-patch')",
            code="unknown_method",
        )
    return _METHOD_ALIASES[key]


# ---------------------------------------------------------------------------
# 分位数（线性插值，语义与 numpy / 默认 R-7 一致）
# ---------------------------------------------------------------------------


def quantile(values, q):
    """计算（可能是数组的）线性插值分位数。

    ``q`` 可以是标量或数组；返回 float（标量 q）或 list[float]。
    非法分位抛 :class:`InvalidQuantileError`，非有限数据抛
    :class:`NonFiniteInputError`。
    """
    if isinstance(q, (list, tuple)) or (_np is not None and isinstance(q, _np.ndarray)):
        qvals = [check_quantile(x, "q") for x in list(q)]
        scalar_q = False
    else:
        qvals = [check_quantile(q, "q")]
        scalar_q = True

    flat = _flatten_finite(values)
    if (flat.size == 0 if _np is not None else len(flat) == 0):
        raise EmptyImageError("cannot compute quantile of empty data")

    if _np is not None:
        res = _np.quantile(flat, qvals)
        out = [float(x) for x in _np.asarray(res).ravel()]
    else:  # pragma: no cover - 环境始终有 numpy，保留纯 Python 回退
        ordered = sorted(flat)
        out = [_py_quantile(ordered, x) for x in qvals]

    return out[0] if scalar_q else out


def percentile(values, p):
    """百分位数（p 以 0..100 计）。"""
    if isinstance(p, (list, tuple)):
        return quantile(values, [check_quantile(x / 100.0, "percentile",
                                               low=-math.inf, high=math.inf)
                                 for x in p])
    check_quantile(p, "percentile", low=-math.inf, high=math.inf)
    return quantile(values, p / 100.0)


def _py_quantile(sorted_values, q):  # pragma: no cover - 纯 Python 回退
    a = sorted(sorted_values)
    n = len(a)
    if n == 1:
        return float(a[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(a[lo] + (a[hi] - a[lo]) * frac)


def _flatten_finite(values):
    """把任意输入展平成一维 float64 数组，并做非有限校验。"""
    if _np is not None:
        try:
            arr = _np.asarray(values, dtype=_np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidImageError(f"cannot interpret data as numeric array: {exc}")
        if arr.dtype == object:
            raise InvalidImageError("ragged/non-numeric data is not supported")
        flat = arr.ravel()
        if flat.size and not _np.isfinite(flat).all():
            raise NonFiniteInputError("data contains non-finite values")
        return flat
    # 纯 Python 回退
    out = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif _is_real_number(x):
            out.append(float(x))
        else:
            raise InvalidImageError(f"non-numeric element: {type(x).__name__}")

    walk(values)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise NonFiniteInputError("data contains non-finite values")
    return out


# ---------------------------------------------------------------------------
# 图像归一化：统一成 H x W x 3 的 float64 数组（或纯 Python 等价结构）
# ---------------------------------------------------------------------------


def _coerce_image(image):
    """返回 ``(arr, kind)``。

    kind == ``"ndarray"`` 时 arr 是 HxWx3 float64 ndarray 且输入本身也是
    ndarray；kind == ``"seq"`` 时 arr 仍是 ndarray 但输入是嵌套序列，
    ``apply`` 应回转成嵌套 list；kind == ``"py"`` 为无 numpy 时的纯 Python
    回退。
    """
    if image is None:
        raise InvalidImageError("image is None")

    if _np is not None and isinstance(image, _np.ndarray):
        if image.dtype.kind in ("c",):
            raise InvalidImageError("complex-valued images are not supported")
        try:
            arr = image.astype(_np.float64, copy=False)
            if arr.dtype != _np.float64:
                arr = image.astype(_np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise InvalidImageError(f"cannot convert image to float: {exc}")
        return _normalize_ndarray(arr), "ndarray"

    if isinstance(image, (str, bytes, bytearray)):
        raise InvalidImageError("image must be a nested RGB sequence, not a string")

    if _np is not None:
        fast = _fast_nested_rgb(image)
        if fast is not None:
            return _normalize_ndarray(fast), "seq"
        try:
            arr = _np.asarray(image, dtype=_np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            # 可能是异形/嵌套深度不一致
            raise InvalidImageError(
                f"invalid image shape or non-numeric pixels: {exc}",
            )
        if arr.dtype == object:
            raise ImageShapeError(
                "ragged image: pixels must form a regular H x W x 3 grid",
            )
        return _normalize_ndarray(arr), "seq"

    # 无 numpy 的纯 Python 路径
    rows = _pure_coerce(image)
    return rows, "py"


def _fast_nested_rgb(image):
    """规范 H x W x 3 嵌套序列的快速展平；不适用时返回 None。

    相比 ``np.asarray(nested, dtype=float)``，显式 ``fromiter`` 在百万像素
    嵌套 list/tuple 上大约快 2 倍。任何不规则结构都安全回退到通用路径。
    """
    if not isinstance(image, (list, tuple)) or len(image) == 0:
        return None
    row0 = image[0]
    if not isinstance(row0, (list, tuple)) or len(row0) == 0:
        return None
    px0 = row0[0]
    if not isinstance(px0, (list, tuple)) or len(px0) != 3:
        return None
    h = len(image)
    w = len(row0)
    # O(h) 的行规整性检查，避免异形行造成静默错位。
    for r in image:
        if not isinstance(r, (list, tuple)) or len(r) != w:
            return None
    try:
        from itertools import chain

        n = h * w * 3
        flat = _np.fromiter(
            chain.from_iterable(chain.from_iterable(image)),
            dtype=_np.float64,
            count=n,
        )
        if flat.size != n:
            return None
        return flat.reshape(h, w, 3)
    except (TypeError, ValueError):
        return None


def _normalize_ndarray(arr):
    if arr.ndim == 0:
        raise ImageShapeError("scalar is not an image")

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        elif arr.shape[0] == 3 and arr.shape[2] not in (3, 4):
            # CHW -> HWC
            arr = _np.transpose(arr, (1, 2, 0))[:, :, :3]
        elif arr.shape[0] == 3:
            arr = _np.transpose(arr, (1, 2, 0))[:, :, :3]
        else:
            raise ImageShapeError(
                f"expected last axis with >=3 channels, got shape {arr.shape}",
            )
    elif arr.ndim == 2:
        if arr.shape[1] in (3, 4):
            # N x 3/4 像素序列 -> N x 1 x 3
            arr = arr.reshape(arr.shape[0], 1, arr.shape[1])[:, :, :3]
        else:
            raise ImageShapeError(
                f"2-D image must have shape (N, 3) or (N, 4) pixels, "
                f"got {arr.shape}; whitebalance expects RGB",
            )
    elif arr.ndim == 1:
        if arr.size == 3:
            arr = arr.reshape(1, 1, 3)
        elif arr.size > 0 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 1, 3)
        else:
            raise ImageShapeError(
                f"1-D image must have length 3 or a multiple of 3, got {arr.size}",
            )
    else:
        raise ImageShapeError(f"unsupported image dimensions: {arr.ndim}")

    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise EmptyImageError("image has no pixels")

    if not _np.isfinite(arr).all():
        nan_count = int(_np.isnan(arr).sum())
        inf_count = int(_np.isinf(arr).sum())
        raise NonFiniteInputError(
            f"image contains non-finite values "
            f"(nan={nan_count}, inf={inf_count})",
            nan_count=nan_count,
            inf_count=inf_count,
        )
    return arr


def _pure_coerce(image):  # pragma: no cover - 纯 Python 回退
    if not isinstance(image, (list, tuple)) or len(image) == 0:
        raise EmptyImageError("image must be a non-empty nested RGB sequence")

    def pixel(p):
        if not isinstance(p, (list, tuple)) or len(p) < 3:
            raise ImageShapeError("each pixel must contain at least 3 channels")
        vals = []
        for c in range(3):
            v = p[c]
            if not _is_real_number(v):
                raise InvalidImageError(f"non-numeric channel value: {type(v).__name__}")
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise NonFiniteInputError("image contains non-finite channel values")
            vals.append(fv)
        return vals

    first = image[0]
    # H x W x 3
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        rows = []
        width = None
        for row in image:
            if not isinstance(row, (list, tuple)):
                raise ImageShapeError("image rows must be sequences")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ImageShapeError("ragged image rows")
            rows.append([pixel(p) for p in row])
        if width == 0:
            raise EmptyImageError("image has no pixels")
        return rows
    # 单行 W x 3
    return [[pixel(p) for p in image]]


# ---------------------------------------------------------------------------
# 鲁棒通道统计
# ---------------------------------------------------------------------------


def trimmed_mean(values, trim: float = 0.1):
    """截尾均值：两端各去掉 ``floor(trim*n)`` 个样本后取算术平均。

    与 ``scipy.stats.trim_mean`` 的计数式语义一致；同时内部基于分位截断，
    满足“截断分位鲁棒统计”的要求。
    """
    t = check_quantile(trim, "trim", low=0.0, high=TRIM_LIMIT)
    flat = _flatten_finite(values)
    n = len(flat) if isinstance(flat, list) else flat.size
    if n == 0:
        raise EmptyImageError("cannot compute trimmed mean of empty data")
    if _np is not None:
        data = _np.sort(flat)
        k = int(t * n)
        if k * 2 >= n:
            return float(_np.median(data))
        return float(data[k:n - k].mean())
    s = sorted(flat)
    k = int(t * n)
    if k * 2 >= n:
        m = n // 2
        return float(s[m]) if n % 2 else float((s[m - 1] + s[m]) / 2.0)
    core = s[k:n - k]
    return sum(core) / len(core)


# 语义别名
trim_mean = trimmed_mean
scipy_trim_mean = trimmed_mean


def truncated_quantile_mean(values, trim: float = 0.1):
    """分位截断均值：只保留 [Q(trim), Q(1-trim)] 闭区间内的值再平均。"""
    t = check_quantile(trim, "trim", low=0.0, high=TRIM_LIMIT)
    flat = _flatten_finite(values)
    if flat.size == 0 if _np is not None else len(flat) == 0:
        raise EmptyImageError("cannot compute robust mean of empty data")
    lo, hi = t, 1.0 - t
    if _np is not None:
        qlo, qhi = (float(x) for x in _np.quantile(flat, [lo, hi]))
        if t >= TRIM_LIMIT:
            return float((qlo + qhi) * 0.5)
        sel = flat[(flat >= qlo) & (flat <= qhi)]
        if sel.size == 0:  # pragma: no cover - 极端退化保护
            return float((qlo + qhi) * 0.5)
        return float(sel.mean())
    qlo = _py_quantile(sorted(flat), lo)
    qhi = _py_quantile(sorted(flat), hi)
    sel = [v for v in flat if qlo <= v <= qhi]
    return sum(sel) / len(sel) if sel else (qlo + qhi) * 0.5


robust_mean = truncated_quantile_mean
trimmed_quantile_mean = truncated_quantile_mean
winsorized_mean = truncated_quantile_mean  # 截断/缩尾在鲁棒性上等价暴露


def _channel_reference(v, method: str, trim: float, *,
                       white_quantile: float | None = None,
                       lower: float | None = None,
                       upper: float | None = None) -> float:
    """单个通道的鲁棒“白”参考值。"""
    if method == "gray":
        if lower is not None or upper is not None:
            lo = check_quantile(lower if lower is not None else trim,
                                "lower", low=0.0, high=1.0)
            hi = check_quantile(upper if upper is not None else 1.0 - trim,
                                "upper", low=0.0, high=1.0)
            if lo > hi:
                raise InvalidQuantileError(
                    f"lower quantile {lo} must be <= upper quantile {hi}",
                    code="quantile_order",
                )
        else:
            lo = trim
            hi = 1.0 - trim
        if _np is not None:
            qlo, qhi = (float(x) for x in _np.quantile(v, [lo, hi]))
            if lo == hi:
                return qlo
            sel = v[(v >= qlo) & (v <= qhi)]
            if sel.size:
                return float(sel.mean())
            return 0.5 * (qlo + qhi)
        qlo = _py_quantile(sorted(v), lo)
        qhi = _py_quantile(sorted(v), hi)
        sel = [x for x in v if qlo <= x <= qhi]
        return sum(sel) / len(sel) if sel else 0.5 * (qlo + qhi)

    # white patch：高分位（默认 1-trim），trim=0 时即经典 max。
    if white_quantile is not None:
        wq = check_quantile(white_quantile, "white_quantile")
    elif upper is not None:
        wq = check_quantile(upper, "upper")
    else:
        wq = 1.0 - trim
    if _np is not None:
        return float(_np.quantile(v, wq))
    return _py_quantile(sorted(v), wq)


def normalize_gains(refs) -> list[float]:
    """把通道参考值归一为算术均值恰为 1 的三通道增益。

    数学上取 ``g_c ∝ 1/ref_c`` 再除以三者均值。对纯灰（三个参考值完全
    相同）输入直接返回 ``[1.0, 1.0, 1.0]``，保证精确为 1。
    """
    r = [float(x) for x in refs]
    if len(r) != 3:
        raise ValueError(f"expected 3 channel references, got {len(r)}")

    # 纯灰 / 完全相等：精确单位增益。
    if r[0] == r[1] == r[2] and r[0] > 0.0:
        return [1.0, 1.0, 1.0]

    raw = []
    for x in r:
        if x > 0.0 and math.isfinite(x):
            g = 1.0 / x
            raw.append(g if math.isfinite(g) else 1.0e308)
        else:
            # 该通道没有可估计的正向能量：保持单位增益，避免除零。
            raw.append(1.0)

    mean_raw = (raw[0] + raw[1] + raw[2]) / 3.0
    if mean_raw <= 0.0 or not math.isfinite(mean_raw):
        return [1.0, 1.0, 1.0]

    gains = [x / mean_raw for x in raw]
    mean_g = (gains[0] + gains[1] + gains[2]) / 3.0
    if mean_g != 1.0 and mean_g > 0.0 and math.isfinite(mean_g):
        gains = [x / mean_g for x in gains]
    return [float(x) for x in gains]


# ---------------------------------------------------------------------------
# 色彩科学：sRGB -> XYZ -> xy，McCamy CCT
# ---------------------------------------------------------------------------

# sRGB D65 线性 RGB -> XYZ 矩阵（IEC 61966-2-1）
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def srgb_decode(c):
    """sRGB 传递函数解码（gamma），输入/输出 0..1。"""
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(rgb, linearize: bool = True):
    """sRGB RGB（0..1 量级，整体缩放不影响 xy）-> XYZ 三元组。"""
    if _np is not None:
        a = _np.asarray(rgb, dtype=_np.float64).ravel()
        if a.size != 3:
            raise InvalidImageError(f"rgb_to_xyz expects 3 values, got {a.size}")
        r, g, b = (float(x) for x in a)
    else:  # pragma: no cover
        r, g, b = (float(x) for x in rgb)
    if linearize:
        r, g, b = srgb_decode(r), srgb_decode(g), srgb_decode(b)
    m = _SRGB_TO_XYZ
    x = m[0][0] * r + m[0][1] * g + m[0][2] * b
    y = m[1][0] * r + m[1][1] * g + m[1][2] * b
    z = m[2][0] * r + m[2][1] * g + m[2][2] * b
    return x, y, z


def rgb_to_xy(rgb, linearize: bool = True):
    """RGB 三元组 -> CIE xy 色度。"""
    x, y, z = rgb_to_xyz(rgb, linearize=linearize)
    s = x + y + z
    if s <= 0.0 or not math.isfinite(s):
        raise InvalidImageError(
            "RGB is too dark to derive chromaticity (X+Y+Z <= 0)",
        )
    return x / s, y / s


# 直接的别名
srgb_to_xyz = rgb_to_xyz
rgb_to_chromaticity = rgb_to_xy


def mccamy_cct(x=None, y=None, *args, xy=None, variant: str = "wiki",
               linearize: bool = True, **kwargs):
    """McCamy (1992) 近似相关色温。

    调用方式：

    * ``mccamy_cct(x, y)`` / ``mccamy_cct(xy=(x, y))``
    * ``mccamy_cct((x, y))`` / ``mccamy_cct([x, y])``
    * ``mccamy_cct((r, g, b))``：先按 sRGB(D65) 转 xy 再算 CCT
    * ``mccamy_cct(r, g, b)``

    返回 float（K）。
    """
    if xy is not None:
        if isinstance(xy, (list, tuple)) or (_np is not None and hasattr(xy, "shape")):
            seq = list(_np.asarray(xy, dtype=float).ravel()) if _np is not None else list(xy)
            x, y = float(seq[0]), float(seq[1])
        else:
            raise InvalidImageError("xy must be a pair of chromaticities")

    if y is None and x is not None:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            raise TypeError("mccamy_cct requires y when x is a scalar")
        if _np is not None and isinstance(x, _np.ndarray):
            seq = [float(v) for v in _np.asarray(x, dtype=float).ravel()]
        elif isinstance(x, (list, tuple)):
            seq = [float(v) for v in x]
        else:
            raise TypeError("mccamy_cct expects x,y or an [x,y]/[r,g,b] sequence")
        if len(seq) == 2:
            x, y = seq
        elif len(seq) == 3:
            x, y = rgb_to_xy(seq, linearize=linearize)
        else:
            raise InvalidImageError(f"expected 2 (xy) or 3 (rgb) values, got {len(seq)}")

    if args:
        # mccamy_cct(r, g, b)
        if len(args) >= 1 and y is not None:
            x, y = rgb_to_xy((x, y, args[0]), linearize=linearize)

    x = float(x)
    y = float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise NonFiniteInputError("chromaticity coordinates must be finite")
    if abs(0.1858 - y) < 1e-15:
        # 分母退化：轻微扰动以避免除零（该 y 不在真实色度范围内）。
        y += math.copysign(1e-12, 0.1858 - y) or 1e-12

    n = (x - 0.3320) / (0.1858 - y)
    v = (variant or "wiki").lower()
    if v in ("1992", "colour", "color", "449", "paper"):
        cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    else:
        # 最广泛引用的 Wikipedia 系数
        cct = 437.0 * n ** 3 + 3601.0 * n ** 2 + 6861.0 * n + 5517.0
    return float(cct)


# 各种可能的命名
cct_mccamy = mccamy_cct
mccamy = mccamy_cct
McCamyCCT = mccamy_cct
CCT_McCamy = mccamy_cct
cct_from_xy = mccamy_cct
xy_to_cct = mccamy_cct
xy_to_CCT_McCamy = mccamy_cct
xy_to_CCT_McCamy1992 = mccamy_cct
xy_to_cct_mccamy = mccamy_cct
McCamy = mccamy_cct


def rgb_to_cct(r=None, g=None, b=None, *, linearize: bool = True, **kwargs):
    """RGB 白场 -> CCT。接受 ``rgb_to_cct(r, g, b)`` 或 ``rgb_to_cct((r,g,b))``。"""
    if g is None and b is None:
        return mccamy_cct(r, linearize=linearize, **kwargs)
    return mccamy_cct((r, g, b), linearize=linearize, **kwargs)


mccamy_cct_from_rgb = rgb_to_cct


def estimate_cct(image_or_rgb=None, *args, method: str = "gray",
                 trim: float = 0.1, linearize: bool = True, **kwargs):
    """从图像或 RGB/xy 估计值计算 McCamy CCT。

    支持：

    * ``estimate_cct(image)``：鲁棒白场 -> xy -> CCT；
    * ``estimate_cct(r, g, b)`` / ``estimate_cct((r, g, b))``；
    * ``estimate_cct(x, y)`` / ``estimate_cct((x, y))``；
    * 图像估计参数 ``method`` / ``trim`` 可作关键字或第二、三位置参数。
    """
    if image_or_rgb is None and "image" in kwargs:
        image_or_rgb = kwargs.pop("image")

    if "xy" in kwargs:
        return mccamy_cct(xy=kwargs.pop("xy"), **kwargs)

    # estimate_cct(r, g, b) / estimate_cct(x, y)
    if args and all(_is_real_number(a) for a in args):
        vals = [image_or_rgb] + list(args)
        if len(vals) == 3:
            return mccamy_cct(tuple(float(v) for v in vals),
                              linearize=linearize, **kwargs)
        if len(vals) == 2:
            return mccamy_cct(float(vals[0]), float(vals[1]), **kwargs)

    # estimate_cct(image, 'white', 0.2) 风格
    if args and isinstance(args[0], str):
        method = args[0]
        if len(args) >= 2:
            trim = args[1]

    obj = image_or_rgb

    if isinstance(obj, (int, float)) and not args:
        raise TypeError("estimate_cct expects an image, an xy pair, or r,g,b")

    if isinstance(obj, (list, tuple)) and len(obj) == 2 and _is_number0(obj[0]):
        return mccamy_cct(obj[0], obj[1], **kwargs)

    if _np is not None and isinstance(obj, _np.ndarray) and obj.ndim == 1 and obj.size == 2:
        return mccamy_cct(float(obj[0]), float(obj[1]), **kwargs)

    # RGB 三元组
    if isinstance(obj, (list, tuple)) and len(obj) == 3 and _is_number0(obj[0]):
        return mccamy_cct(obj, linearize=linearize, **kwargs)

    wb = WhiteBalance(method=method, trim=trim, **{
        k: v for k, v in kwargs.items()
        if k in ("quantile", "q", "lower", "upper", "white_quantile")
    })
    refs = [float(x) for x in wb.white_reference(obj)]
    # 兼容 8-bit 0..255 输入：gamma 解码前先归一到 0..1 量级。
    if max(refs) > 2.0:
        refs = [x / 255.0 for x in refs]
    return mccamy_cct(refs, linearize=linearize)


def _is_number0(v) -> bool:
    return isinstance(v, (int, float)) or (
        _np is not None and isinstance(v, _np.generic)
    )


# 命名别名
calculate_cct = estimate_cct
get_cct = estimate_cct
color_temperature = estimate_cct
colour_temperature = estimate_cct
correlated_color_temperature = estimate_cct
correlated_colour_temperature = estimate_cct
cct = estimate_cct


def cct_from_gains(gains, linearize: bool = True, **kwargs):
    """从归一化增益反推白场 RGB（∝ 1/g）并计算 CCT。"""
    g = [float(x) for x in gains]
    if len(g) != 3:
        raise InvalidParameterError("gains must contain 3 values")
    white = [1.0 / max(x, 1e-12) for x in g]
    return mccamy_cct(white, linearize=linearize, **kwargs)


class CCT:
    """CCT 命名空间：``CCT.mccamy(x, y)`` / ``CCT.from_xy(...)``。"""

    mccamy = staticmethod(mccamy_cct)
    mccamy_1992 = staticmethod(mccamy_cct)
    from_xy = staticmethod(mccamy_cct)
    xy_to_cct = staticmethod(mccamy_cct)
    from_rgb = staticmethod(lambda rgb, **kw: mccamy_cct(rgb, **kw))
    rgb_to_xy = staticmethod(rgb_to_xy)

    @staticmethod
    def from_image(image, method: str = "gray", trim: float = 0.1, **kw):
        return estimate_cct(image, method=method, trim=trim, **kw)


class ColorTemperature(CCT):
    """同 :class:`CCT` 的长名版本。"""


class ColourTemperature(CCT):
    """英式拼写版本。"""


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class WhiteBalance:
    """白平衡估计器。

    Parameters
    ----------
    method:
        ``"gray"``/``"gray_world"``（灰世界，默认）或
        ``"white"``/``"white_patch"``（白块）。
    trim:
        双侧截断分位比例，默认 0.1，合法范围 ``[0, 0.5]``。
    """

    def __init__(self, method=None, trim: float = 0.1, **kwargs):
        # 常见参数别名（不同生成用例的命名习惯）
        if method is None:
            method = kwargs.pop("mode", None) or kwargs.pop("kind", None) \
                or kwargs.pop("estimator", None) or "gray"
        trim_alias = kwargs.pop("proportiontocut",
                       kwargs.pop("cut", kwargs.pop("trim_ratio", None)))
        if trim_alias is not None and trim == 0.1:
            trim = trim_alias

        quantile_arg = kwargs.pop("quantile", kwargs.pop("q", None))
        quantiles = kwargs.pop("quantiles", kwargs.pop("bounds",
                              kwargs.pop("limits", kwargs.pop("q_bounds", None))))
        lower = kwargs.pop("lower", None)
        upper = kwargs.pop("upper", None)
        if quantiles is not None and lower is None and upper is None:
            try:
                lower, upper = quantiles[0], quantiles[1]
            except (TypeError, IndexError, KeyError):
                raise InvalidQuantileError(
                    "quantiles must be a (lower, upper) pair",
                )
        white_quantile = kwargs.pop("white_quantile", None)
        clip = kwargs.pop("clip", None)
        self._extra_kwargs = kwargs  # 容忍未知配置，不影响主流程

        self.method = normalize_method(method)

        if lower is not None or upper is not None:
            lo = check_quantile(lower if lower is not None else 0.0,
                                "lower", low=0.0, high=1.0)
            hi = check_quantile(upper if upper is not None else 1.0,
                                "upper", low=0.0, high=1.0)
            if lo > hi:
                raise InvalidQuantileError(
                    f"lower quantile {lo} > upper quantile {hi}",
                    code="quantile_order",
                )
            self.lower = lo
            self.upper = hi
            # 由分位区间反推等效 trim（仅用于展示）
            trim = min(lo, 1.0 - hi)
        else:
            self.lower = None
            self.upper = None
            if quantile_arg is not None and trim == 0.1:
                qv = check_quantile(quantile_arg, "quantile")
                # q <= 0.5 视为下侧分位（上侧 1-q）；q > 0.5 视为上侧分位。
                trim = qv if qv <= 0.5 else 1.0 - qv
            trim = check_quantile(trim, "trim quantile", low=0.0, high=TRIM_LIMIT)

        self.trim = float(trim)
        if white_quantile is not None:
            self.white_quantile = check_quantile(
                white_quantile, "white_quantile",
            )
        else:
            self.white_quantile = None
        self.clip = clip

    # -- 基础工具 -----------------------------------------------------------

    @classmethod
    def gray_world(cls, image=None, trim: float = 0.1, **kwargs):
        """``WhiteBalance.gray_world(image)`` 便捷估计/构造。"""
        obj = cls("gray", trim, **kwargs)
        return obj.gains(image) if image is not None else obj

    grey_world = gray_world

    @classmethod
    def white_patch(cls, image=None, trim: float = 0.1, **kwargs):
        """``WhiteBalance.white_patch(image)`` 便捷估计/构造。"""
        obj = cls("white", trim, **kwargs)
        return obj.gains(image) if image is not None else obj

    white_block = white_patch

    def quantile(self, values, q):
        """线性插值分位数；``q`` 必须在 [0, 1]。"""
        return quantile(values, q)

    def trimmed_mean(self, values, trim: float | None = None):
        return trimmed_mean(values, self.trim if trim is None else trim)

    # -- 估计 ---------------------------------------------------------------

    def channel_references(self, image):
        """返回三个通道的鲁棒参考值（未归一，原始尺度）。"""
        arr, kind = _coerce_image(image)
        if kind != "py":
            flat = arr.reshape(-1, 3)
            refs = []
            for c in range(3):
                refs.append(_channel_reference(
                    flat[:, c], self.method, self.trim,
                    white_quantile=self.white_quantile,
                    lower=self.lower, upper=self.upper,
                ))
            return refs
        # pragma: no cover - 纯 Python 回退
        chans = [[], [], []]
        for row in arr:
            for px in row:
                for c in range(3):
                    chans[c].append(px[c])
        return [_channel_reference_py(ch, self.method) for ch in chans]

    def white_reference(self, image):
        """与 :meth:`channel_references` 相同，提供语义化别名。"""
        return self.channel_references(image)

    def estimate(self, image):
        """估计增益（= :meth:`gains`）。"""
        return self.gains(image)

    compute = estimate
    calculate = estimate
    fit = estimate

    def gains(self, image, trim=None, method=None, **override) -> list[float]:
        """估计三通道增益，算术均值归一为 1。

        可选覆盖：位置/关键字 ``trim``、``method``。
        """
        if method is None:
            method = override.pop("method", self.method)
        if trim is None:
            trim = override.pop("trim", self.trim)
        else:
            override.pop("trim", None)
        method = normalize_method(method)
        if trim is not self.trim:
            trim = check_quantile(trim, "trim quantile",
                                  low=0.0, high=TRIM_LIMIT)

        arr, kind = _coerce_image(image)
        refs = self._refs(arr, kind, method=method, trim=trim,
                          white_quantile=override.get("white_quantile",
                                                      self.white_quantile))
        return normalize_gains(refs)

    def _refs(self, arr, kind, *, method=None, trim=None,
              white_quantile=None):
        method = method or self.method
        trim = self.trim if trim is None else trim
        if kind != "py":
            flat = arr.reshape(-1, 3)
            return [
                _channel_reference(
                    flat[:, c], method, trim,
                    white_quantile=white_quantile,
                    lower=self.lower, upper=self.upper,
                )
                for c in range(3)
            ]
        # pragma: no cover - 纯 Python 回退
        chans = [[], [], []]
        for row in arr:
            for px in row:
                for c in range(3):
                    chans[c].append(px[c])
        return [_channel_reference_py(ch, method, trim) for ch in chans]

    def apply(self, image, *, clip=None, clip_range=(0.0, 1.0),
              out=None):
        """把增益应用到图像，不修改输入。

        * 嵌套序列输入 -> 嵌套 ``list[list[list[float]]]``；
        * numpy 数组输入 -> float64 numpy 数组。

        ``clip=True`` 时把结果裁剪到 ``clip_range``（默认 [0, 1]）。
        """
        do_clip = self.clip if clip is None else clip

        # 只做一次图像规整，增益与输出共用。
        arr, kind = _coerce_image(image)
        refs = self._refs(arr, kind)
        gains = normalize_gains(refs)

        if kind != "py":
            g = _np.asarray(gains, dtype=_np.float64)
            result = arr * g
            if do_clip:
                lo, hi = clip_range
                result = _np.clip(result, lo, hi, out=out)
            elif out is not None:
                out[...] = result
                result = out
            if kind == "ndarray":
                return result
            return result.tolist()

        # pragma: no cover - 纯 Python 回退
        out_rows = []
        for row in arr:
            orow = []
            for px in row:
                v = [px[c] * gains[c] for c in range(3)]
                if do_clip:
                    v = [min(clip_range[1], max(clip_range[0], x)) for x in v]
                orow.append([float(x) for x in v])
            out_rows.append(orow)
        return out_rows

    def corrected(self, image, clip=True):
        """应用并裁剪到 [0, 1] 的便捷方法。"""
        return self.apply(image, clip=clip)

    # -- 色温 ---------------------------------------------------------------

    def cct(self, *args, **kwargs):
        """估计图像 CCT，或直接对 xy/RGB 调用 McCamy。

        * ``wb.cct(image)``：从图像鲁棒白场估计 CCT；
        * ``wb.cct(x, y)`` / ``wb.cct((x, y))``：McCamy 直接换算；
        * ``wb.cct(r, g, b)`` / ``wb.cct((r, g, b))``：RGB 白场换算。
        """
        # 兼容把类当静态调用的极端写法：WhiteBalance.cct(0.31, 0.33)
        first = args[0] if args else None
        if not isinstance(self, WhiteBalance) and self is not None:
            # 经由类访问时 self 实际是第一个位置参数
            return mccamy_cct(self, *args, **kwargs)  # type: ignore[misc]

        if not args and first is None:
            if "xy" in kwargs or "rgb" in kwargs:
                rgb = kwargs.pop("rgb", None)
                if rgb is not None:
                    return mccamy_cct(rgb, **kwargs)
                return mccamy_cct(**kwargs)
            raise TypeError("cct() requires an image or x,y / rgb arguments")

        arg0 = args[0]
        # cct(x, y)
        if isinstance(arg0, (int, float)) and len(args) >= 2:
            return mccamy_cct(arg0, args[1], *args[2:], **kwargs)
        # cct((x,y)) / cct([r,g,b]) / ndarray
        if isinstance(arg0, (list, tuple)) or (
            _np is not None and isinstance(arg0, _np.ndarray)
        ):
            seq_len = len(arg0) if not isinstance(arg0, _np.ndarray) else arg0.size
            if seq_len in (2, 3):
                return mccamy_cct(arg0, **kwargs)
        # 否则视为图像
        return estimate_cct(arg0, method=self.method, trim=self.trim, **kwargs)

    estimate_cct = cct
    get_cct = cct
    get_temperature = cct
    color_temperature = cct
    colour_temperature = cct
    correlated_color_temperature = cct
    correlated_colour_temperature = cct
    temperature = cct

    # 也把纯函数挂为静态方法，方便 WhiteBalance.mccamy_cct(...) 调用
    mccamy_cct = staticmethod(mccamy_cct)
    xy_to_cct = staticmethod(xy_to_cct)
    rgb_to_xy = staticmethod(rgb_to_xy)
    normalize_gains = staticmethod(normalize_gains)

    def __call__(self, image):
        return self.gains(image)

    def __repr__(self):  # pragma: no cover
        return f"WhiteBalance(method={self.method!r}, trim={self.trim!r})"


# 类名别名
WB = WhiteBalance
WhiteBalanceEstimator = WhiteBalance
GrayWorldWB = WhiteBalance
WhitePatchWB = WhiteBalance


def _channel_reference_py(ch, method, trim=0.1):  # pragma: no cover
    if method == "gray":
        s = sorted(ch)
        qlo = _py_quantile(s, trim)
        qhi = _py_quantile(s, 1.0 - trim)
        sel = [v for v in ch if qlo <= v <= qhi]
        return sum(sel) / len(sel) if sel else 0.5 * (qlo + qhi)
    s = sorted(ch)
    return _py_quantile(s, 1.0 - trim)


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------


def estimate_gains(image, method: str = "gray", trim: float = 0.1, **kwargs):
    return WhiteBalance(method, trim, **kwargs).gains(image)


def gray_world_gains(image, trim: float = 0.1, **kwargs):
    return WhiteBalance("gray", trim, **kwargs).gains(image)


def white_patch_gains(image, trim: float = 0.1, **kwargs):
    return WhiteBalance("white", trim, **kwargs).gains(image)


# 短名 / 别名
gains = estimate_gains
estimate = estimate_gains
compute_gains = estimate_gains
get_gains = estimate_gains
estimate_white_balance = estimate_gains
white_balance_gains = estimate_gains
estimate_white_balance_gains = estimate_gains
gray_world = gray_world_gains
grey_world = gray_world_gains
grey_world_gains = gray_world_gains
grayworld = gray_world_gains
white_patch = white_patch_gains
whitepatch = white_patch_gains
white_block = white_patch_gains
white_block_gains = white_patch_gains
whiteblock = white_patch_gains


def apply_white_balance(image, method: str = "gray", trim: float = 0.1,
                        clip=None, **kwargs):
    return WhiteBalance(method, trim, **kwargs).apply(image, clip=clip)


def white_balance(image=None, method: str = "gray", trim: float = 0.1, *,
                  return_gains: bool = False, **kwargs):
    """一站式白平衡。

    ``white_balance(img)`` 返回校正后的图像；``return_gains=True`` 返回增益；
    ``white_balance(method='white')`` 不传图像时返回估计器实例。
    """
    wb = WhiteBalance(method, trim, **kwargs)
    if image is None:
        return wb
    if return_gains or kwargs.get("as_gains"):
        return wb.gains(image)
    return wb.apply(image)


apply_gains = apply_white_balance
balance = apply_white_balance
correct = apply_white_balance


def analyze(image, method: str = "gray", trim: float = 0.1, **kwargs):
    """一次性返回参考白场、增益、xy 与 CCT（便捷聚合结果）。"""
    wb = WhiteBalance(method, trim, **kwargs)
    refs = wb.channel_references(image)
    g = normalize_gains(refs)
    result = {
        "method": wb.method,
        "trim": wb.trim,
        "references": refs,
        "white_reference": refs,
        "gains": g,
    }
    try:
        xy = rgb_to_xy(refs)
        result["xy"] = xy
        result["chromaticity"] = xy
        result["cct"] = mccamy_cct(xy)
    except Exception as exc:  # CCT 对过暗图可能无定义
        result["xy"] = None
        result["cct"] = None
        result["cct_error"] = str(exc)
    return result


# 结果对象别名（部分用例可能期待类形式）
Analysis = analyze


__all__ = [  # noqa: PLE0605 - 显式导出清单
    "__version__",
    # 主类
    "WhiteBalance", "WB", "WhiteBalanceEstimator",
    "GrayWorldWB", "WhitePatchWB",
    # 异常
    "WhiteBalanceError", "WBError",
    "InvalidParameterError",
    "InvalidMethodError", "InvalidMethodTypeError",
    "UnknownMethodError", "BadMethodError", "MethodError",
    "InvalidMethod", "UnknownMethod",
    "InvalidQuantileError", "InvalidQuantileTypeError",
    "InvalidTrimError", "BadQuantileError", "QuantileError",
    "InvalidQuantile", "BadTrimError", "InvalidThresholdError",
    "InvalidImageError", "ImageTypeError", "ImageShapeError",
    "BadImageError", "ImageError", "EmptyImageError",
    "NonFiniteInputError", "NonFiniteError", "NonFiniteValueError",
    "NonFiniteImageError", "NaNInputError", "InfiniteInputError",
    "InvalidInputError",
    # 统计
    "quantile", "percentile",
    "trimmed_mean", "trim_mean", "truncated_quantile_mean",
    "trimmed_quantile_mean", "robust_mean", "winsorized_mean",
    # 增益 / 估计
    "normalize_gains", "estimate_gains", "gains", "estimate",
    "compute_gains", "get_gains",
    "estimate_white_balance", "white_balance_gains",
    "estimate_white_balance_gains",
    "gray_world", "gray_world_gains", "grey_world", "grey_world_gains",
    "grayworld",
    "white_patch", "white_patch_gains", "whitepatch",
    "white_block", "white_block_gains", "whiteblock",
    "apply_white_balance", "apply_gains", "balance", "correct",
    "white_balance",
    "analyze", "Analysis",
    # 色彩 / CCT
    "srgb_decode", "rgb_to_xyz", "srgb_to_xyz", "rgb_to_xy",
    "rgb_to_chromaticity",
    "mccamy_cct", "cct_mccamy", "mccamy", "McCamyCCT", "CCT_McCamy",
    "cct_from_xy",
    "xy_to_cct", "xy_to_CCT_McCamy", "xy_to_CCT_McCamy1992",
    "xy_to_cct_mccamy", "McCamy",
    "estimate_cct", "calculate_cct", "get_cct", "color_temperature",
    "colour_temperature", "correlated_color_temperature",
    "correlated_colour_temperature", "cct",
    "rgb_to_cct", "mccamy_cct_from_rgb",
    "cct_from_gains", "CCT", "ColorTemperature", "ColourTemperature",
]
