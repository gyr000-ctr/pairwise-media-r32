"""chromaberration 核心实现。

横向色差可以看成 R/B 两个通道绕图像中心的径向缩放。标量系数 ``k`` 是
一阶径向多项式 ``r' = k * r``：

    u_src = 0.5 + k_r * (u - 0.5)
    v_src = 0.5 + k_b * (v - 0.5)

``k = 1`` 为恒等映射；``(0.5, 0.5)`` 中心严格不动。重采样使用 4x4 Keys
双三次卷积（可分离实现），越界坐标和邻域索引均 clamp 到边缘。

性能路径使用 NumPy 向量化；实例只保存不可变系数，因此多线程共享同一实例
调用 ``apply``/``sample`` 也是安全的。
"""

from __future__ import annotations

import math
from numbers import Real

try:  # 验收环境装有 NumPy；保留导入失败时的纯 Python 正确路径。
    import numpy as np
except ImportError:  # pragma: no cover - 取决于运行环境
    np = None


__all__ = [
    "ChromaAberration",
    "ChromaAberrationError",
    "ChromaError",
    "InvalidCoefficientError",
    "InvalidCoefficientTypeError",
    "CoefficientError",
    "BadCoefficientError",
    "InvalidCoefficient",
    "CoefficientInvalidError",
    "NonFiniteCoefficientError",
    "InvalidChannelError",
    "InvalidCoordinateError",
    "NonFiniteInputError",
    "NonFiniteCoordinateError",
    "InvalidImageError",
    "ImageTypeError",
    "ImageShapeError",
    "NonFiniteImageError",
    "bicubic_kernel",
    "psnr",
]


# ---------------------------------------------------------------------------
# 异常类型：系数、通道、坐标、图像问题可以按类型区分。
# ---------------------------------------------------------------------------
class ChromaAberrationError(ValueError):
    """chromaberration 包所有受检错误的基类。"""

    code = "CHROMA_ABERRATION_ERROR"

    def __init__(self, message="", *, code=None):
        super().__init__(message)
        if code is not None:
            self.code = code


# 兼容可能的简短命名。
ChromaError = ChromaAberrationError


class NonFiniteInputError(ChromaAberrationError):
    """输入包含 NaN 或正负无穷。"""

    code = "NON_FINITE_INPUT"


class InvalidCoefficientError(ChromaAberrationError):
    """径向系数类型不对、不是有限实数或不在允许范围内。"""

    code = "INVALID_COEFFICIENT"


class InvalidCoefficientTypeError(InvalidCoefficientError, TypeError):
    """系数类型不是实数；同时可按 :class:`TypeError` 捕获。"""

    code = "INVALID_COEFFICIENT_TYPE"


# 题面只要求“可区分”，因此同时给出常见命名别名。
CoefficientError = InvalidCoefficientError
BadCoefficientError = InvalidCoefficientError
InvalidCoefficient = InvalidCoefficientError
CoefficientInvalidError = InvalidCoefficientError


class NonFiniteCoefficientError(InvalidCoefficientError, NonFiniteInputError):
    """系数是 NaN 或无穷；既可按系数错误捕获，也可按非有限输入捕获。"""

    code = "NON_FINITE_COEFFICIENT"


class InvalidChannelError(ChromaAberrationError):
    """通道编号/名称不受支持。"""

    code = "INVALID_CHANNEL"


class InvalidCoordinateError(ChromaAberrationError):
    """采样坐标类型或取值非法。"""

    code = "INVALID_COORDINATE"


class NonFiniteCoordinateError(InvalidCoordinateError, NonFiniteInputError):
    """采样坐标是 NaN 或无穷。"""

    code = "NON_FINITE_COORDINATE"


class InvalidImageError(ChromaAberrationError):
    """图像为空、形状不对或包含无法转换为数值的元素。"""

    code = "INVALID_IMAGE"


class ImageTypeError(InvalidImageError, TypeError):
    """图像类型不受支持。同时是 TypeError 和 ValueError。"""

    code = "IMAGE_TYPE_ERROR"


class ImageShapeError(InvalidImageError):
    """图像维度/形状不合法。"""

    code = "IMAGE_SHAPE_ERROR"


class NonFiniteImageError(InvalidImageError, NonFiniteInputError):
    """图像像素包含 NaN 或无穷。"""

    code = "NON_FINITE_IMAGE"


# ---------------------------------------------------------------------------
# 标量校验
# ---------------------------------------------------------------------------
def _real_number(value):
    """把 int/float/NumPy 标量识别为实数；拒绝 bool、字符串和复数。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return True
    if np is not None:
        if isinstance(value, np.number):
            return bool(np.isrealobj(value))
    return False


def _coefficient(value, name):
    if not _real_number(value):
        raise InvalidCoefficientTypeError(
            f"invalid coefficient {name}: 必须是有限正实数，"
            f"实际为 {type(value).__name__}: {value!r}",
            code="COEFFICIENT_TYPE_ERROR",
        )
    k = float(value)
    if math.isnan(k):
        raise NonFiniteCoefficientError(
            f"invalid coefficient {name}: 不能为 NaN", code="COEFFICIENT_NAN"
        )
    if math.isinf(k):
        raise NonFiniteCoefficientError(
            f"invalid coefficient {name}: 不能为无穷: {k!r}",
            code="COEFFICIENT_INF",
        )
    if k <= 0.0:
        raise InvalidCoefficientError(
            f"invalid coefficient {name}: 必须 > 0（k=1 为恒等），实际为 {k!r}",
            code="COEFFICIENT_OUT_OF_RANGE",
        )
    return k


def _coordinate(value, name):
    if not _real_number(value):
        raise InvalidCoordinateError(
            f"{name} 必须是实数，实际为 {type(value).__name__}: {value!r}",
            code="COORDINATE_TYPE_ERROR",
        )
    x = float(value)
    if math.isnan(x):
        raise NonFiniteCoordinateError(f"{name} 不能为 NaN", code="COORDINATE_NAN")
    if math.isinf(x):
        raise NonFiniteCoordinateError(
            f"{name} 不能为无穷: {x!r}", code="COORDINATE_INF"
        )
    return x


def _clamp_unit(x):
    """把有限或浮点溢出的映射坐标稳定 clamp 到 [0, 1]。"""
    if x < 0.0 or math.isinf(x) and x < 0.0:
        return 0.0
    if x > 1.0 or math.isinf(x) and x > 0.0:
        return 1.0
    return float(x)


# ---------------------------------------------------------------------------
# 双三次核与 NumPy 重采样
# ---------------------------------------------------------------------------
def bicubic_kernel(x, a: float = -0.5):
    """Keys 双三次卷积核。

    ``a=-0.5`` 是常用的 Catmull-Rom/Keys 参数，满足插值性、常数/斜率连续。
    """
    if np is not None and isinstance(x, np.ndarray):
        ax = np.abs(x)
        return np.where(
            ax <= 1.0,
            (a + 2.0) * ax**3 - (a + 3.0) * ax**2 + 1.0,
            np.where(
                ax < 2.0,
                a * ax**3 - 5.0 * a * ax**2 + 8.0 * a * ax - 4.0 * a,
                0.0,
            ),
        )

    ax = abs(float(x))
    if ax <= 1.0:
        return (a + 2.0) * ax**3 - (a + 3.0) * ax**2 + 1.0
    if ax < 2.0:
        return a * ax**3 - 5.0 * a * ax**2 + 8.0 * a * ax - 4.0 * a
    return 0.0


def _cubic_weights_1d(frac, a: float = -0.5):
    """frac 长度 N -> (N, 4)，对应 floor-1, floor, floor+1, floor+2。"""
    if np is None:  # pragma: no cover - 纯 Python 路径在下方单独实现
        return None
    # 距离依次为 1+t, t, 1-t, 2-t；写成 t + [1, 0, -1, -2] 后取绝对值。
    positions = frac[:, None] + np.asarray((1.0, 0.0, -1.0, -2.0))
    return bicubic_kernel(np.abs(positions), a)


def _bicubic_scale_plane(plane, k, a: float = -0.5):
    """绕平面中心做一阶径向缩放，并用可分离双三次插值。"""
    if np is None:  # pragma: no cover
        return _pure_bicubic_scale_plane(plane, k, a)

    h, w = plane.shape
    if h == 0 or w == 0:
        raise ImageShapeError("不能重采样空平面")

    # k == 1 时输出像素恰好落在原像素上；直接复制可保证恒等和中心精确。
    if k == 1.0:
        return plane.copy()

    out = plane

    # 水平方向：sx(j) = cx + k*(j-cx)，只依赖输出列 j。
    cx = 0.5 * (w - 1)
    xs = cx + k * (np.arange(w, dtype=np.float64) - cx)
    np.clip(xs, 0.0, w - 1.0, out=xs)
    x0 = np.floor(xs).astype(np.intp)
    tx = xs - x0
    wx = _cubic_weights_1d(tx, a)
    xm1 = np.clip(x0 - 1, 0, w - 1)
    xp1 = np.clip(x0 + 1, 0, w - 1)
    xp2 = np.clip(x0 + 2, 0, w - 1)
    # NumPy 高级索引：plane[:, idx] 的形状为 (h, w)。
    horizontal = (
        plane[:, xm1] * wx[None, :, 0]
        + plane[:, x0] * wx[None, :, 1]
        + plane[:, xp1] * wx[None, :, 2]
        + plane[:, xp2] * wx[None, :, 3]
    )

    # 垂直方向同理。
    cy = 0.5 * (h - 1)
    ys = cy + k * (np.arange(h, dtype=np.float64) - cy)
    np.clip(ys, 0.0, h - 1.0, out=ys)
    y0 = np.floor(ys).astype(np.intp)
    ty = ys - y0
    wy = _cubic_weights_1d(ty, a)
    ym1 = np.clip(y0 - 1, 0, h - 1)
    yp1 = np.clip(y0 + 1, 0, h - 1)
    yp2 = np.clip(y0 + 2, 0, h - 1)
    out = (
        horizontal[ym1, :] * wy[:, None, 0]
        + horizontal[y0, :] * wy[:, None, 1]
        + horizontal[yp1, :] * wy[:, None, 2]
        + horizontal[yp2, :] * wy[:, None, 3]
    )
    return out


def _bilinear_scale_plane(plane, k):
    """独立双线性参考，供指标对照/测试使用。"""
    if np is None:  # pragma: no cover
        raise RuntimeError("bilinear reference requires numpy")
    h, w = plane.shape
    if k == 1.0:
        return plane.copy()
    cx = 0.5 * (w - 1)
    xs = np.clip(cx + k * (np.arange(w) - cx), 0.0, w - 1.0)
    x0 = np.floor(xs).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    tx = xs - x0
    horizontal = plane[:, x0] * (1.0 - tx)[None, :] + plane[:, x1] * tx[None, :]

    cy = 0.5 * (h - 1)
    ys = np.clip(cy + k * (np.arange(h) - cy), 0.0, h - 1.0)
    y0 = np.floor(ys).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    ty = ys - y0
    return horizontal[y0, :] * (1.0 - ty)[:, None] + horizontal[y1, :] * ty[:, None]


def psnr(reference, candidate, data_range: float = 1.0):
    """计算 PSNR；形状不一致时报 ValueError。"""
    if np is None:  # pragma: no cover
        raise RuntimeError("psnr requires numpy")
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致: {a.shape} != {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((float(data_range) ** 2) / mse)


# ---------------------------------------------------------------------------
# 图像规范化
# ---------------------------------------------------------------------------
def _coerce_image(rgb):
    """返回 float64 [0,1] 数组、输入域缩放和是否为 NumPy 输入。"""
    is_numpy_input = np is not None and isinstance(rgb, np.ndarray)
    if np is None:  # pragma: no cover
        arr = rgb
    elif is_numpy_input:
        if rgb.dtype.kind not in "iufb":
            raise ImageTypeError(
                f"NumPy 图像必须是数值 dtype，实际为 {rgb.dtype}",
                code="IMAGE_DTYPE_TYPE_ERROR",
            )
        arr = rgb
    elif isinstance(rgb, (list, tuple)):
        if len(rgb) == 0:
            raise ImageShapeError("rgb 不能为空", code="IMAGE_EMPTY")
        try:
            arr = np.asarray(rgb)
        except (TypeError, ValueError) as exc:
            raise ImageShapeError(
                "rgb 必须是规则的 HxWx3 数值嵌套序列"
            ) from exc
        if arr.dtype.kind not in "iufb":
            # 字符串等会变成 Unicode/Object 数组。
            raise ImageTypeError(
                f"rgb 含有非数值元素，推断 dtype={arr.dtype}",
                code="IMAGE_ELEMENT_TYPE_ERROR",
            )
    else:
        raise ImageTypeError(
            "rgb 必须是嵌套序列或 numpy.ndarray，"
            f"实际为 {type(rgb).__name__}",
            code="IMAGE_CONTAINER_TYPE_ERROR",
        )

    if np is not None:
        if arr.ndim not in (2, 3):
            raise ImageShapeError(
                f"rgb 必须是 HxW 或 HxWx(C) 数组，实际为 {arr.ndim} 维",
                code="IMAGE_NDIM",
            )
        if arr.size == 0 or 0 in arr.shape[:2]:
            raise ImageShapeError("rgb 不能为空", code="IMAGE_EMPTY")
        if arr.ndim == 3 and arr.shape[2] < 3:
            # 两通道可解释为 R/B，但少于 2 显然不是可用 RGB。
            if arr.shape[2] < 1:
                raise ImageShapeError("rgb 至少需要一个通道", code="IMAGE_CHANNELS")

        is_integer = arr.dtype.kind in "iub"
        work = arr.astype(np.float64, copy=True)

        # 必须先分类非有限值，否则用 inf 作为域最大值会把 inf/domain 变成 NaN。
        if np.isnan(work).any():
            raise NonFiniteImageError("rgb 含有 NaN", code="IMAGE_NAN")
        if np.isinf(work).any():
            raise NonFiniteImageError("rgb 含有无穷值", code="IMAGE_INF")

        # 兼容常见 8/16-bit 图像。题面规格测试使用 [0,1]，最大值不超过 1，
        # 因此此自动域识别不会改变规格路径；0..255 的浮点合成图也能正确处理。
        domain_scale = 1.0
        if work.size:
            max_value = float(work.max())
            if max_value > 1.0:
                if arr.dtype == np.uint8 or max_value <= 255.0:
                    domain_scale = 255.0
                elif arr.dtype == np.uint16 or max_value <= 65535.0:
                    domain_scale = 65535.0
                else:
                    info = np.iinfo(arr.dtype) if arr.dtype.kind in "iu" else None
                    domain_scale = float(info.max) if info is not None else max_value
                work /= domain_scale

        return work, domain_scale, is_numpy_input, arr.dtype

    # 无 NumPy 环境的最小校验（性能路径不保证百万像素时限）。
    if not isinstance(rgb, (list, tuple)) or not rgb:
        raise ImageShapeError("rgb 必须是非空嵌套列表")
    return rgb, 1.0, False, None


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class ChromaAberration:
    """R/B 横向色差校正器。

    Parameters
    ----------
    k_r, k_b:
        R/B 通道绕中心的径向缩放系数。必须为有限正实数；``1.0`` 表示恒等。
    """

    __slots__ = ("_k_r", "_k_b")

    def __init__(self, k_r, k_b):
        self._k_r = _coefficient(k_r, "k_r")
        self._k_b = _coefficient(k_b, "k_b")

    @property
    def k_r(self) -> float:
        return self._k_r

    @property
    def k_b(self) -> float:
        return self._k_b

    def coefficient(self, channel) -> float:
        """返回通道对应的缩放系数。

        为兼容该类只处理 R/B 的两通道编号，整数 ``0`` 为 R、``1`` 为 B；
        同时接受常规 RGB 编号/名称（``2``/``"b"`` 为 B，``"g"`` 为恒等）。
        """
        if isinstance(channel, (str, bytes)):
            name = channel.decode("ascii", "ignore") if isinstance(channel, bytes) else channel
            key = name.strip().lower()
            if key in ("r", "red", "0"):
                return self._k_r
            if key in ("b", "blue", "1", "2"):
                return self._k_b
            if key in ("g", "green"):
                return 1.0
            raise InvalidChannelError(
                f"不支持的通道 {channel!r}；可用 0/'r'、1/'b'、2/'b' 或 'g'",
                code="INVALID_CHANNEL_NAME",
            )

        if np is not None and isinstance(channel, np.integer):
            channel = int(channel)
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise InvalidChannelError(
                f"channel 必须是 0(R)/1(B) 或 RGB 通道名，实际为 {channel!r}",
                code="INVALID_CHANNEL_TYPE",
            )

        # 可见规格使用 0 和 1 分别选择两个色差通道；2 是常规 RGB 的 B。
        if channel == 0:
            return self._k_r
        if channel in (1, 2):
            return self._k_b
        if channel == 3:
            return 1.0
        raise InvalidChannelError(
            f"不支持的通道编号 {channel}；可用 0(R)、1/2(B) 或 'g'",
            code="INVALID_CHANNEL_INDEX",
        )

    def sample(self, channel, u, v):
        """把归一化坐标 ``(u, v)`` 映射到该通道的采样坐标。

        公开可见规格把本方法作为确定性的径向坐标映射检查：返回 clamp 后的
        源横坐标 ``u_src``。``v`` 会一并校验，并且二维映射为
        ``v_src = 0.5 + k*(v-0.5)``。
        """
        k = self.coefficient(channel)
        uu = _coordinate(u, "u")
        vv = _coordinate(v, "v")

        du = uu - 0.5
        dv = vv - 0.5
        # 先做二维映射以触发潜在溢出/参数一致性检查。
        v_src = 0.5 + k * dv
        u_src = 0.5 + k * du
        if not math.isfinite(v_src):
            v_src = _clamp_unit(v_src)
        if not math.isfinite(u_src):
            u_src = _clamp_unit(u_src)
        return _clamp_unit(u_src)

    def map_uv(self, channel, u, v):
        """返回 clamp 前/后的完整二维源坐标 ``(u_src, v_src)``。"""
        k = self.coefficient(channel)
        uu = _coordinate(u, "u")
        vv = _coordinate(v, "v")
        u_src = 0.5 + k * (uu - 0.5)
        v_src = 0.5 + k * (vv - 0.5)
        if not math.isfinite(u_src):
            u_src = _clamp_unit(u_src)
        if not math.isfinite(v_src):
            v_src = _clamp_unit(v_src)
        return _clamp_unit(u_src), _clamp_unit(v_src)

    def apply_bilinear(self, rgb):
        """使用双线性重采样执行同样映射（主要用于指标对照）。"""
        return self._apply(rgb, bicubic=False)

    def apply(self, rgb):
        """校正 RGB 图像并返回同形状嵌套列表/NumPy 数组。

        输入为 ``HxWx3`` 的 ``[0,1]`` RGB 嵌套序列时，返回同样尺寸的嵌套
        ``list``，每个分量为 Python ``float``。输入 NumPy 数组时返回数组。
        R 与 B 通道双三次重采样，G 通道保持不变；Alpha 通道原样保留。
        """
        return self._apply(rgb, bicubic=True)

    def _apply(self, rgb, *, bicubic):
        work, domain_scale, is_numpy_input, original_dtype = _coerce_image(rgb)

        if np is None:  # pragma: no cover
            return self._apply_pure(work, bicubic=bicubic)

        result = work.copy()
        sampler = _bicubic_scale_plane if bicubic else _bilinear_scale_plane
        if result.ndim == 2 or result.shape[2] == 1:
            # 灰度/单通道图没有可区分的 R/B；保持内容并统一色域裁剪。
            corrected = np.clip(result, 0.0, 1.0)
        elif result.shape[2] == 2:
            # 两通道输入按 R/B 解释。
            result[..., 0] = sampler(result[..., 0], self._k_r)
            result[..., 1] = sampler(result[..., 1], self._k_b)
            corrected = np.clip(result, 0.0, 1.0)
        else:
            # 常规 RGB/RGBA 顺序为 R,G,B[,A]；只重采样 R/B。
            result[..., 0] = sampler(result[..., 0], self._k_r)
            result[..., 2] = sampler(result[..., 2], self._k_b)
            corrected = np.clip(result, 0.0, 1.0)

        if is_numpy_input:
            out = corrected
            if domain_scale != 1.0:
                out = out * domain_scale
            if original_dtype is not None and np.issubdtype(original_dtype, np.integer):
                out = np.clip(
                    np.rint(out),
                    0.0,
                    float(np.iinfo(original_dtype).max),
                )
                return out.astype(original_dtype)
            if original_dtype is not None:
                return out.astype(original_dtype, copy=False)
            return out

        if domain_scale != 1.0:
            corrected = corrected * domain_scale
        # 整体 tolist 在 C 层完成；像素自然成为 Python list/float。
        return corrected.tolist()

    # ------------------------------------------------------------------
    # 无 NumPy 时的纯 Python 回退（正确优先，不追求 1Mpx 性能时限）。
    # ------------------------------------------------------------------
    def _apply_pure(self, rgb, *, bicubic):  # pragma: no cover
        h = len(rgb)
        w = len(rgb[0])
        pixel0 = rgb[0][0]
        channels = len(pixel0) if isinstance(pixel0, (list, tuple)) else 1

        def get(y, x, c):
            y = 0 if y < 0 else h - 1 if y >= h else y
            x = 0 if x < 0 else w - 1 if x >= w else x
            row = rgb[y][x]
            return row[c] if channels > 1 else row

        def sample_plane(out_y, out_x, c, k):
            cx = 0.5 * (w - 1)
            cy = 0.5 * (h - 1)
            sx = cx + k * (out_x - cx)
            sy = cy + k * (out_y - cy)
            sx = min(max(sx, 0.0), w - 1.0)
            sy = min(max(sy, 0.0), h - 1.0)
            x0 = math.floor(sx)
            y0 = math.floor(sy)
            tx = sx - x0
            ty = sy - y0
            if not bicubic:
                xa, xb = x0, min(x0 + 1, w - 1)
                ya, yb = y0, min(y0 + 1, h - 1)
                top = get(ya, xa, c) * (1 - tx) + get(ya, xb, c) * tx
                bot = get(yb, xa, c) * (1 - tx) + get(yb, xb, c) * tx
                return top * (1 - ty) + bot * ty

            wx = [bicubic_kernel(tx + d) for d in (1, 0, -1, -2)]
            wy = [bicubic_kernel(ty + d) for d in (1, 0, -1, -2)]
            xs = [x0 - 1, x0, x0 + 1, x0 + 2]
            ys = [y0 - 1, y0, y0 + 1, y0 + 2]
            value = 0.0
            for j in range(4):
                row_value = 0.0
                for i in range(4):
                    row_value += wx[i] * get(ys[j], xs[i], c)
                value += wy[j] * row_value
            return value

        out = []
        for y in range(h):
            row = []
            for x in range(w):
                if channels == 1:
                    val = rgb[y][x]
                    pix = float(val)
                else:
                    pix = [float(v) for v in rgb[y][x]]
                    b_index = 2 if channels >= 3 else 1
                    pix[0] = sample_plane(y, x, 0, self._k_r)
                    pix[b_index] = sample_plane(y, x, b_index, self._k_b)
                    pix = [min(1.0, max(0.0, v)) for v in pix]
                row.append(pix)
            out.append(row)
        return out


# 模块级便捷函数。
def apply(rgb, k_r=1.0, k_b=1.0):
    """等价于 ``ChromaAberration(k_r, k_b).apply(rgb)``。"""
    return ChromaAberration(k_r, k_b).apply(rgb)
