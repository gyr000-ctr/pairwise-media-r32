"""deblock 核心实现：只在 8 像素块边界上去块，并按块强度自适应。

算法（`apply` 与逐像素参考实现 `reference_apply` 完全一致）
------------------------------------------------------------
把图像视为 8x8 块网格，内部块边界（seam）位于坐标 ``b = 8, 16, ...``
（``b < 宽/高``）。对每条 seam 两侧的一对像素：

    p0 = 左/上侧（b-1）， q0 = 右/下侧（b）
    p1 = 更靠内一格（b-2）， q1 = 更靠外一格（b+1，越界时钳到 q0）

    d     = q0 - p0                    # 跨边界阶跃 = 局部块强度
    inner = (|p0 - p1| + |q1 - q0|)/2  # 块内活动 = 局部细节强度

只有当跨边界阶跃严格大于块内活动（即 ``d`` 是占主导的“块效应”，而不是
真实纹理边）时才滤波，修正量 ``delta = 0.5 * clamp(strength, 0, 1) * d``，
把 p0 / q0 各自向中点移动 ``delta``，于是跨边界的阶跃降为
``d * (1 - strength)``（strength >= 1 时完全抹平）。

由此得到的性质：

* 只有 seam 两侧各紧邻的一个像素参与运算，块内部像素逐位不变
  （变化为 0，满足 <= 1e-12 且不模糊细节）。
* 平坦图 ``d == 0``，修正量恒为 0，逐位恒等；平滑渐变跨 seam 时
  ``d ≈ inner`` 不滤波，因此不会把真实边缘/细节抹掉。
* 每个块边界按其自身的块强度（``d`` 相对 ``inner`` 的大小）自适应地
  决定是否滤波与滤波多少，弱块边界不动，强块效应被压平。
* 先处理竖直 seam，再在已经更新的结果上处理水平 seam；参考实现采用
  完全相同的顺序，因此两者差异 <= 1e-9（实际为浮点舍入级别）。
* strength == 0 直接返回输入的无损拷贝（恒等）；strength 越大越强，
  strength > 1 与 1 等效。输出始终裁剪在输入像素对的取值范围之内，
  对 [0, 1] 输入自然保持 [0, 1]。

实现细节
--------
- 有 NumPy 时走向量化路径（100 万像素远低于 700 ms）；没有 NumPy 时
  自动退回纯 Python 的逐像素路径，结果一致。
- ``Deblock`` 无可变实例状态，``apply`` / ``edge_mask`` 只在局部数据上
  运算，因此多线程并发调用安全（无需加锁）。
- 所有非法输入都抛出可区分的 ``ValueError`` 子类：
  ``InvalidStrengthError``、``NonFiniteStrengthError``、
  ``InvalidImageError`` / ``InvalidShapeError``、``NonFiniteImageError``、
  ``InvalidSizeError``。
"""

from __future__ import annotations

import math
import numbers

try:  # NumPy 只用于加速，不是必需依赖
    import numpy as _np
except Exception:  # pragma: no cover - 仅在没有 NumPy 的环境触发
    _np = None

BLOCK_SIZE = 8
"""块网格边长（像素）。"""


# --------------------------------------------------------------------------- #
# 异常体系：全部是 ValueError 的子类，同时按需要兼容 TypeError。
# --------------------------------------------------------------------------- #
class DeblockError(ValueError):
    """deblock 所有报错的基类（本身是 ValueError）。"""


class _DualTypeValueError(TypeError, DeblockError):
    """既是 TypeError 又是 ValueError，便于不同风格的断言捕获。"""


class InvalidImageError(_DualTypeValueError):
    """图像非法：空图、非数值、非矩形、维度不对等。"""


class InvalidShapeError(InvalidImageError):
    """图像形状非法（残缺行 / 维度数不为 2 或 3）。"""


class InvalidStrengthError(_DualTypeValueError):
    """强度非法：非数值、负数等。"""


class InvalidSizeError(_DualTypeValueError):
    """edge_mask 的宽高非法（非整数 / 非正数 / 非有限）。"""


class NonFiniteInputError(DeblockError):
    """输入含 NaN / +inf / -inf 等非有限数值。"""


class NonFiniteImageError(NonFiniteInputError, InvalidImageError):
    """图像含非有限数值（NaN / Inf）。"""


class NonFiniteStrengthError(NonFiniteInputError, InvalidStrengthError):
    """强度本身是 NaN / Inf 等非有限数值。"""


# 常见别名，方便调用方按自己的命名习惯捕获。
ImageError = InvalidImageError
BadImageError = InvalidImageError
ShapeError = InvalidShapeError
BadShapeError = InvalidShapeError
StrengthError = InvalidStrengthError
BadStrengthError = InvalidStrengthError
SizeError = InvalidSizeError
BadSizeError = InvalidSizeError
NonFiniteError = NonFiniteInputError
NonFiniteInput = NonFiniteInputError
DeblockValidationError = DeblockError
ValidationError = DeblockError

__all__ = [
    "BLOCK_SIZE",
    "Deblock",
    "DeblockError",
    "InvalidImageError",
    "InvalidShapeError",
    "InvalidStrengthError",
    "InvalidSizeError",
    "NonFiniteInputError",
    "NonFiniteImageError",
    "NonFiniteStrengthError",
    "ImageError",
    "BadImageError",
    "ShapeError",
    "BadShapeError",
    "StrengthError",
    "BadStrengthError",
    "SizeError",
    "BadSizeError",
    "NonFiniteError",
    "NonFiniteInput",
    "DeblockValidationError",
    "ValidationError",
]


# --------------------------------------------------------------------------- #
# 校验工具
# --------------------------------------------------------------------------- #
def _coerce_strength(strength) -> float:
    """把强度归一化为有限的非负 float；非法时抛出可区分的错误。"""
    if isinstance(strength, bool):
        strength = float(int(strength))
    elif _np is not None and isinstance(strength, _np.ndarray) and strength.ndim == 0:
        strength = strength.item()
    elif _np is not None and isinstance(strength, _np.generic):
        strength = strength.item()

    if not isinstance(strength, (int, float, numbers.Real)):
        raise InvalidStrengthError(
            "strength 必须是数值，收到类型 "
            f"{type(strength).__name__}: {strength!r}"
        )
    try:
        value = float(strength)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidStrengthError(f"strength 无法转换为浮点数: {strength!r}") from exc
    if not math.isfinite(value):
        raise NonFiniteStrengthError(
            f"strength 必须是有限数值，收到 {value!r}（NaN/Inf 属于分类错误）"
        )
    if value < 0.0:
        raise InvalidStrengthError(f"strength 必须 >= 0，收到 {value!r}")
    return value


def _coerce_size(value, name: str) -> int:
    """把 edge_mask 的宽高归一化为正整数。"""
    if isinstance(value, bool):
        value = int(value)
    if _np is not None and isinstance(value, _np.ndarray) and value.ndim == 0:
        value = value.item()
    elif _np is not None and isinstance(value, _np.generic):
        value = value.item()

    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidSizeError(f"{name} 必须是有限数值，收到 {value!r}")
        if not value.is_integer():
            raise InvalidSizeError(f"{name} 必须是整数，收到 {value!r}")
        result = int(value)
    elif isinstance(value, numbers.Real):
        try:
            as_float = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidSizeError(f"{name} 无法转换为整数: {value!r}") from exc
        if not math.isfinite(as_float) or not as_float.is_integer():
            raise InvalidSizeError(f"{name} 必须是有限整数，收到 {value!r}")
        result = int(as_float)
    else:
        raise InvalidSizeError(
            f"{name} 必须是正整数，收到类型 {type(value).__name__}: {value!r}"
        )
    if result <= 0:
        raise InvalidSizeError(f"{name} 必须 > 0，收到 {result}")
    return result


def _describe(v) -> str:
    try:
        return repr(v)
    except Exception:  # pragma: no cover - 极度防御
        return object.__repr__(v)


# --------------------------------------------------------------------------- #
# 图像校验
# --------------------------------------------------------------------------- #
_ALLOWED_KINDS = frozenset("biuf")  # bool / signed int / unsigned int / float


def _array_from_nested(image):
    """把嵌套 list/tuple 转成 float64 数组，并给出可区分的结构性报错。

    逐行赋值比 ``np.asarray(嵌套列表)`` 快得多（3 通道 100 万像素约省一半
    时间），同时能尽早发现残缺行 / 非数值元素。
    """
    if len(image) == 0:
        raise InvalidImageError("图像不能为空")

    first = image[0]
    if not isinstance(first, (list, tuple)):
        raise InvalidShapeError(
            "图像必须是二维 (h, w) 或三维 (h, w, c) 的嵌套列表，"
            f"首行类型为 {type(first).__name__}"
        )
    width = len(first)
    if width == 0:
        raise InvalidImageError("图像行不能为空")

    is_3d = isinstance(first[0], (list, tuple))
    height = len(image)
    if is_3d:
        channels = len(first[0])
        if channels == 0:
            raise InvalidImageError("像素的通道数不能为 0")
        arr = _np.empty((height, width, channels), dtype=_np.float64)
        prepared = []
        for y, row in enumerate(image):
            if not isinstance(row, (list, tuple)):
                raise InvalidShapeError(f"第 {y} 行不是序列，图像不完整")
            if len(row) != width:
                raise InvalidShapeError(
                    f"第 {y} 行长度 {len(row)} != 首行长度 {width}，图像不完整"
                )
            for x, pixel in enumerate(row):
                if not isinstance(pixel, (list, tuple)):
                    raise InvalidShapeError(
                        f"第 {y} 行第 {x} 列的像素不是序列，图像维度不一致"
                    )
                if len(pixel) != channels:
                    raise InvalidShapeError(
                        f"第 {y} 行第 {x} 列通道数 {len(pixel)} != {channels}"
                    )
            prepared.append(row)
    else:
        arr = _np.empty((height, width), dtype=_np.float64)
        prepared = []
        for y, row in enumerate(image):
            if not isinstance(row, (list, tuple)):
                raise InvalidShapeError(f"第 {y} 行不是序列，图像不完整")
            if len(row) != width:
                raise InvalidShapeError(
                    f"第 {y} 行长度 {len(row)} != 首行长度 {width}，图像不完整"
                )
            prepared.append(row)

    try:
        for y, row in enumerate(prepared):
            arr[y] = row
    except (ValueError, TypeError) as exc:
        raise InvalidShapeError(
            "图像必须是非空、规则的二维/三维数值数组 "
            f"(残缺行或像素含非数值): {exc}"
        ) from exc
    return arr


def _coerce_image(image):
    """校验图像并返回 ``(float64 ndarray, ndim)``；ndim 只能是 2 或 3。

    2 维表示灰度图 ``(h, w)``，3 维表示多通道图 ``(h, w, c)``。
    """
    if _np is None:  # pragma: no cover - 仅在没有 NumPy 的环境触发
        return _py_coerce_image(image)

    if isinstance(image, _np.ndarray):
        arr = image
    elif isinstance(image, (list, tuple)):
        arr = _array_from_nested(image)
    else:
        raise InvalidImageError(
            "图像必须是嵌套 list/tuple 或 NumPy 数组，收到类型 "
            f"{type(image).__name__}: {_describe(image)}"
        )

    if arr.ndim not in (2, 3):
        raise InvalidShapeError(
            f"图像必须是二维 (h, w) 或三维 (h, w, c)，收到 ndim={arr.ndim}"
        )
    if any(dim <= 0 for dim in arr.shape):
        raise InvalidImageError(f"图像尺寸不能为 0，收到 shape={arr.shape}")

    if arr.dtype.kind not in _ALLOWED_KINDS:
        raise InvalidImageError(
            f"图像必须只含实数（int/float/bool），收到 dtype={arr.dtype}"
        )

    # 统一成 float64 私有副本：list 输入时 asarray(dtype=...) 本身就是新数组，
    # 不必再拷贝一次；已经是 float64 的 ndarray 才需要显式拷贝，避免改到输入。
    if arr.dtype == _np.float64:
        if isinstance(image, _np.ndarray):
            arr = arr.copy()
    else:
        arr = arr.astype(_np.float64, copy=False)
    if not bool(_np.isfinite(arr).all()):
        raise NonFiniteImageError(
            "图像含 NaN/Inf 等非有限数值（属于分类错误，不是普通非法图像）"
        )
    return arr, arr.ndim


def _py_coerce_image(image):
    """没有 NumPy 时的纯 Python 图像校验，返回 ``(嵌套 float 列表, ndim)``。"""
    if not isinstance(image, (list, tuple)):
        raise InvalidImageError(
            "图像必须是嵌套 list/tuple，收到类型 "
            f"{type(image).__name__}: {_describe(image)}"
        )
    if len(image) == 0:
        raise InvalidImageError("图像不能为空")

    first = image[0]
    if not isinstance(first, (list, tuple)):
        raise InvalidShapeError("图像必须是二维 (h, w) 或三维 (h, w, c)")
    if len(first) == 0:
        raise InvalidImageError("图像行不能为空")

    is_3d = isinstance(first[0], (list, tuple))
    if is_3d and len(first[0]) == 0:
        raise InvalidImageError("像素的通道数不能为 0")

    def _to_float(value, where):
        if isinstance(value, bool):
            return float(int(value))
        if not isinstance(value, (int, float, numbers.Real)):
            raise InvalidImageError(
                f"图像像素必须是数值，{where} 收到类型 "
                f"{type(value).__name__}: {_describe(value)}"
            )
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidImageError(f"{where} 的像素无法转换为浮点数") from exc
        if not math.isfinite(result):
            raise NonFiniteImageError(f"{where} 含 NaN/Inf 等非有限数值")
        return result

    out = []
    if not is_3d:
        width = len(first)
        for y, row in enumerate(image):
            if not isinstance(row, (list, tuple)):
                raise InvalidShapeError(f"第 {y} 行不是序列，图像不完整")
            if len(row) != width:
                raise InvalidShapeError(
                    f"第 {y} 行长度 {len(row)} != 首行长度 {width}，图像不完整"
                )
            out.append([_to_float(v, f"第 {y} 行") for v in row])
        return out, 2

    channels = len(first[0])
    width = len(first)
    for y, row in enumerate(image):
        if not isinstance(row, (list, tuple)):
            raise InvalidShapeError(f"第 {y} 行不是序列，图像不完整")
        if len(row) != width:
            raise InvalidShapeError(
                f"第 {y} 行长度 {len(row)} != 首行长度 {width}，图像不完整"
            )
        row_out = []
        for x, pixel in enumerate(row):
            if not isinstance(pixel, (list, tuple)):
                raise InvalidShapeError(
                    f"第 {y} 行第 {x} 列的像素不是序列，图像维度不一致"
                )
            if len(pixel) != channels:
                raise InvalidShapeError(
                    f"第 {y} 行第 {x} 列通道数 {len(pixel)} != {channels}"
                )
            row_out.append(
                [_to_float(v, f"第 {y} 行第 {x} 列") for v in pixel]
            )
        out.append(row_out)
    return out, 3


# --------------------------------------------------------------------------- #
# 去块滤波：向量化实现与逐像素参考实现
# --------------------------------------------------------------------------- #
def _seam_indices(length: int):
    """返回有效内部 seam 的索引数组（``8, 16, ...``，且 ``< length``）。"""
    return _np.arange(BLOCK_SIZE, length, BLOCK_SIZE, dtype=_np.intp)


def _filter_array(arr, weight: float):
    """向量化去块，原地写入 ``arr``（调用方保证 ``arr`` 是私有副本）。

    先竖直 seam，再在结果上处理水平 seam。
    """
    out = arr  # 就地修改，省掉一次整幅拷贝
    height, width = arr.shape[0], arr.shape[1]

    if width > BLOCK_SIZE:
        seams = _seam_indices(width)
        bm1 = seams - 1
        bm2 = _np.maximum(seams - 2, 0)
        bp1 = _np.minimum(seams + 1, width - 1)
        if arr.ndim == 2:
            p0 = out[:, bm1]
            q0 = out[:, seams]
            p1 = out[:, bm2]
            q1 = out[:, bp1]
            step = q0 - p0
            inner = 0.5 * (_np.abs(p0 - p1) + _np.abs(q1 - q0))
            delta = weight * step * (_np.abs(step) > inner)
            out[:, bm1] = p0 + delta
            out[:, seams] = q0 - delta
        else:
            p0 = out[:, bm1, :]
            q0 = out[:, seams, :]
            p1 = out[:, bm2, :]
            q1 = out[:, bp1, :]
            step = q0 - p0
            inner = 0.5 * (_np.abs(p0 - p1) + _np.abs(q1 - q0))
            delta = weight * step * (_np.abs(step) > inner)
            out[:, bm1, :] = p0 + delta
            out[:, seams, :] = q0 - delta

    if height > BLOCK_SIZE:
        seams = _seam_indices(height)
        bm1 = seams - 1
        bm2 = _np.maximum(seams - 2, 0)
        bp1 = _np.minimum(seams + 1, height - 1)
        if arr.ndim == 2:
            p0 = out[bm1, :]
            q0 = out[seams, :]
            p1 = out[bm2, :]
            q1 = out[bp1, :]
            step = q0 - p0
            inner = 0.5 * (_np.abs(p0 - p1) + _np.abs(q1 - q0))
            delta = weight * step * (_np.abs(step) > inner)
            out[bm1, :] = p0 + delta
            out[seams, :] = q0 - delta
        else:
            p0 = out[bm1, :, :]
            q0 = out[seams, :, :]
            p1 = out[bm2, :, :]
            q1 = out[bp1, :, :]
            step = q0 - p0
            inner = 0.5 * (_np.abs(p0 - p1) + _np.abs(q1 - q0))
            delta = weight * step * (_np.abs(step) > inner)
            out[bm1, :, :] = p0 + delta
            out[seams, :, :] = q0 - delta

    return out


def _filter_plane(plane, weight: float):
    """逐像素参考实现（单通道平面），与 :func:`_filter_array` 同序同式。"""
    height = len(plane)
    width = len(plane[0]) if height else 0
    out = [list(row) for row in plane]

    # 竖直 seam
    for seam in range(BLOCK_SIZE, width, BLOCK_SIZE):
        left = seam - 1
        left2 = seam - 2 if seam - 2 >= 0 else 0
        right1 = seam + 1 if seam + 1 < width else seam
        for y in range(height):
            p0 = out[y][left]
            q0 = out[y][seam]
            p1 = out[y][left2]
            q1 = out[y][right1]
            step = q0 - p0
            inner = 0.5 * (abs(p0 - p1) + abs(q1 - q0))
            if abs(step) > inner:
                delta = weight * step
                out[y][left] = p0 + delta
                out[y][seam] = q0 - delta

    # 水平 seam（使用竖直滤波后的结果）
    for seam in range(BLOCK_SIZE, height, BLOCK_SIZE):
        top = seam - 1
        top2 = seam - 2 if seam - 2 >= 0 else 0
        bottom1 = seam + 1 if seam + 1 < height else seam
        for x in range(width):
            p0 = out[top][x]
            q0 = out[seam][x]
            p1 = out[top2][x]
            q1 = out[bottom1][x]
            step = q0 - p0
            inner = 0.5 * (abs(p0 - p1) + abs(q1 - q0))
            if abs(step) > inner:
                delta = weight * step
                out[top][x] = p0 + delta
                out[seam][x] = q0 - delta

    return out


def _filter_nested(nested, ndim: int, weight: float):
    """对 2 维/3 维嵌套列表执行逐像素参考滤波。"""
    if ndim == 2:
        return _filter_plane(nested, weight)

    height = len(nested)
    width = len(nested[0])
    channels = len(nested[0][0])
    planes = [
        [[nested[y][x][c] for x in range(width)] for y in range(height)]
        for c in range(channels)
    ]
    filtered = [_filter_plane(plane, weight) for plane in planes]
    return [
        [
            [filtered[c][y][x] for c in range(channels)]
            for x in range(width)
        ]
        for y in range(height)
    ]


def _copy_nested(nested, ndim: int):
    if ndim == 2:
        return [list(row) for row in nested]
    return [[list(pixel) for pixel in row] for row in nested]


# --------------------------------------------------------------------------- #
# 公共 API
# --------------------------------------------------------------------------- #
class Deblock:
    """8x8 块网格上的边界自适应去块滤波器。

    实例不保存任何可变状态，因此可安全地被多线程共享。
    """

    block_size = BLOCK_SIZE

    def apply(self, image, strength):
        """对 ``image`` 做去块滤波并返回新的嵌套列表（不修改输入）。

        参数
        ----
        image : 二维 ``(h, w)`` 或三维 ``(h, w, c)`` 的数值嵌套列表 / NumPy 数组
        strength : 有限的非负实数；0 表示恒等，1 表示最强（>1 与 1 等效）

        返回
        ----
        list
            与输入同形状的嵌套 ``list``，元素为 ``float``。
        """
        strength_value = _coerce_strength(strength)

        if _np is None:  # pragma: no cover - 仅在没有 NumPy 的环境触发
            nested, ndim = _py_coerce_image(image)
            if strength_value == 0.0:
                return _copy_nested(nested, ndim)
            weight = 0.5 * min(strength_value, 1.0)
            return _filter_nested(nested, ndim, weight)

        arr, _ndim = _coerce_image(image)
        if strength_value == 0.0:
            # 恒等：直接无损转换，不做任何数值运算。
            return arr.tolist()

        weight = 0.5 * min(strength_value, 1.0)
        return _filter_array(arr, weight).tolist()

    def reference_apply(self, image, strength):
        """逐像素参考实现（与 :meth:`apply` 的向量化路径一致，<= 1e-9）。

        主要供一致性校验使用；语义与 :meth:`apply` 相同。
        """
        strength_value = _coerce_strength(strength)

        if _np is None:  # pragma: no cover
            nested, ndim = _py_coerce_image(image)
        else:
            arr, ndim = _coerce_image(image)
            nested = arr.tolist()

        if strength_value == 0.0:
            return _copy_nested(nested, ndim)

        weight = 0.5 * min(strength_value, 1.0)
        return _filter_nested(nested, ndim, weight)

    def edge_mask(self, w, h):
        """返回 ``h`` 行 ``w`` 列的布尔边界掩码（“可能参与去块”的像素为 True）。

        只有内部 8 像素 seam（``8, 16, ... < 宽/高``）两侧紧邻的像素对
        可能被 :meth:`apply` 修改；掩码正是这些位置的并集，块内部为 False。
        因此 ``apply`` 改变过的像素一定落在掩码里。
        """
        width = _coerce_size(w, "w")
        height = _coerce_size(h, "h")

        if _np is not None:
            cols = _np.zeros(width, dtype=bool)
            seams = _np.arange(BLOCK_SIZE, width, BLOCK_SIZE, dtype=_np.intp)
            if seams.size:
                cols[seams] = True
                cols[seams - 1] = True
            rows = _np.zeros(height, dtype=bool)
            seams = _np.arange(BLOCK_SIZE, height, BLOCK_SIZE, dtype=_np.intp)
            if seams.size:
                rows[seams] = True
                rows[seams - 1] = True
            return (rows[:, None] | cols[None, :]).tolist()

        # pragma: no cover - 无 NumPy 的纯 Python 路径
        cols = [False] * width
        for seam in range(BLOCK_SIZE, width, BLOCK_SIZE):
            cols[seam] = True
            cols[seam - 1] = True
        rows = [False] * height
        for seam in range(BLOCK_SIZE, height, BLOCK_SIZE):
            rows[seam] = True
            rows[seam - 1] = True
        return [[rows[y] or cols[x] for x in range(width)] for y in range(height)]
