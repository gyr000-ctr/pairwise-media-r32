"""Brown-Conrady 镜头畸变模型（全参数 k1/k2/p1/p2）。

坐标约定：归一化图像坐标 ``u, v ∈ [0, 1]``，主点（畸变中心）固定在
``(0.5, 0.5)``。所有几何计算在以主点为原点的中心化坐标 ``x = u - 0.5``、
``y = v - 0.5`` 中进行。

正向畸变（理想坐标 -> 畸变坐标）：

    r² = x² + y²
    R  = 1 + k1·r² + k2·r⁴
    xd = x·R + 2·p1·x·y + p2·(r² + 2·x²)
    yd = y·R + p1·(r² + 2·y²) + 2·p2·x·y

逆向（``undistort``）采用牛顿迭代，每步使用正向映射的解析雅可比求解
2×2 线性修正量，二次收敛，往返误差远优于 1e-9。

实例只保存不可变的系数，方法内不使用任何可变共享状态，因此多线程并发
调用是安全的。
"""

import math

import numpy as np

__all__ = ["LensDistort"]

_CENTER = 0.5
_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-13
_NEWTON_STEP_TOL = 1e-14


def _check_coef(name, value):
    """校验畸变系数：非数值报 TypeError，非有限值报 ValueError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须为实数，实际类型: {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{name} 必须为有限实数，得到: {value!r}")
    return fv


def _check_point(u, v):
    """校验点坐标：非数值报 TypeError，非有限值报 ValueError。"""
    if isinstance(u, bool) or not isinstance(u, (int, float)):
        raise TypeError(f"u 必须为实数，实际类型: {type(u).__name__}")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"v 必须为实数，实际类型: {type(v).__name__}")
    fu, fv = float(u), float(v)
    if not math.isfinite(fu) or not math.isfinite(fv):
        raise ValueError(f"坐标必须为有限实数，得到: ({u!r}, {v!r})")
    return fu, fv


class LensDistort:
    """Brown-Conrady 镜头畸变模型。

    Parameters
    ----------
    k1, k2:
        径向畸变系数（二阶、四阶）。
    p1, p2:
        切向（偏心）畸变系数，缺省为 0。
    """

    __slots__ = ("k1", "k2", "p1", "p2")

    def __init__(self, k1, k2, p1=0.0, p2=0.0):
        self.k1 = _check_coef("k1", k1)
        self.k2 = _check_coef("k2", k2)
        self.p1 = _check_coef("p1", p1)
        self.p2 = _check_coef("p2", p2)

    # ------------------------------------------------------------------ #
    # 标量几何核心（中心化坐标）
    # ------------------------------------------------------------------ #
    def _distort_xy(self, x, y):
        """正向畸变，输入/输出均为中心化归一化坐标。"""
        k1, k2, p1, p2 = self.k1, self.k2, self.p1, self.p2
        r2 = x * x + y * y
        radial = 1.0 + (k1 + k2 * r2) * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    def _jac_xy(self, x, y):
        """正向畸变解析雅可比（中心化坐标），返回展平的 2×2 元素。"""
        k1, k2, p1, p2 = self.k1, self.k2, self.p1, self.p2
        r2 = x * x + y * y
        a = k1 + 2.0 * k2 * r2          # ∂R/∂r²
        radial = 1.0 + (k1 + k2 * r2) * r2
        drdx = 2.0 * x * a
        drdy = 2.0 * y * a
        j00 = radial + x * drdx + 2.0 * p1 * y + 6.0 * p2 * x
        j01 = x * drdy + 2.0 * p1 * x + 2.0 * p2 * y
        j10 = y * drdx + 2.0 * p1 * x + 2.0 * p2 * y
        j11 = radial + y * drdy + 6.0 * p1 * y + 2.0 * p2 * x
        return j00, j01, j10, j11

    def _undistort_xy(self, xd, yd):
        """牛顿迭代求逆：求解 D(x, y) = (xd, yd)，中心化坐标。

        牛顿法在雅可比非奇异（映射局部可逆）时二次收敛。仅在迭代真正发散
        （出现非有限值或残差持续放大）时启用步长回退兜底；若映射本身非单射
        （雅可比行列式过零，物理上不存在唯一逆），抛出 RuntimeError。
        """
        x, y = xd, yd
        prev_res = math.inf
        stagnant = 0
        for _ in range(_NEWTON_MAX_ITER):
            dx, dy = self._distort_xy(x, y)
            fx, fy = dx - xd, dy - yd
            res = math.hypot(fx, fy)
            if res <= _NEWTON_TOL:
                return x, y
            j00, j01, j10, j11 = self._jac_xy(x, y)
            det = j00 * j11 - j01 * j10
            if det == 0.0 or not math.isfinite(det):
                raise RuntimeError("畸变雅可比奇异，该点不存在唯一逆变换")
            # 解 J·δ = f（2×2 显式逆）
            delta_x = (j11 * fx - j01 * fy) / det
            delta_y = (j00 * fy - j10 * fx) / det
            step_len = math.hypot(delta_x, delta_y)
            if step_len <= _NEWTON_STEP_TOL:
                if res <= 1e-12:
                    return x, y
                raise RuntimeError("牛顿迭代停滞，逆变换未收敛")

            # 常规情况直接取完整牛顿步（保持二次收敛，无额外求值）；
            # 仅当结果非有限或残差上升（真正发散）时才进入步长回退。
            nx, ny = x - delta_x, y - delta_y
            if math.isfinite(nx) and math.isfinite(ny):
                ddx, ddy = self._distort_xy(nx, ny)
                new_res = math.hypot(ddx - xd, ddy - yd)
            else:
                new_res = math.inf
            if not math.isfinite(new_res) or new_res >= res:
                new_res, nx, ny = self._line_search(x, y, delta_x, delta_y,
                                                     xd, yd, res)

            # 残差长期不下降则判定停滞（防止强非单射参数下空转）
            if new_res >= prev_res * (1.0 - 1e-12):
                stagnant += 1
                if stagnant >= 8:
                    raise RuntimeError("牛顿迭代停滞，逆变换未收敛")
            else:
                stagnant = 0
            prev_res = new_res
            x, y = nx, ny
        raise RuntimeError("牛顿迭代达到最大次数，逆变换未收敛")

    def _line_search(self, x, y, delta_x, delta_y, xd, yd, res):
        """发散兜底：沿牛顿方向半步回退，找到首个使残差严格下降的步长。"""
        step = 0.5
        for _ in range(20):
            tx, ty = x - step * delta_x, y - step * delta_y
            ddx, ddy = self._distort_xy(tx, ty)
            new_res = math.hypot(ddx - xd, ddy - yd)
            if math.isfinite(new_res) and new_res < res:
                return new_res, tx, ty
            step *= 0.5
        # 残差无法继续下降：通常意味着落入非单射区域
        raise RuntimeError("牛顿迭代无法收敛，该畸变参数下映射可能非单射")

    # ------------------------------------------------------------------ #
    # 公开标量 API
    # ------------------------------------------------------------------ #
    def distort(self, u, v):
        """正向畸变：理想归一化坐标 -> 畸变归一化坐标，返回 ``(float, float)``。"""
        fu, fv = _check_point(u, v)
        xd, yd = self._distort_xy(fu - _CENTER, fv - _CENTER)
        return xd + _CENTER, yd + _CENTER

    def undistort(self, u, v):
        """逆向校正：畸变归一化坐标 -> 理想归一化坐标（牛顿迭代）。"""
        fu, fv = _check_point(u, v)
        x, y = self._undistort_xy(fu - _CENTER, fv - _CENTER)
        return x + _CENTER, y + _CENTER

    def jacobian(self, u, v):
        """``distort`` 在 ``(u, v)`` 处的解析雅可比。

        返回 2×2 元组矩阵 ``((du'/du, du'/dv), (dv'/du, dv'/dv))``，
        其行列式即该点的局部面积缩放。
        """
        fu, fv = _check_point(u, v)
        j00, j01, j10, j11 = self._jac_xy(fu - _CENTER, fv - _CENTER)
        return ((float(j00), float(j01)), (float(j10), float(j11)))

    def jacobian_det(self, u, v):
        """``distort`` 解析雅可比的行列式（局部缩放因子）。"""
        (j00, j01), (j10, j11) = self.jacobian(u, v)
        return j00 * j11 - j01 * j10

    def undistort_jacobian(self, u, v):
        """``undistort`` 在畸变点 ``(u, v)`` 处的解析雅可比。

        由反函数定理，它等于正向雅可比在对应理想点处的逆矩阵。
        """
        fu, fv = _check_point(u, v)
        x, y = self._undistort_xy(fu - _CENTER, fv - _CENTER)
        j00, j01, j10, j11 = self._jac_xy(x, y)
        det = j00 * j11 - j01 * j10
        if det == 0.0 or not math.isfinite(det):
            raise RuntimeError("畸变雅可比奇异，逆雅可比不存在")
        return (
            (j11 / det, -j01 / det),
            (-j10 / det, j00 / det),
        )

    # ------------------------------------------------------------------ #
    # 图像校正
    # ------------------------------------------------------------------ #
    def apply(self, image):
        """对整幅图像做畸变校正（重采样）。

        校正图每个像素按正向映射 ``D`` 回采原图：
        ``out(u, v) = in(D(u, v))``，双线性插值，越界坐标钳制到边界，
        输出值钳制到 ``[0, 1]``。支持灰度（标量像素）与彩色
        （等长序列像素）输入，返回与输入同构的嵌套 Python ``list``，
        不修改输入。
        """
        arr, multichannel = self._image_to_array(image)
        h, w = arr.shape[:2]

        # 输出像素中心的归一化坐标
        cols = (np.arange(w, dtype=np.float64) + 0.5) / w
        rows = (np.arange(h, dtype=np.float64) + 0.5) / h
        grid_u, grid_v = np.meshgrid(cols, rows)

        xd, yd = self._distort_arr(grid_u - _CENTER, grid_v - _CENTER)

        # 映射回原图的连续像素下标并钳制到边界
        sx = np.clip((xd + _CENTER) * w - 0.5, 0.0, w - 1.0)
        sy = np.clip((yd + _CENTER) * h - 0.5, 0.0, h - 1.0)
        c0 = np.floor(sx).astype(np.int64)
        r0 = np.floor(sy).astype(np.int64)
        c1 = np.minimum(c0 + 1, w - 1)
        r1 = np.minimum(r0 + 1, h - 1)
        wc = sx - c0          # (H, W)
        wr = sy - r0

        # 扁平化后用 np.take 聚合并复用缓冲，减少大临时数组的分配与访存
        n = h * w
        flat = arr.reshape(n, -1) if multichannel else arr.reshape(n)
        wcol = wc.reshape(n, 1) if multichannel else wc.reshape(n)
        wrow = wr.reshape(n, 1) if multichannel else wr.reshape(n)

        g00 = np.take(flat, (r0 * w + c0).ravel(), axis=0)
        g01 = np.take(flat, (r0 * w + c1).ravel(), axis=0)
        g10 = np.take(flat, (r1 * w + c0).ravel(), axis=0)
        g11 = np.take(flat, (r1 * w + c1).ravel(), axis=0)
        # 双线性：先沿列，再沿行；就地复用缓冲
        np.subtract(g01, g00, out=g01)
        np.subtract(g11, g10, out=g11)
        top = g00
        top = top + wcol * g01
        bot = g10 + wcol * g11
        out = top + wrow * (bot - top)
        if multichannel:
            out = out.reshape(h, w, flat.shape[1])
        else:
            out = out.reshape(h, w)
        np.clip(out, 0.0, 1.0, out=out)

        if multichannel:
            return [[tuple(px) for px in row] for row in out.tolist()]
        return out.tolist()

    def _distort_arr(self, x, y):
        """正向畸变的向量化实现（numpy 中心化坐标）。"""
        k1, k2, p1, p2 = self.k1, self.k2, self.p1, self.p2
        r2 = x * x + y * y
        radial = 1.0 + (k1 + k2 * r2) * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    @staticmethod
    def _image_to_array(image):
        """结构校验并转为 float64 ndarray，返回 (array, multichannel)。

        结构（行列、通道一致性）用短循环快速拒绝，数值校验（实数、有限）
        交给向量化的 numpy，避免百万像素级的 Python 逐元素循环。
        """
        if isinstance(image, np.ndarray):
            if image.dtype.kind not in ("f", "i", "u"):
                # bool、字符串、object 等非数值 dtype 一律拒绝
                raise TypeError("图像像素必须为实数数值类型")
            arr = image.astype(np.float64, copy=True)
            if arr.ndim not in (2, 3):
                raise ValueError("图像必须为 H×W 或 H×W×C 数组")
            if not np.all(np.isfinite(arr)):
                raise ValueError("图像包含非有限像素值")
            return arr, arr.ndim == 3

        if not isinstance(image, (list, tuple)) or len(image) == 0:
            raise ValueError("图像必须为非空的嵌套序列（行×列）")
        first_row = image[0]
        if not isinstance(first_row, (list, tuple)) or len(first_row) == 0:
            raise ValueError("图像行必须为非空序列")
        width = len(first_row)
        first_px = first_row[0]
        multichannel = isinstance(first_px, (list, tuple))
        channels = len(first_px) if multichannel else None

        # 结构检查仅验证行列/通道一致性，不做逐元素 Python 遍历（百万像素下
        # 逐元素 isinstance 代价过高）。元素类型与有限性统一由 numpy 向量化兜底。
        for row in image:
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise ValueError("图像各行长度必须一致且非空")
            if multichannel:
                for px in row:
                    if not isinstance(px, (list, tuple)) or len(px) != channels:
                        raise ValueError("彩色像素通道数必须一致")
            elif any(isinstance(px, (list, tuple)) for px in row):
                raise ValueError("灰度像素必须为标量")

        # 单次转换。纯 bool 图像保持 bool dtype（kind='b'），字符串等得到
        # 非数值 dtype，均按类型错误拒绝；数值输入转 float64 后再查有限性。
        probe = np.asarray(image)
        if probe.dtype.kind not in ("f", "i", "u"):
            raise TypeError("像素值必须为实数数值类型")
        arr = probe if probe.dtype.kind == "f" else probe.astype(np.float64)
        if arr.ndim not in (2, 3):
            raise ValueError("图像必须为 H×W 或 H×W×C 结构")
        if not np.all(np.isfinite(arr)):
            raise ValueError("像素值必须为有限实数")
        return arr, multichannel
