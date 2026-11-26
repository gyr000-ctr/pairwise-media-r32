"""自动曝光：分位亮度估计 + IIR 时域平滑。

设计要点
--------
- 亮度按分位数（默认中位数）估计，而非均值，避免高光区域主导曝光。
- 分位数由定宽直方图一次扫描得到（O(n)），不排序，满足百万像素时延约束。
- 原始增益 ``target / median`` 经 IIR 一阶低通时域平滑，并限制相邻帧增益的
  相对变化幅度，保证场景不变时单调收敛、场景切换时不闪烁。
- 增益限制在 ``[1/max_gain, max_gain]``；``apply`` 输出裁剪到 ``[0, 1]``，不过曝。
- 所有非法参数抛出 :class:`ValueError` 的可区分子类，便于上层分类处理。
- 估计/应用过程加锁，可被多线程安全调用。
"""

import math
import threading

__all__ = [
    "AutoExposure",
    "ExposureError",
    "InvalidTargetError",
    "InvalidGainError",
    "InvalidImageError",
    "InvalidQuantileError",
    "InvalidParameterError",
]


class ExposureError(ValueError):
    """自动曝光相关错误的基类，同时是 :class:`ValueError`。"""


class InvalidTargetError(ExposureError):
    """目标亮度非法（非有限值或不在 (0, 1) 内）。"""


class InvalidGainError(ExposureError):
    """最大增益非法（非有限值或非正）。"""


class InvalidImageError(ExposureError):
    """图像非法（空、结构错误或含非有限像素）。"""


class InvalidQuantileError(ExposureError):
    """分位点非法（非有限值或不在 [0, 1] 内）。"""


class InvalidParameterError(ExposureError):
    """其它调节参数非法。"""


def _finite_number(value, name, exc):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise exc(f"{name} must be finite, got {value!r}")
    return float(value)


class AutoExposure:
    """分位驱动、时域平滑的自动曝光控制器。

    Parameters
    ----------
    target:
        目标中位（分位）亮度，取值开区间 ``(0, 1)``。
    max_gain:
        增益上界（正数）。增益下界对称取 ``1 / max_gain``。
    quantile_point:
        亮度估计所用分位点，``[0, 1]``，默认 ``0.5``（中位数）。
    smoothing:
        IIR 平滑系数 alpha，``(0, 1]``，越大跟随越快。
    max_step:
        相邻帧增益相对变化上界（如 ``0.5`` 表示单帧变化不超过 50%）。
    bins:
        内部直方图箱数，箱越多分位精度越高，默认 1024（量化误差 < 0.1%）。
    """

    def __init__(
        self,
        target,
        max_gain,
        quantile_point=0.5,
        smoothing=0.5,
        max_step=0.5,
        bins=1024,
    ):
        target = _finite_number(target, "target", InvalidTargetError)
        if not (0.0 < target < 1.0):
            raise InvalidTargetError(
                f"target must be within (0, 1) exclusive, got {target!r}"
            )

        max_gain = _finite_number(max_gain, "max_gain", InvalidGainError)
        if max_gain <= 0.0:
            raise InvalidGainError(
                f"max_gain must be positive, got {max_gain!r}"
            )

        quantile_point = _finite_number(
            quantile_point, "quantile_point", InvalidQuantileError
        )
        if not (0.0 <= quantile_point <= 1.0):
            raise InvalidQuantileError(
                f"quantile_point must be within [0, 1], got {quantile_point!r}"
            )

        smoothing = _finite_number(
            smoothing, "smoothing", InvalidParameterError
        )
        if not (0.0 < smoothing <= 1.0):
            raise InvalidParameterError(
                f"smoothing must be within (0, 1], got {smoothing!r}"
            )

        max_step = _finite_number(max_step, "max_step", InvalidParameterError)
        if not (0.0 < max_step <= 1.0):
            raise InvalidParameterError(
                f"max_step must be within (0, 1], got {max_step!r}"
            )

        if not isinstance(bins, int) or bins <= 0:
            raise InvalidParameterError(
                f"bins must be a positive integer, got {bins!r}"
            )

        self.target = target
        self.max_gain = max_gain
        self.min_gain = 1.0 / max_gain
        self.quantile_point = quantile_point
        self.smoothing = smoothing
        self.max_step = max_step
        self.bins = bins

        self._gain = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 输入校验与直方图统计
    # ------------------------------------------------------------------ #
    def _rows(self, image):
        """返回行迭代器；支持嵌套序列与一维扁平序列。"""
        if image is None or isinstance(image, (str, bytes)):
            raise InvalidImageError("image must be a non-empty numeric matrix")
        try:
            iterator = iter(image)
        except TypeError as exc:
            raise InvalidImageError(
                f"image must be iterable, got {type(image).__name__}"
            ) from exc
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise InvalidImageError("image must not be empty") from exc

        if isinstance(first, (int, float)) and not isinstance(first, bool):
            return (image,)
        if not hasattr(first, "__iter__"):
            raise InvalidImageError(
                "image rows must be iterable sequences of numbers"
            )
        return (first, *iterator) if not isinstance(image, list) else image

    def _histogram(self, image):
        """一次扫描构建直方图，返回 (hist, count, min, max)。"""
        rows = self._rows(image)
        bins = self.bins
        hist = [0] * bins
        scale = float(bins)
        finite = math.isfinite
        count = 0
        vmin = math.inf
        vmax = -math.inf

        try:
            for row in rows:
                for value in row:
                    if not finite(value):
                        raise InvalidImageError(
                            f"pixels must be finite, got {value!r}"
                        )
                    value = float(value)
                    if value < vmin:
                        vmin = value
                    if value > vmax:
                        vmax = value
                    index = int(value * scale)
                    if index >= bins:
                        index = bins - 1
                    elif index < 0:
                        index = 0
                    hist[index] += 1
                    count += 1
        except TypeError as exc:
            raise InvalidImageError(
                f"image must contain numeric pixels: {exc}"
            ) from exc

        if count == 0:
            raise InvalidImageError("image must contain at least one pixel")
        return hist, count, vmin, vmax

    def _quantile_from_hist(self, hist, count, vmin, vmax, q):
        """基于直方图做线性插值的分位估计。"""
        if q <= 0.0:
            return float(vmin)
        if q >= 1.0:
            return float(vmax)

        bins = self.bins
        target = q * count
        cumulative = 0
        for index, bucket in enumerate(hist):
            if bucket == 0:
                continue
            if cumulative + bucket >= target:
                low = index / bins
                high = (index + 1) / bins
                fraction = (target - cumulative) / bucket
                if fraction < 0.0:
                    fraction = 0.0
                elif fraction > 1.0:
                    fraction = 1.0
                return low + fraction * (high - low)
            cumulative += bucket
        # 理论不可达：浮点兜底返回最大值
        return float(vmax)

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def quantile(self, image, q):
        """计算图像亮度的 ``q`` 分位数（按分位而非均值估计亮度）。"""
        q = _finite_number(q, "q", InvalidQuantileError)
        if not (0.0 <= q <= 1.0):
            raise InvalidQuantileError(
                f"quantile q must be within [0, 1], got {q!r}"
            )
        hist, count, vmin, vmax = self._histogram(image)
        return self._quantile_from_hist(hist, count, vmin, vmax, q)

    def _raw_gain(self, hist, count, vmin, vmax):
        """由分位亮度直接推出的目标增益（经增益限幅）。"""
        brightness = self._quantile_from_hist(
            hist, count, vmin, vmax, self.quantile_point
        )
        if brightness <= 0.0:
            return self.max_gain
        gain = self.target / brightness
        if gain > self.max_gain:
            return self.max_gain
        if gain < self.min_gain:
            return self.min_gain
        return gain

    def estimate(self, image):
        """估计当前帧应使用的曝光增益（含 IIR 时域平滑与帧间限幅）。"""
        with self._lock:
            hist, count, vmin, vmax = self._histogram(image)
            raw = self._raw_gain(hist, count, vmin, vmax)

            previous = self._gain
            if previous is None:
                # 首帧直接落在理想增益，恒定场景下即刻满足收敛精度
                gain = raw
            else:
                # IIR 一阶低通
                candidate = previous + self.smoothing * (raw - previous)
                # 帧间相对变化有界，防止闪烁
                upper = previous * (1.0 + self.max_step)
                lower = previous * (1.0 - self.max_step)
                if candidate > upper:
                    candidate = upper
                elif candidate < lower:
                    candidate = lower
                # 增益限幅
                if candidate > self.max_gain:
                    candidate = self.max_gain
                elif candidate < self.min_gain:
                    candidate = self.min_gain
                gain = candidate

            self._gain = float(gain)
            return self._gain

    def apply(self, image):
        """按当前增益返回曝光后的新图像（不修改输入，输出裁剪到 [0, 1]）。"""
        with self._lock:
            if self._gain is None:
                self.estimate(image)
            gain = self._gain
            rows = self._rows(image)
            finite = math.isfinite

            result = []
            for row in rows:
                exposed = []
                for value in row:
                    if not finite(value):
                        raise InvalidImageError(
                            f"pixels must be finite, got {value!r}"
                        )
                    scaled = float(value) * gain
                    if scaled > 1.0:
                        scaled = 1.0
                    elif scaled < 0.0:
                        scaled = 0.0
                    exposed.append(scaled)
                result.append(exposed)
            return result

    def reset(self):
        """清空时域状态，下一次 :meth:`estimate` 按首帧处理。"""
        with self._lock:
            self._gain = None
