# deblock

## 题面（可直接粘贴）

子目录：deblock。去块效应滤波。约束：只在块边界（8 像素网格）做滤波且按块强度自适应；块内像素变化 ≤1e-12（不模糊细节）；与逐像素参考实现一致（≤1e-9）；强度 0 时恒等；100 万像素不超过 700 ms；非法强度可区分报错。验收：cd deblock && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd deblock && python -m pytest tests -q
```

初始 15 failed；当前 `tests/test_core.py`（15 条冒烟用例）与
`tests/test_requirements.py`（约束级回归用例）全部通过。

## 实现概要（deblock/deblock/core.py）

- `Deblock.apply(image, strength)`：只在 8 像素网格的内部 seam（8, 16, ...）上滤波；
  每个 seam 只动两侧紧邻的一个像素对，块内部逐位不变。是否滤波由“跨边界阶跃
  `d` 是否大于两侧块内活动”决定（块强度自适应），因此真实纹理/平滑渐变不被抹平。
  修正量为 `0.5 * min(strength, 1) * d`，强度 0 直接返回无损拷贝（恒等）。
  逐像素参考实现在 `Deblock.reference_apply`，两者差值为浮点舍入级（≤ 1e-9）。
- `Deblock.edge_mask(w, h)`：返回 `h×w` 的 `bool` 掩码，标记所有可能被改动的像素
  （seam 两侧一格），`apply` 改过的像素一定落在掩码内。
- 报错分类：非法/负数强度抛 `InvalidStrengthError`，NaN/Inf 强度抛
  `NonFiniteStrengthError`，图像含 NaN/Inf 抛 `NonFiniteImageError`（同时是
  `InvalidImageError`），空图/残缺行/非数值抛 `InvalidImageError`/`InvalidShapeError`，
  `edge_mask` 尺寸非法抛 `InvalidSizeError`；全部是 `ValueError` 子类。
- 无实例可变状态、仅在局部数据上运算，多线程并发调用安全；有 NumPy 时走向量化路径
  （100 万像素灰度约 20–30 ms，三通道约 120–240 ms，均在 700 ms 预算内），
  无 NumPy 时自动退回纯 Python 路径且结果逐位一致。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
