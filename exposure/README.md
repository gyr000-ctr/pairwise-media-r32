# exposure

## 题面（可直接粘贴）

子目录：exposure。自动曝光。约束：按分位（非均值）估计亮度以避免高光主导；曝光增益使中位亮度收敛到目标（≤1%）；迭代收敛且单调；增益限幅（不得过曝）；100 万像素不超过 500 ms；非法目标可区分报错。验收：cd exposure && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd exposure && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
