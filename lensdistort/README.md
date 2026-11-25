# lensdistort

## 题面（可直接粘贴）

子目录：lensdistort。镜头畸变校正（Brown 模型）。约束：径向 k1/k2 与切向 p1/p2 全参数；正逆变换往返 ≤1e-6（牛顿迭代）；中心不动；校正后直线弯曲 ≤0.5 像素（合成网格校验）；100 万像素不超过 800 ms；非法参数可区分报错。验收：cd lensdistort && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd lensdistort && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
