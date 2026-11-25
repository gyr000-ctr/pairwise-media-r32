# whitebalance

## 题面（可直接粘贴）

子目录：whitebalance。白平衡估计（灰世界/白块）。约束：灰世界与白块两种估计可分别调用；估计增益归一为行列式 1 或均值为 1（≤1e-9）；对纯灰图增益为 1；必须用鲁棒统计（截断分位）而非全均值；100 万像素不超过 600 ms；非法分位可区分报错。验收：cd whitebalance && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd whitebalance && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
