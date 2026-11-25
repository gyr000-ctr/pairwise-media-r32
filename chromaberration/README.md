# chromaberration

## 题面（可直接粘贴）

子目录：chromaberration。色差（横向）校正。约束：按径向多项式缩放 R/B 通道（系数可配）；中心不位移（≤1e-12）；校正后通道对齐误差 ≤0.5 像素（用合成网格校验）；边界采样 clamp；100 万像素不超过 700 ms；非法系数可区分报错。验收：cd chromaberration && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd chromaberration && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
