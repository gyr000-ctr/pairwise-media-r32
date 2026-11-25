"""deblock 核心 API（初始未实现）。"""


class Deblock:
    def apply(self, image, strength):
        """滤波"""
        raise NotImplementedError

    def edge_mask(self, w, h):
        """边界掩码"""
        raise NotImplementedError
