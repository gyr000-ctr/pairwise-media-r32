"""exposure 核心 API（初始未实现）。"""


class AutoExposure:
    def __init__(self, target, max_gain):
        """构造"""
        raise NotImplementedError

    def estimate(self, image):
        """估计增益"""
        raise NotImplementedError

    def apply(self, image):
        """应用"""
        raise NotImplementedError
