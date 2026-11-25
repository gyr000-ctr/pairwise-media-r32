"""whitebalance 核心 API（初始未实现）。"""


class WhiteBalance:
    def __init__(self, method='gray', trim=0.1):
        """构造"""
        raise NotImplementedError

    def gains(self, image):
        """估计增益"""
        raise NotImplementedError

    def apply(self, image):
        """应用"""
        raise NotImplementedError
