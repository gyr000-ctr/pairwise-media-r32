"""lensdistort 核心 API（初始未实现）。"""


class LensDistort:
    def __init__(self, k1, k2, p1=0.0, p2=0.0):
        """构造"""
        raise NotImplementedError

    def distort(self, u, v):
        """正向"""
        raise NotImplementedError

    def undistort(self, u, v):
        """逆向"""
        raise NotImplementedError
