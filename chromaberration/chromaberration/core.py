"""chromaberration 核心 API（初始未实现）。"""


class ChromaAberration:
    def __init__(self, k_r, k_b):
        """构造"""
        raise NotImplementedError

    def sample(self, channel, u, v):
        """采样"""
        raise NotImplementedError

    def apply(self, rgb):
        """校正"""
        raise NotImplementedError
