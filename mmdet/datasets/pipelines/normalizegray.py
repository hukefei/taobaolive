from ..registry import PIPELINES
import numpy as np

@PIPELINES.register_module
class NormalizeGray(object):
    """Normalize the gray image with n channels.
    Args:
        mean (sequence or float): Mean values of n channels.
        std (sequence or float): Std values of n channels.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        results['img'] = (results['img'] - self.mean) / self.std
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={})'.format(
            self.mean, self.std)
        return repr_str