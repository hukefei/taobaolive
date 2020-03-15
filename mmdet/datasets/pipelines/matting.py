import cv2
import glob
import mmcv
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class Matting(object):
    """Matting defect bbox to normal image.
    Args:
        normal_path (str): normal images path
        matting_ratio (float): Whether to keep the original defect image.
        blend_range (tuple[float]): (min_ratio, max_ratio)
        mode (str): Either "fixed" or "random". If "fixed", cut the defect
                    area to the same position in the normal image.
    """

    def __init__(self,
                 normal_path,
                 matting_ratio=0.5,
                 blend_range=None,
                 mode='fixed'):
        self.normal_path = normal_path
        self.matting_ratio = matting_ratio
        self.normal_imgs = glob.glob(normal_path + '*.jpg')

        if blend_range is None:
            self.blend_ratio = 1
        else:
            assert len(blend_range) == 2
            self.blend_ratio = np.random.uniform(blend_range[0], blend_range[1])

        assert mode in ['random', 'fixed']
        self.mode = mode

    def __call__(self, results):
        if 'matting' not in results:
            matting = True if np.random.rand() < self.matting_ratio else False
            results['matting'] = matting
        if results['matting']:
            ind = np.random.randint(len(self.normal_imgs))
            normal_name = self.normal_imgs[ind]
            normal_img = mmcv.imread(normal_name)
            normal_img = mmcv.imresize_like(normal_img, results['img'])
            gt_bboxes_ = []
            for bbox in results['gt_bboxes']:
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                if self.mode == 'fixed':
                    normal_img[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                        results['img'][ymin:ymax, xmin:xmax, :], self.blend_ratio,
                        normal_img[ymin:ymax, xmin:xmax, :], 1 - self.blend_ratio, 0)
                    gt_bboxes_.append(bbox)
                elif self.mode == 'random':
                    h, w = ymax - ymin, xmax - xmin
                    img_h, img_w, _ = results['img_shape']
                    xmin = np.random.randint(0, img_w - w)
                    ymin = np.random.randint(0, img_h - h)
                    xmax = xmin + w
                    ymax = ymin + h
                    normal_img[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                        results['img'][ymin:ymax, xmin:xmax, :], self.blend_ratio,
                        normal_img[ymin:ymax, xmin:xmax, :], 1 - self.blend_ratio, 0)
                    gt_bboxes_.append([xmin, ymin, xmax, ymax])
                else:
                    raise NotImplementedError
            results['img'] = normal_img
            results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(normal_path={}, matting_ratio={}, ' \
                    'blend_ratio={}, mode={})'.format(self.normal_path,
                                                      self.matting_ratio,
                                                      self.blend_ratio,
                                                      self.mode)
        return repr_str
