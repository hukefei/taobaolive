import os
import cv2
import numpy as np
import random
from ..registry import PIPELINES


@PIPELINES.register_module
class Concat(object):
    """Concat five images.
    Args:
        img_path (str): images should be placed in one path.
    """

    def __init__(self, img_path):
        self.img_path = img_path

    def __call__(self, results):
        filename = results['img_info']['filename']
        image = []
        for i in range(5):
            img_name = filename.replace('_0.jpg', '_{}.jpg'.format(str(i)))
            img = cv2.imread(os.path.join(self.img_path, img_name), 0)
            img = img[..., np.newaxis]
            image.append(img)
        concat = np.concatenate(image, axis=-1)
        results['img'] = concat
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_path={})'.format(
            self.img_path)
        return repr_str


@PIPELINES.register_module
class ConcatV2(object):
    """Concat five images in a different way
    Args:
        img_path (str): images should be placed in one path.
        main_pattern: 0 for gray, 1 for bgr
        by_order: if True, align other images by order(from 0 to 4), otherwise random align the images
    """

    def __init__(self, img_path, main_pattern=0, by_order=True):
        self.img_path = img_path
        assert main_pattern in (0, 1)
        self.main_pattern = main_pattern
        self.by_order = by_order

    def __call__(self, results):
        filename = results['img_info']['filename']
        ind = int(filename[-5])
        image = []
        other_images = []
        # add main channels
        if self.main_pattern == 0:
            img = cv2.imread(os.path.join(self.img_path, filename), 0)
            img = img[..., np.newaxis]
            image.append(img)
        else:
            img = cv2.imread(os.path.join(self.img_path, filename))
            image.append(img)
        # add other channels
        for i in range(5):
            if i == ind:
                continue
            img_name = filename.replace('_{}.jpg'.format(str(ind)), '_{}.jpg'.format(str(i)))
            img = cv2.imread(os.path.join(self.img_path, img_name), 0)
            img = img[..., np.newaxis]
            other_images.append(img)
        # merge main image and other images
        if not self.by_order:
            random.shuffle(other_images)
        image.extend(other_images)
        concat = np.concatenate(image, axis=-1)
        results['img'] = concat
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_path={}, main_pattern={}, by_order={})'.format(
            self.img_path, self.main_pattern, self.by_order)
        return repr_str

@PIPELINES.register_module
class Concat_DDJ(object):
    """Concat five images in a different way
    Args:
        img_path (str): images should be placed in one path.
        main_pattern: 0 for gray, 1 for bgr
        by_order: order of ddj patterns
    """

    def __init__(self, img_path, main_pattern=1, by_order=('BLUE', 'GREEN', 'RED', 'WHITE')):
        self.img_path = img_path
        assert main_pattern in (0, 1)
        self.main_pattern = main_pattern
        self.by_order = by_order

    def __call__(self, results):
        filename = results['img_info']['filename']
        image = []

        for pattern in self.by_order:
            img_name = filename.replace('WHITE', pattern)
            img = cv2.imread(os.path.join(self.img_path, img_name), self.main_pattern)
            if self.main_pattern == 0:
                img = img[..., np.newaxis]
            image.append(img)

        concat = np.concatenate(image, axis=-1)
        results['img'] = concat
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_path={}, main_pattern={}, by_order={})'.format(
            self.img_path, self.main_pattern, self.by_order)
        return repr_str