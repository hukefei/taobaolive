import os, random
from .coco_cq import CocoDataset_CQ
from .registry import DATASETS

import torch
from .pipelines import Compose
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class CocoDataset_pair(CocoDataset_CQ):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 pg_normal_path=None,
                 ps_normal_path=None,
                 normal_pipeline=None):
        super(CocoDataset_pair, self).__init__(ann_file,
                                               pipeline,
                                               data_root,
                                               img_prefix,
                                               seg_prefix,
                                               proposal_file,
                                               test_mode,
                                               filter_empty_gt)
        self.pg_normal_path = pg_normal_path
        self.ps_normal_path = ps_normal_path
        self.pg_normal_imgs = []
        self.ps_normal_imgs = []
        for lists in os.listdir(pg_normal_path):
            self.pg_normal_imgs.append(lists)
        for lists in os.listdir(ps_normal_path):
            self.ps_normal_imgs.append(lists)
        self.normal_pipeline = Compose(normal_pipeline)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        # random choice a normal img according to original shape
        ori_h, ori_w, = img_info['height'], img_info['width']
        if ori_h < 1000:
            name = random.choice(self.pg_normal_imgs)
            img_prefix = self.pg_normal_path
        else:
            name = random.choice(self.ps_normal_imgs)
            img_prefix = self.ps_normal_path
        img_info_ = {}
        img_info_['file_name'] = name
        img_info_['filename'] = name
        img_info_['height'] = ori_h
        img_info_['width'] = ori_w
        results_normal = dict(img_info=img_info_)
        results_normal['scale'] = results['img_meta'].data['scale_factor']
        results_normal['flip'] = results['img_meta'].data['flip']
        results_normal['img_prefix'] = img_prefix
        results_normal = self.normal_pipeline(results_normal)

        results['img'] = DC(torch.cat((results['img'].data, results_normal['img'].data)), stack=True)
        return results
