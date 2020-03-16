import random, json
import numpy as np
from .coco import CocoDataset
from .registry import DATASETS
from pycocotools.coco import COCO


@DATASETS.register_module
class CocoDataset_triplet(CocoDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 gallery_img_path=None,
                 gallery_ann_file=None):
        super(CocoDataset_triplet, self).__init__(ann_file,
                                                  pipeline,
                                                  data_root,
                                                  img_prefix,
                                                  seg_prefix,
                                                  proposal_file,
                                                  test_mode,
                                                  filter_empty_gt)
        self.gallery_img_path = gallery_img_path
        self.gallery_ann_file = gallery_ann_file
        self.gallery_img_infos = self.load_gallery_annotations(self.gallery_ann_file)
        self.gallery_instance_dict = self.get_gallery_instances(self.gallery_ann_file)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def load_gallery_annotations(self, ann_file):
        self.coco2 = COCO(ann_file)
        self.cat_ids = self.coco2.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco2.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco2.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_gallery_ann_info(self, idx):
        img_id = self.gallery_img_infos[idx]['id']
        ann_ids = self.coco2.getAnnIds(imgIds=[img_id])
        ann_info = self.coco2.loadAnns(ann_ids)
        return self._parse_ann_info(self.gallery_img_infos[idx], ann_info)

    def get_gallery_instances(self, ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        instance_dict = {}
        for anno in annotations:
            img_id = anno['image_id']
            instance_id = anno['instance_id']
            if instance_id > 0:
                instance_dict.setdefault(instance_id, []).append(img_id)
        return instance_dict

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_instances = []
        gt_viewpoints = []
        gt_displays = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_instances.append(ann['instance_id'])
                gt_viewpoints.append(ann['viewpoint'])
                gt_displays.append(ann['display'])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_instances = np.array(gt_instances, dtype=np.int64)
            gt_viewpoints = np.array(gt_viewpoints, dtype=np.int64)
            gt_displays = np.array(gt_displays, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_instances = np.array([], dtype=np.int64)
            gt_viewpoints = np.array([], dtype=np.int64)
            gt_displays = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            instances=gt_instances,
            viewpoints=gt_viewpoints,
            displays=gt_displays,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        instance_ids = list(set(ann_info['instances']))
        instance_ids = [idx for idx in instance_ids if idx > 0]
        instance_id = random.choice(instance_ids)

        # get positive example
        pos_img_id = random.choice(self.gallery_instance_dict[instance_id])
        pos_img_info = self.gallery_img_infos[pos_img_id - 1]
        pos_ann_info = self.get_gallery_ann_info(pos_img_id - 1)
        results_pos = dict(img_info=pos_img_info, ann_info=pos_ann_info)
        results_pos['scale'] = results['img_meta'].data['scale_factor']
        results_pos['flip'] = results['img_meta'].data['flip']
        results_pos['img_prefix'] = self.gallery_img_path
        results_pos = self.pipeline(results_pos)

        # get negative example
        neg_id = random.choice(list(self.gallery_instance_dict.keys()))
        while neg_id == instance_id:
            neg_id = random.choice(list(self.gallery_instance_dict.keys()))
        neg_img_id = random.choice(self.gallery_instance_dict[neg_id])
        neg_img_info = self.gallery_img_infos[neg_img_id - 1]
        neg_ann_info = self.get_gallery_ann_info(neg_img_id - 1)
        results_neg = dict(img_info=neg_img_info, ann_info=neg_ann_info)
        results_neg['scale'] = results['img_meta'].data['scale_factor']
        results_neg['flip'] = results['img_meta'].data['flip']
        results_neg['img_prefix'] = self.gallery_img_path
        results_neg = self.pipeline(results_neg)

        triplet = [results, results_neg, results_pos]
        return triplet
