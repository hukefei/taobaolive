import random, json
from .coco import CocoDataset
import logging
import os.path as osp
import tempfile
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.utils import print_log
from .registry import DATASETS
from .evaluator import load_coco_bboxes, Evaluator


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
        if self.gallery_img_path is not None:
            self.gallery_img_infos = self.load_gallery_annotations(self.gallery_ann_file)
        if self.gallery_ann_file is not None:
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
        self.img_ids2 = self.coco2.getImgIds()
        img_infos = []
        for i in self.img_ids2:
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

    def gallery_pre_pipeline(self, results):
        results['img_prefix'] = self.gallery_img_path
        results['seg_prefix'] = []
        results['proposal_file'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_img(self, idx):
        # TODO: find out a new sampling method instead of random sampling
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        instance_ids = list(set(ann_info['instances']))
        instance_ids = [idx for idx in instance_ids if idx > 0]
        if instance_ids == []:
            return {}
        instance_id = random.choice(instance_ids)
        loop = 0
        while instance_id not in self.gallery_instance_dict.keys():
            instance_id = random.choice(instance_ids)
            loop += 1
            if loop >= 100:
                print(instance_ids)
                raise Exception

        # get positive example
        pos_img_id = random.choice(self.gallery_instance_dict[instance_id])
        pos_img_info = self.gallery_img_infos[pos_img_id - 1]
        pos_ann_info = self.get_gallery_ann_info(pos_img_id - 1)
        results_pos = dict(img_info=pos_img_info, ann_info=pos_ann_info)
        self.gallery_pre_pipeline(results_pos)
        results_pos = self.pipeline(results_pos)

        # get negative example
        neg_id = random.choice(list(self.gallery_instance_dict.keys()))
        while neg_id in pos_ann_info['instances']:
            neg_id = random.choice(list(self.gallery_instance_dict.keys()))
        neg_img_id = random.choice(self.gallery_instance_dict[neg_id])
        neg_img_info = self.gallery_img_infos[neg_img_id - 1]
        neg_ann_info = self.get_gallery_ann_info(neg_img_id - 1)
        results_neg = dict(img_info=neg_img_info, ann_info=neg_ann_info)
        self.gallery_pre_pipeline(results_neg)
        results_neg = self.pipeline(results_neg)

        triplet = {}
        # triplet = [results, results_neg, results_pos]
        for k, v in results.items():
            triplet.setdefault(k, []).append(v)
            triplet[k].append(results_neg[k])
            triplet[k].append(results_pos[k])
        return triplet

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 iou_thr_by_class=0.2,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        bbox_results = []
        embedding_results = []
        for res in results:
            bbox_results.append(res[0])
            embedding_results.append(res[1])

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(bbox_results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    bbox_results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    gt_lst = load_coco_bboxes(cocoGt, is_gt=True)
                    dt_lst = load_coco_bboxes(cocoDt, is_gt=False)
                    evaluator = Evaluator()
                    ret, mAP = evaluator.GetMAPbyClass(
                        gt_lst,
                        dt_lst,
                        method='EveryPointInterpolation',
                        iou_thr=iou_thr_by_class
                    )
                    # Get metric values per each class
                    for metricsPerClass in ret:
                        cl = metricsPerClass['class']
                        ap = metricsPerClass['AP']
                        ap_str = '{0:.3f}'.format(ap)
                        eval_results['class_{}'.format(cl)] = float(ap_str)
                        print('AP: %s (%s)' % (ap_str, cl))
                    mAP_str = '{0:.3f}'.format(mAP)
                    eval_results['mAP'] = float(mAP_str)
                    print('mAP: {}\n'.format(mAP_str))
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
