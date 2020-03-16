from __future__ import division
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--work_dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--resume_from', help='the checkpoint file to resume from')
#     parser.add_argument(
#         '--validate',
#         action='store_true',
#         help='whether to evaluate the checkpoint during training')
#     parser.add_argument(
#         '--gpus',
#         type=int,
#         default=1,
#         help='number of gpus to use '
#         '(only applicable to non-distributed training)')
#     parser.add_argument('--seed', type=int, default=None, help='random seed')
#     parser.add_argument(
#         '--deterministic',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument(
#         '--autoscale-lr',
#         action='store_true',
#         help='automatically scale lr with the number of gpus')
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#
#     return args

def main(config,
         work_dir=None,
         resume_from=None,
         validate=False,
         gpus=1,
         seed=None,
         launcher='none',
         local_rank=0,
         autoscale_lr=False,
         deterministic=False):

    cfg = Config.fromfile(config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if work_dir is not None:
        cfg.work_dir = work_dir
    if resume_from is not None:
        cfg.resume_from = resume_from
    cfg.gpus = gpus

    if autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            seed, deterministic))
        set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    config = r'/data/sdv2/cq/mmdet_cq/CQ_cfg/ROUND2_dyy/test_pair_train.py'
    main(config)