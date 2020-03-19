import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable


from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy
from mmdet.models.registry import HEADS


@HEADS.register_module
class TripletHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256):
        super(TripletHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

    def init_weights(self):
        pass

    def forward(self, x):
        # x should be triplet list
        embedded = []
        if self.with_avg_pool:
            for i in x:
                embedded.append(self.avg_pool(i))
        else:
            embedded = x
        dista = F.pairwise_distance(embedded[0], embedded[1], 2)
        distb = F.pairwise_distance(embedded[0], embedded[2], 2)

        return dista, distb, embedded

    def loss(self,
             dista,
             distb,
             embedded,
             device='cuda',
             margin=0,
             loss_weight=0.001):
        losses = dict()
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if device == 'cuda':
            target = target.cuda()
        target = Variable(target)
        criterion = torch.nn.MarginRankingLoss(margin=margin)
        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded[0].norm(2) + embedded[1].norm(2) + embedded[2].norm(2)
        loss = loss_triplet + loss_weight * loss_embedd
        losses.setdefault('triplet_loss', loss)

        return losses