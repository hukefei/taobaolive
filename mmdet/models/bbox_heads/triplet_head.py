import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
from ..utils import ConvModule


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
                 num_convs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 out_channels=1000):
        super(TripletHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_convs = True if num_convs > 0 else False
        self.num_convs = num_convs

        if self.with_convs:
            self.embedding_convs = nn.ModuleList()
            for i in range(self.num_convs):
                if i == 0:
                    in_ = self.in_channels
                else:
                    in_ = self.out_channels
                self.embedding_convs.append(ConvModule(
                        in_,
                        self.out_channels,
                        1))

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

    def init_weights(self):
        pass

    def embedding(self, x):
        if self.with_convs:
            for conv in self.embedding_convs:
                x = conv(x)
        if self.with_avg_pool:
            embedded = self.avg_pool(x)
            embedded = embedded.squeeze(-1)
            embedded = embedded.squeeze(-1)
        else:
            embedded = x
        return embedded

    def forward(self, x):
        # x should be triplet list
        embedded = []
        for i in x:
            embedded.append(self.embedding(i))

        dista = F.pairwise_distance(embedded[0], embedded[1], 2)
        distb = F.pairwise_distance(embedded[0], embedded[2], 2)

        return dista, distb, embedded

    def loss(self,
             dista,
             distb,
             embedded,
             device='cuda',
             margin=0.2,
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
        losses.setdefault('triplet_loss', loss_triplet)
        losses.setdefault('embedded_loss', loss_weight * loss_embedd)

        return losses
