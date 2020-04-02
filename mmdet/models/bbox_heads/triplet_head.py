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
                 roi_feat_size=7,
                 roi_channels=256,
                 feat_dim=1024,
                 embed_dim=256):
        super(TripletHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
            in_ = roi_channels
        else:
            in_ = roi_channels * self.roi_feat_size[0] * self.roi_feat_size[1]
        self.fc1 = nn.Linear(in_, feat_dim)
        self.fc2 = nn.Linear(feat_dim, embed_dim)

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)

    def l2_norm(self, x):
        if len(x.shape):
            x = x.reshape((x.shape[0], -1))
        return F.normalize(x, p=2, dim=1)

    def embedding(self, x):
        if self.with_avg_pool:
            embedded = self.avg_pool(x)
            embedded = embedded.squeeze(-1)
            embedded = embedded.squeeze(-1)
        else:
            embedded = x
        embedded = embedded.view(embedded.size(0), -1)
        embedded = self.fc1(embedded)
        embedded = self.fc2(embedded)
        embedded = self.l2_norm(embedded)
        return embedded

    def forward(self, a, n, p):
        # x should be triplet list
        a_feat = self.embedding(a)
        n_feat = self.embedding(n)
        p_feat = self.embedding(p)

        dan = F.pairwise_distance(a_feat, n_feat, 2)
        dap = F.pairwise_distance(a_feat, p_feat, 2)

        return dan, dap

    def loss(self,
             dan,
             dap,
             device='cuda',
             margin=0.2,
             loss_weight=1):
        losses = dict()
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dan.size()).fill_(1)
        if device == 'cuda':
            target = target.cuda()
        criterion = torch.nn.MarginRankingLoss(margin=margin)
        loss_triplet = criterion(dan, dap, target) * loss_weight
        losses.setdefault('triplet_loss', loss_triplet)

        return losses
