from ..registry import DETECTORS
from .two_stage_triplet import TwoStageDetector_Triplet


@DETECTORS.register_module
class FasterRCNN_Triplet(TwoStageDetector_Triplet):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 embedding_head,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN_Triplet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            embedding_head=embedding_head,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
