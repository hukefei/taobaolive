import torch

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def random_choice_pair(self, gallery_1, gallery_2, num):
        """
        :param gallery_1: image ind with gt
        :param gallery_2: image ind without gt
        :param num: sample num
        :return:
        """
        assert len(gallery_1) + len(gallery_2) >= num

        is_tensor_1 = isinstance(gallery_1, torch.Tensor)
        if not is_tensor_1:
            gallery_1 = torch.tensor(
                gallery_1, dtype=torch.long, device=torch.cuda.current_device())

        is_tensor_2 = isinstance(gallery_2, torch.Tensor)
        if not is_tensor_2:
            gallery_2 = torch.tensor(
                gallery_2, dtype=torch.long, device=torch.cuda.current_device())

        num_1 = int(num * self.pair_fraction)
        num_1 = min(len(gallery_1), num_1)
        perm_1 = torch.randperm(gallery_1.numel(), device=gallery_1.device)[:num_1]
        rand_inds_1 = gallery_1[perm_1]
        if not is_tensor_1:
            rand_inds_1 = rand_inds_1.cpu().numpy()

        num_2 = num - num_1
        num_2 = min(len(gallery_2), num_2)
        perm_2 = torch.randperm(gallery_2.numel(), device=gallery_2.device)[:num_2]
        rand_inds_2 = gallery_2[perm_2]
        if not is_tensor_2:
            rand_inds_2 = rand_inds_2.cpu().numpy()

        return rand_inds_1, rand_inds_2

    def _sample_pos_pair(self,
                         assign_result,
                         num_expected,
                         **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg_pair(self,
                         assign_result_train,
                         assign_result_normal,
                         num_expected,
                         **kwargs):
        neg_inds_train = torch.nonzero(assign_result_train.gt_inds == 0)
        neg_inds_normal = torch.nonzero(assign_result_normal.gt_inds == 0)
        if neg_inds_train.numel() != 0:
            neg_inds_train = neg_inds_train.squeeze(1)
        if neg_inds_normal.numel() != 0:
            neg_inds_normal = neg_inds_normal.squeeze(1)
        if len(neg_inds_train) + len(neg_inds_normal) <= num_expected:
            return neg_inds_train, neg_inds_normal
        else:
            return self.random_choice_pair(neg_inds_train, neg_inds_normal, num_expected)
