import torch
import random


class TripletSampler(object):

    def __init__(self,
                 num,
                 add_gt_as_proposals,
                 ):
        self.num = num
        self.add_gt_as_proposals = add_gt_as_proposals

    def sample(self,
               assign_result_lst,
               bboxes_lst,
               gt_bboxes_lst,
               gt_instances_lst,
               batch_size,
               ):
        sampler_bboxes_lst = []

        for i in range(batch_size):
            for j in range(3):
                if len(bboxes_lst[j][i].shape) < 2:
                    bboxes_lst[j][i] = bboxes_lst[j][i][None, :]

                bboxes_lst[j][i] = bboxes_lst[j][i][:, :4]

        for i in range(batch_size):
            # sampling for each batch
            batch_sampling_result = []

            # choose an instance both in normal and plus
            sampler_instance = random.choice(gt_instances_lst[0][i])
            loop_count = 0
            while (sampler_instance not in gt_instances_lst[2][i]) or (sampler_instance == 666):
                sampler_instance = random.choice(gt_instances_lst[0][i])
                loop_count += 1
                if loop_count > 100:
                    print(gt_instances_lst[0][i], gt_instances_lst[1][i], gt_instances_lst[2][i])
                    raise Exception

            # add gt instance bboxes
            for j in range(3):
                # loop for normal, minus and plus
                if self.add_gt_as_proposals and len(gt_instances_lst[j][i]) > 0:
                    if gt_instances_lst[j][i] is None:
                        raise ValueError(
                            'gt_labels must be given when add_gt_as_proposals is True')
                    bboxes_lst[j][i] = torch.cat([gt_bboxes_lst[j][i], bboxes_lst[j][i]], dim=0)
                    assign_result_lst[j][i].add_gt_(gt_instances_lst[j][i])

            normal_inds = torch.nonzero(assign_result_lst[0][i].labels == sampler_instance)
            normal_pre_sample = torch.Tensor([random.choice(normal_inds) for _ in range(self.num)]).squeeze().int()

            plus_inds = torch.nonzero(assign_result_lst[2][i].labels == sampler_instance)
            plus_pre_sample = torch.Tensor([random.choice(plus_inds) for _ in range(self.num)]).squeeze().int()

            minus_inds = torch.nonzero(
                (assign_result_lst[1][i].labels != sampler_instance) & (assign_result_lst[1][i].labels != 0))
            minus_pre_sample = torch.Tensor([random.choice(minus_inds) for _ in range(self.num)]).squeeze().int()

            for t in range(self.num):
                triplet = [bboxes_lst[0][i][normal_pre_sample[t]], bboxes_lst[1][i][minus_pre_sample[t]],
                           bboxes_lst[2][i][plus_pre_sample[t]]]
                batch_sampling_result.append(triplet)

            sampler_bboxes_lst.append(batch_sampling_result)

        return sampler_bboxes_lst
