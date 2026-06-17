from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
class WeightSupCLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(WeightSupCLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def trans_matrix(self, ori_mask):
        # ori_mask为-1-0-1三值矩阵，不同罪名但关键词相似为-1，不同罪名且关键词不相似为0，罪名相同为1

        # 2. 将原矩阵中的 -1 变为 0
        mask = torch.where(ori_mask == -1, torch.tensor(0), ori_mask)

        return mask,ori_mask

    def forward(self, features, labels=None, mask=None):
        # neg_weight: [bsz, bsz] 负样本权重矩阵，仅在mask=-1的位置生效,我们对不同罪名但关键词相似的位置加上1.5倍权重，让它更重要
        device = features.device
        mask, neg_weight = self.trans_matrix(mask)
        # 处理输入维度
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # 处理mask和labels
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 无监督情况：只有自身为正样本
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 有监督情况：同类为正样本
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 处理多视图
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        contrast_feature = F.normalize(contrast_feature, dim=-1)
        anchor_feature = F.normalize(anchor_feature, dim=-1)

        # 计算相似度 logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        logits = anchor_dot_contrast / self.temperature

        # 数值稳定
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 扩展mask
        mask = mask.repeat(anchor_count, contrast_count)

        # 创建logits_mask（排除自身对比）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # 应用负样本权重到分母
        if neg_weight is not None:
            # 确保neg_weight在正确的设备上
            neg_weight = neg_weight.to(device)
            # 扩展neg_weight到多视图
            neg_weight = neg_weight.repeat(anchor_count, contrast_count)
            # 只对困难负样本应用权重，其他位置权重为1
            weight_for_exp = torch.ones_like(neg_weight)  # 默认权重为1

            # 只对困难负样本增加权重
            simple_neg_mask = (neg_weight == -1) #困难负样本的位置
            weight_for_exp[simple_neg_mask] = 1.5  # 如1.5

            exp_logits = torch.exp(logits) * logits_mask * weight_for_exp
        else:
            exp_logits = torch.exp(logits) * logits_mask

        # 计算log概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本的平均log概率（mask保持0-1）
        mask_sum = mask.sum(1)
        # 避免除以0
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
