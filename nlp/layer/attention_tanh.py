import torch
import torch.nn as nn
import torch.nn.functional as F
"""
传统Scaled Dot-Product是
Q = X·W_Q  # 查询
K = X·W_K  # 键
V = X·W_V  # 值
attention = softmax(Q·K^T/√d_k) · V
传统加性注意力
e_i = v^T · tanh(W·h_i + U·q)
α_i = exp(e_i) / Σ_j exp(e_j)
context = Σ α_i · h_i
简化加性注意力
e_i = tanh(W·h_i) · q  # 简化了，没有U·q和v^T
α_i = softmax(e_i)
context = Σ α_i · h_i
但在这个实现中，V同时充当了V和K的角色，用于计算相似度的特征和用于提取值的特征相同
attention = softmax(V·Q) · V
"""


class AttentionTanH(nn.Module):
    def __init__(self, config):
        super(AttentionTanH, self).__init__()
        # pass
        self.features = config.getint("net", "hidden_size")
        self.fc = nn.Linear(self.features*2, self.features*2, bias=False)
        # bias=False(通常不包括额外的偏置项，因为注意力关注的是特征之间的相对重要性)
        # 如果有偏置这可能导致即使没有相关信息也有基础注意力分数
        # 可学习的线性变换

    def forward(self, feature, hidden):
        # feature（Q） 通常是一个查询向量（query），表示当前需要关注什么,来自 解码器(Decoder)的当前状态
        # hidden（V） 通常是一个序列的表示，包含多个元素的编码信息
        feature = feature.view(feature.size(0), -1, 1)
        """
        feature.view(feature.size(0), -1, 1)
        =输入: [batch_size, hidden_size]
        =输出: [batch_size, hidden_size, 1]
        """
        hidden = torch.tanh(self.fc(hidden))
        """
        输入: [batch_size, seq_len, hidden_size]
        输出: [batch_size, seq_len, hidden_size]（维度不变）
        目的：通过线性变换 + tanh 激活，增强隐藏状态的表达能力
        """
        ratio = torch.bmm(hidden, feature)  # [batch_size, seq_len, 1] hidden·Q (K·Q的替代)
        """
        [batch_size, seq_len, hidden_size]-->[seq_len, hidden_size]
        [batch_size, hidden_size, 1]-->[hidden_size, 1]
        对于每个序列位置 i：
        ratio[i] = hidden[i] · feature（点积） --> [seq_len, hidden_size]*[hidden_size, 1]
        表示该位置与特征的相似度
        """
        ratio = ratio.view(ratio.size(0), ratio.size(1))  # [batch_size, seq_len]
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)  # 沿序列维度归一化，得到注意力权重[batch_size, seq_len, 1]
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        """
        hidden.transpose(1, 2): [batch_size, hidden_size, seq_len]
        ratio: [batch_size, seq_len, 1]
        结果: [batch_size, hidden_size, 1]
        """
        result = result.view(result.size(0), -1)

        return result