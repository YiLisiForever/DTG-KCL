"""
多尺度CNN（TextCNN）特征提取器 + 拓扑关系lstmcell
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Attention
from data.utils import generate_graph
from data.deve_formatter import get_num_classes, get_allword_len
from loss.weightsuploss import WeightSupCLoss

class LSTMDecoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMDecoder, self).__init__()
        self.feature_len = config.getint("net", "hidden_size")
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.task_name = task_name
        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        self.vec_size = config.getint("data", "vec_size")
        # CNN编码器
        self.keyword_num = get_allword_len()
        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"),
                                        (a, self.vec_size)))

        self.convs = nn.ModuleList(self.convs)

        self.outfc = []  # 为每个任务创建一个输出全连接层（nn.Linear）
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")#law,crit,time
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

        self.midfc = []  # 为每个任务创建一个中间全连接层（如果启用 more_fc 选项）
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]  # 为每个任务创建一个 LSTMCell，输入和隐藏状态维度均为 hidden_size。第一个元素为 None（占位符）
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(config.getint("net", "hidden_size"), config.getint("net", "hidden_size")))

        self.hidden_state_fc_list = []  # 二维列表，用于在不同任务间传递隐藏状态时进行线性变换
        for a in range(0, len(task_name) + 1):
            arr = []  # range(0, len(task_name) + 1)表示包含一个占位符（索引0）和所有实际任务（索引1到N）
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)  # 将一维列表转换为ModuleList（PyTorch要求）
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []  # 二维列表，用于在不同任务间传递细胞状态时进行线性变换
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.suploss = WeightSupCLoss(contrast_mode='one')
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, config, usegpu):
        self.hidden_list = []  # 其中每个元素是一个元组 (h, c)，表示一个任务的初始隐藏状态和细胞状态
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append((
                    torch.autograd.Variable(
                        torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                    torch.autograd.Variable(
                        torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda())))
            else:
                self.hidden_list.append((
                    torch.autograd.Variable(torch.zeros(config.getint("data", "batch_size"), self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(config.getint("data", "batch_size"), self.hidden_dim))))

    def forward(self, x, config):
        # 多尺度CNN（TextCNN）特征提取器
        # x --> (batch, 2 * max_len_f, 200)
        # CNN特征提
        outputs = []
        batch_size = config.getint("data", "batch_size")
        x = x.view(batch_size, 1, -1, self.vec_size)
        # x.view(batch_size, 1, -1, self.vec_size) --> (batch, 1, 2 * max_len_f, vec_size)
        batch_size, channels, seq_len, feature_dim = x.shape
        conv_out = []
        gram = config.getint("net", "min_gram")
        for conv in self.convs:
            # kernel_size: 池化窗口的大小（长度）
            y = F.relu(conv(x)).view(batch_size, config.getint("net", "filters"), -1)
            # 原：F.max_pool1d(y, kernel_size=200 - gram + 1)
            y = F.max_pool1d(y, kernel_size=seq_len - gram + 1).view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        # 拼接所有CNN特征
        # (5-2+1) * 64 == 256
        cnn_features = torch.cat(conv_out, dim=1)
        # 经过设置min_gram和max_gram还有filters使得输入数据维度符合hiddensize
        fc_input = cnn_features

        h_list = [None, fc_input, fc_input, fc_input]
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")


        for a in range(1, len(task_name) + 1):
            h = h_list[a]
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

        return outputs
