import torch
import torch.nn as nn
import torch.nn.functional as F
from .xslstm import XSLstm
from layer import AttentionTanH
from data.utils import generate_graph
from data.deve_formatter import get_num_classes
from data.case_processed import get_case_graph

class stackedLSTM(nn.Module):
    def textcnn(self, config, x):
        conv_out = []
        gram = config.getint("net", "min_gram")
        # print(x.shape)
        batch_size, seq_len, hidden_dim = x.shape  # 这里的x是所有时刻的隐藏层输出
        # print(x.shape)
        x = x.view(batch_size, 1, -1, hidden_dim)
        # print(x.shape)
        for conv in self.convs:
            # kernel_size: 池化窗口的大小（长度）
            y = F.relu(conv(x)).view(batch_size, config.getint("net", "filters"), -1)
            """
            conv(x) 输出形状: (batch_size, filters, new_seq_len, 1)
            view后形状: (batch_size, filters, new_seq_len)
            """
            # 原：F.max_pool1d(y, kernel_size=200 - gram + 1)
            y = F.max_pool1d(y, kernel_size=seq_len - gram + 1).view(batch_size, -1)
            """
            因为 池化核大小kernel_size, 整个序列被一个池化窗口覆盖, 每个通道（filter）的输出是一个最大值
            池化后形状: (batch_size, filters, 1)
            view(batch_size, -1)展平后形状: (batch_size, filters)
            """
            conv_out.append(y)
            gram += 1

        # 拼接所有CNN特征
        # (5-2+1) * 64 == 256
        cnn_features = torch.cat(conv_out, dim=1)
        x = x.view(batch_size, seq_len, hidden_dim)
        return cnn_features.view(1, batch_size, -1) #[1, batch, hidden]

    def __init__(self, config, usegpu):
        super(stackedLSTM, self).__init__()
        self.usegpu = usegpu
        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.num_layers = len(task_name)
        self.vec_size = config.getint("data", "vec_size")

        # CNN编码器
        self.convs = []
        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"),
                                        (a, self.hidden_dim)))
        self.convs = nn.ModuleList(self.convs)

        # 添加输入归一化层
        self.input_norm = nn.LayerNorm(self.vec_size)
        self.outfc = []  # 为每个任务创建一个输出全连接层（nn.Linear）

        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))
        self.midfc = []  # 为每个任务创建一个中间全连接层（如果启用 more_fc 选项）
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))
        """
        self.lstm_list = [None]  # 为每个任务创建一个LSTM，输入为vec_size隐藏状态维度为 hidden_size。第一个元素为 None（占位符）
        for x in range(len(task_name)):
            self.lstm_list.append(nn.LSTM(
                input_size=self.vec_size,
                hidden_size=features,
                num_layers=1,  # 单层LSTM
                batch_first=True,  # 输入形状为(batch, seq_len, features)
            ))

        """
        self.lstm_list = [None]  # 为每个任务创建一个xsLSTM，输入为vec_size隐藏状态维度为 hidden_size。第一个元素为 None（占位符）
        for x in range(len(task_name)):
            self.lstm_list.append(XSLstm(config, self.vec_size, features))

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
        """"""
        self.attention = AttentionTanH(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.lstm_list = nn.ModuleList(self.lstm_list)
        self.midfc = nn.ModuleList(self.midfc)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, config, usegpu):
        self.hidden_list = []#其中每个元素是一个 h，表示一个任务的初始隐藏状态
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append((
                    torch.autograd.Variable(
                        torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                    torch.autograd.Variable(
                        torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda())))
            else:
                self.hidden_list.append((
                    torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim))))
    def forward(self, x, config):
        # x.shape: (batch_size, seq_len, hidden_size)
        siminfo = get_case_graph(config, x)
        outputs = []
        current_input = x
        current_input = self.input_norm(current_input)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = generate_graph(config)  # 生成任务依赖图（邻接矩阵）
        first = []  # 标记每个任务是否是第一次被更新,first[b] = True 表示任务 b 的隐藏状态尚未被任何其他任务更新过
        for a in range(0, len(task_name)+1):
            first.append(True)

        for a in range(1, len(task_name)+1):  # 遍历每个任务（跳过占位符0）
            change_h = self.hidden_list[a][0].clone()   #(1, batch, hidden)
            change_c = self.hidden_list[a][1].clone() #(1, batch, hidden)
            output, (hn, cn) = self.lstm_list[a](current_input, (change_h, change_c), siminfo)
            # output, (hn, cn) = self.lstm_list[a](current_input, (change_h, change_c))
            fc_change = self.textcnn(config, output) #(1, batch, hidden)
            fc_change = F.layer_norm(fc_change, [self.hidden_dim])
            # print(fc_change.shape)
            cn = F.layer_norm(cn, [self.hidden_dim])

            for b in range(1, len(task_name)+1):
                if graph[a][b]:#如果任务a影响任务b
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = fc_change, cn
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](fc_change)
                        cp = cp + self.cell_state_fc_list[a][b](cn)
                    self.hidden_list[b] = (hp, cp)

            ch_hn = hn.squeeze(0)  # (1, batch, hidden)-->(batch, hidden)
            hn = self.attention(ch_hn, output)

            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](hn))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](hn).view(config.getint("data", "batch_size"), -1))

        return outputs

