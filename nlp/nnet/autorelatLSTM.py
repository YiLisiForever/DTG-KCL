import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import AttentionTanH
from data.utils import generate_graph
from data.deve_formatter import get_num_classes
from data.case_processed import get_case_graph

class AutoRelatLSTM(nn.Module):

    def __init__(self, config, usegpu):
        super(AutoRelatLSTM, self).__init__()
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
        """
        self.Full_inter_1 = nn.Linear(features * 2, features)
        self.Full_inter_2 = nn.Linear(features, features)

        self.attention = AttentionTanH(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.lstm_list = nn.ModuleList(self.lstm_list)
        self.midfc = nn.ModuleList(self.midfc)


    def init_hidden(self, config, usegpu):
        self.hidden_list = []#(hn)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append(
                    torch.autograd.Variable(
                        torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                    )
            else:
                self.hidden_list.append(
                    torch.autograd.Variable(torch.zeros(config.getint("data", "batch_size"), self.hidden_dim)))
    def forward(self, x, config):
        # x.shape: (batch_size, seq_len, hidden_size)
        # siminfo = get_case_graph(config, x)
        outputs = []
        current_input = x
        current_input = self.input_norm(current_input)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        # law--crit--time的全连接图

        for a in range(1, len(task_name)+1):  # 遍历每个任务（跳过占位符0）
            output, (hn, cn) = self.lstm_list[a](current_input)
            #fc_change = F.layer_norm(fc_change, [self.hidden_dim])
            ch_hn = hn.squeeze(0)  # (1, batch, hidden)-->(batch, hidden)
            hn = self.attention(ch_hn, output)
            self.hidden_list[a] = hn
# ----------------------第一层任务卷积-------------------------*
        new_hn = [None, ]
        for a in range(1, len(task_name) + 1):
            neig_rep_list = []
            for b in range(1, len(task_name) + 1):
                if a != b:
                    neig_rep = torch.cat([self.hidden_list[a], self.hidden_list[b]], dim=-1)  # (batch, 2 * hidden)
                    neig_rep = self.Full_inter_1(neig_rep)  # (batch, hidden)
                    neig_rep_list.append(neig_rep)
            stacked = torch.stack(neig_rep_list, dim=0)  # [len(list), batch, hidden]
            mean_rep = torch.mean(stacked, dim=0)  # [batch, hidden]
            h_conv = torch.tanh(self.Full_inter_2(self.hidden_list[a] + mean_rep))
            # h_conv = self.hidden_list[a] + 0.5 * h_conv  # 残差连接
            new_hn.append(h_conv)
# ----------------------第二层任务卷积-------------------------*
        new_hn_2 = [None, ]
        for a in range(1, len(task_name) + 1):
            neig_rep_list = []
            for b in range(1, len(task_name) + 1):
                if a != b:
                    neig_rep = torch.cat([new_hn[a], new_hn[b]], dim=-1)  # (batch, 2 * hidden)
                    neig_rep = self.Full_inter_1(neig_rep)  # (batch, hidden)
                    neig_rep_list.append(neig_rep)
            stacked = torch.stack(neig_rep_list, dim=0)  # [len(list), batch, hidden]
            mean_rep = torch.mean(stacked, dim=0)  # [batch, hidden]
            h_conv = torch.tanh(self.Full_inter_2(new_hn[a] + mean_rep))
            # h_conv = new_hn[a] + 0.5 * h_conv  # 残差连接
            new_hn_2.append(h_conv)
# ----------------------卷积结束-------------------------*
        for a in range(1, len(task_name) + 1):
            hn = new_hn_2[a]
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](hn))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](hn).view(config.getint("data", "batch_size"), -1))

        return outputs

