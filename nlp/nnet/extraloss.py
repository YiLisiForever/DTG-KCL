import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import AttentionTanH
from data.utils import generate_graph
from data.deve_formatter import get_num_classes, get_allword_len
from data.case_processed import get_case_graph

class ExtralossLSTM(nn.Module):

    def textcnn(self, config, x):
        conv_out = []
        gram = config.getint("net", "min_gram")
        # print(x.shape)
        batch_size, seq_len, hidden_dim = x.shape
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
        return cnn_features  # [batch, hidden]

    def __init__(self, config, usegpu):
        super(ExtralossLSTM, self).__init__()
        self.usegpu = usegpu
        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.task_name = task_name
        self.num_layers = len(task_name)
        self.vec_size = config.getint("data", "vec_size")
        self.keyword_num = get_allword_len()
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
                batch_first=True,  # 输入形状为(batch, seq_len, vec_size)
            ))
        self.input_con = nn.Linear(self.vec_size,features)
        # 词库编码器
        self.word_encoder_for = nn.LSTMCell(
                input_size=self.vec_size,
                hidden_size=features,
            )
        self.word_encoder_back = nn.LSTMCell(
            input_size=self.vec_size,
            hidden_size=features,
            )
        # 模拟双向
        self.keyword_full = nn.Linear(features * 2, features)
        self.fact_select = nn.Linear(features, self.keyword_num)
        self.ronghe_full = nn.Linear(features * 2, features)
        self.soft = nn.Softmax(dim=-1)
        # 事实(batch_size,seq_len,vec_size) --textcnn--> (batch_size,hidden_size)
        # 将事实映射到选词空间，并与标准答案进行loss计算，一个案件可以对应多个词，loss传到外面进行学习
        # 然后用softmax对事实映射（batch，keyword_num）进行概率化
        # 然后用概率（即权重）（batch，keyword_num）乘以词库编码(keyword_num, hidden_size)后的张量得到辅助词表示(batch, hidden_size)
        # 最后把辅助词表示与卷积处理的案件表示进行linear操作
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
    def forward(self, x, config, keywords_rep, key_gold):
        # x.shape: (batch_size, seq_len, vec_size)
        # siminfo = get_case_graph(config, x)
        # key_gold (batch_size, allword_len)
        outputs = []
        current_input = x
        current_input = self.input_norm(current_input)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        # law--crit--time的全连接图
# ----------------------软选择词汇表示-------------------------*
        keywords_rep_forw = self.word_encoder_for(keywords_rep)[0]  # [allword_num,vec_size]-->[allword_num,hidden]
        keywords_rep_back = self.word_encoder_back(torch.flip(keywords_rep, dims=[0]))[0] # [allword_num,vec_size]-->[allword_num,hidden]
        keywords_rep_backward = torch.flip(keywords_rep_back, dims=[0])  # 对齐
        keywords_rep = self.keyword_full(torch.cat([keywords_rep_forw, keywords_rep_backward], dim=1))  # [allword_num,hidden * 2]-->[allword_num,hidden]

        change_x = self.textcnn(config, self.input_con(x))  # (batch_size, seq_len, vec_size) --> (batch_size, hidden)
        logit_x = self.fact_select(change_x)  # (batch_size, hidden) --> (batch_size, allword_len)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logit_x, key_gold.float())
        word_weight = self.soft(logit_x)  # (batch_size, allword_len)
        word_rep = word_weight @ keywords_rep  # (batch_size, allword_len) * (allword_num,hidden)=(batch_size, hidden)
# ----------------------软选择词汇表示结束-------------------------*
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
            #h_conv = self.hidden_list[a] + h_conv  # 残差连接
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
            #h_conv = new_hn[a] + h_conv  # 残差连接
            new_hn_2.append(h_conv)
# ----------------------卷积结束-------------------------*
# ----------------------词汇表示融合-------------------------*
        for a in range(1, len(task_name) + 1):
            hn = new_hn_2[a]
            if self.task_name[a-1] == "crit":
                hn = self.ronghe_full(torch.cat([hn, word_rep], dim=1))
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](hn))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](hn).view(config.getint("data", "batch_size"), -1))

        return outputs, loss

