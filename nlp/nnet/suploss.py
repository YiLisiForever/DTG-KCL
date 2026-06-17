import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import AttentionTanH
from data.deve_formatter import get_num_classes, get_allword_len
from loss.weightsuploss import WeightSupCLoss
from nnet.dynamicGCN import DynamicGraphConv

class SuplossLSTM(nn.Module):

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
            # 原：F.max_pool1d(y, kernel_size=200 - gram + 1)
            y = F.max_pool1d(y, kernel_size=seq_len - gram + 1).view(batch_size, -1)
            conv_out.append(y)
            gram += 1

        # 拼接所有CNN特征
        # (5-2+1) * 64 == 256
        cnn_features = torch.cat(conv_out, dim=1)
        x = x.view(batch_size, seq_len, hidden_dim)
        return cnn_features  # [batch, hidden]

    def __init__(self, config, usegpu):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        super(SuplossLSTM, self).__init__()
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
        self.pinjie_norm = nn.LayerNorm(self.vec_size)
        self.outfc = []  # 为每个任务创建一个输出全连接层（nn.Linear）

        for x in task_name:
            self.outfc.append(nn.Linear(
                features * 2, get_num_classes(x)
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
                bidirectional=True
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
        self.ronghe_full = nn.Linear(features * 3, features * 2) # (batch_size,hidden_size * 2 + hidden_size)
        self.soft = nn.Softmax(dim=-1)

        self.suploss = WeightSupCLoss(contrast_mode='one')
        self.conv_1 = DynamicGraphConv(config)
        self.conv_2 = DynamicGraphConv(config)
        """"""
        self.attention = AttentionTanH(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.lstm_list = nn.ModuleList(self.lstm_list)
        self.midfc = nn.ModuleList(self.midfc)

    def daluan(self, x):

        batch_size, seq_len, vec_size = x.shape
        num_to_shuffle = 5

        # 转置: [batch_size, vec_size, seq_len]
        x_t = x.permute(0, 2, 1)

        # 随机选择20个位置
        selected_positions = torch.randperm(seq_len, device=x.device)[:num_to_shuffle]

        # 提取这20个位置的词（在序列维度上）
        subset = x_t[:, :, selected_positions]  # [batch_size, vec_size, 20]

        # 打乱这20个位置
        shuffled_order = torch.randperm(num_to_shuffle, device=x.device)
        shuffled_subset = subset[:, :, shuffled_order]

        # 放回原位置
        x_t_shuffled = x_t.clone()
        x_t_shuffled[:, :, selected_positions] = shuffled_subset

        # 转置回原形状
        return x_t_shuffled.permute(0, 2, 1)
    def init_hidden(self, config, usegpu):
        self.hidden_list = []#(hn)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append(
                    torch.autograd.Variable(
                        torch.zeros(2 * config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                    )
            else:
                self.hidden_list.append(
                    torch.autograd.Variable(torch.zeros(2 * config.getint("data", "batch_size"), self.hidden_dim)))

    def forward(self, x, config, keywords_rep, key_gold, crit_label, crit_mask):
        # x.shape: (batch_size, seq_len, vec_size)
        # siminfo = get_case_graph(config, x)
        # key_gold (batch_size, allword_len)
        # crit_label (batch_size)
        # crit_mask (batch_size, batch_size)
        mat_for_tsne = None
        conv_weight = None
        outputs = []
        current_input = x
        current_input = self.input_norm(current_input)
        shuffled_x = self.daluan(current_input)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        # law--crit--time的全连接图
        """"""
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
        current_input = torch.cat([current_input, shuffled_x], dim=0)  # [2*batch, seq_len, vec_size]
        current_input = self.pinjie_norm(current_input)
        for a in range(1, len(task_name)+1):  # 遍历每个任务（跳过占位符0）
            output, (hn, cn) = self.lstm_list[a](current_input)

            forward_h = hn[-2]  # [batch, hidden_size]
            backward_h = hn[-1]  # [batch, hidden_size]
            forback_h = torch.cat([forward_h, backward_h], dim=-1)  # [batch, hidden_size*2]
            """"""
            hn = self.attention(forback_h, output) #(batch, hidden * 2)
            self.hidden_list[a] = hn
# -------------------二层动态任务卷积-------------------*
        new_hn, _ = self.conv_1(self.hidden_list)
        new_hn_2, conv_weight = self.conv_2(new_hn)
# ----------------------卷积结束------------------------*
# ----------------------将两个batch拆出来处理------------------------*
        batch_size = x.size(0)
        shuffle_new_hn_2 = [None]
        original_new_hn_2 = [None]
        for a in range(1, len(task_name)+1):
            shuffle_new_hn_2.append(new_hn_2[a][batch_size:])
            original_new_hn_2.append(new_hn_2[a][:batch_size])

# ----------------------拆出来处理------------------------*
# ----------------------对比学习------------------------*
        new_list = []
        for a in range(1, len(task_name) + 1):
            if self.task_name[a - 1] == "crit":
                hn = original_new_hn_2[a]
                shuff_hn = shuffle_new_hn_2[a]
                new_list.append(hn)
                new_list.append(shuff_hn)
            """
            if self.task_name[a-1] == "crit":
                new_list.append(hn)
                new_list.append(hn)
            """
        contrast_loss = self.suploss(torch.stack(new_list, dim=1), None, crit_mask)
# ----------------------词汇表示融合-------------------------*
        """"""
        for a in range(1, len(task_name) + 1):
            hn = original_new_hn_2[a]
            if self.task_name[a-1] == "crit":
                mat_for_tsne = hn
                hn = self.ronghe_full(torch.cat([hn, word_rep], dim=1))
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](hn))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](hn).view(config.getint("data", "batch_size"), -1))

        return outputs, loss, contrast_loss, mat_for_tsne, conv_weight

