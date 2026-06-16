import torch
import torch.nn as nn

class XSLstm(nn.Module):

    def keep_top_k_sim(self, similarity_matrix, neigh_index, k):
        # similarity_matrix中自己与自己的相似度为-inf
        # 控制最大相似列表，就是一个案件最多有多少相似案件
        # 获取每行前k个最大值的索引

        top_k_values, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

        for i in range(similarity_matrix.shape[0]):
            if len(neigh_index[i]) > k:
                neigh_index[i] = top_k_indices[i].tolist()
        """
        for idx in range(batch_size):
            adj_matrix[idx, neigh_index[idx]] = 1
        weight_matrix = (neigh_mat * adj_matrix).float()
        在外面就通过neigh_index把每行多余50个相似案件的位置相似度置0了
        """
        return neigh_index

    def xsoperator(self, x, pre_h, temp_output, siminfo):

        batch_size, seq_len, input_size = x.shape
        neigh_mat, graph_list_1, graph_membership, neigh_index = siminfo
        new_h = torch.zeros_like(temp_output)  # [batch_size, seq_len, hidden_size]
        # 将neigh_index转换为邻接矩阵 [num_cases, num_cases]
        adj_matrix = torch.zeros((batch_size, batch_size), device=x.device)
        neigh_mat = torch.from_numpy(neigh_mat).to(x.device)
        max_sim = self.config.getint("data", "max_sim")# 每个案件的相似案件最多 max_sim 个
        neigh_index = self.keep_top_k_sim(neigh_mat, neigh_index, max_sim)# 保留每行前 max_sim 个最大值的相似度，其余置0,并返回处理后矩阵和词典

        for idx in range(batch_size):
            adj_matrix[idx, neigh_index[idx]] = 1

        # 计算相似度权重矩阵 [batch_size, batch_size]
        # 这里假设neigh_mat已经是相似度矩阵
        weight_matrix = (neigh_mat * adj_matrix).float()  # 只保留有连接的相似度
        #
        # 计算每个案件的相似案件数量 [batch_size]
        sim_count = adj_matrix.sum(dim=1)  # [batch_size]
        sim_count = torch.where(sim_count > 0, sim_count, torch.ones_like(sim_count))  # 避免除0
        """
        如果sim_count(案件数量) > 0，保持原值
        如果sim_count(案件数量) == 0，替换为1
        """
        # 整个序列的combined计算
        pre_h_reshaped = temp_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        pre_x = x.permute(1, 0, 2)
        # print(f"pre_h_reshaped{pre_h_reshaped.shape}")
        # print(f"x{x.shape}")
        combined = torch.cat([pre_h_reshaped, pre_x], dim=-1)  # [seq_len, batch_size, hidden_size+input_size]

        # 计算所有时间步的u_t
        u_t_all = torch.sigmoid(self.linear(combined))  # [seq_len, batch_size, hidden_size]
        u_t_all = u_t_all.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        # 对于每个时间步进行向量化计算
        for t in range(seq_len):
            # 获取当前时间步的temp_output [batch_size, hidden_size]
            temp_t = temp_output[:, t, :]  # [batch_size, hidden_size]
            """
            temp_output = [
            # 批次1
            [
                [h00, h01, h02],  # 时间步0
                [h10, h11, h12],  # 时间步1
                [h20, h21, h22],  # 时间步2  ← 我们要取这个
            ],
            # 批次2
            [
                [h00', h01', h02'],  # 时间步0
                [h10', h11', h12'],  # 时间步1
                [h20', h21', h22'],  # 时间步2  ← 和这个
            ]
        ]
            # 提取后得到：
            temp_output[:, t, :] = [
            [h20, h21, h22],  # 批次1的第2个时间步
            [h20', h21', h22']  # 批次2的第2个时间步
            ]
            """
            # 向量化计算相似度加权和
            # weight_matrix: [batch_size, batch_size], temp_t: [batch_size, hidden_size]
            sim_sum = torch.matmul(weight_matrix, temp_t)  # [batch_size, hidden_size]
            """
            # 创建示例数据
            weight_matrix = torch.tensor([
                [0.0, 0.8, 0.0, 0.2],  # 案件0的相似度权重
                ...
            ])
            temp_t = torch.tensor([
                [1.0, 2.0, 3.0],  # 案件0的隐藏状态
                [4.0, 5.0, 6.0],  # 案件1的隐藏状态
                [7.0, 8.0, 9.0],  # 案件2的隐藏状态
                [10.0, 11.0, 12.0],  # 案件3的隐藏状态
            ])
            现在需要0.0 * [1.0, 2.0, 3.0]加上0.8 * [4.0, 5.0, 6.0]加上0.0 * [7.0, 8.0, 9.0]加上0.2 * [10.0, 11.0, 12.0]
            即矩阵乘法matmul
            """
            # 计算相似度平均值
            simzhi = sim_sum / sim_count.unsqueeze(1)  # [batch_size, hidden_size]
            """
            unsqueeze(1) 的作用是在第1维度（索引1）增加一个维度：
            从 [batch_size] → [batch_size, 1]
            这样就能与 sim_sum 的形状 [batch_size, hidden_size] 进行广播除法
            """
            # 创建掩码：有相似案件的案件
            has_sim_mask = (sim_count > 1).float()  # [batch_size]
            """
            torch.tensor([3, 0, 5, 2])
            (sim_count > 1)变成
            tensor([True, False, True, True])
            .float()变成
            tensor([1., 0., 1., 1.])
            """
            # 创建掩码：没有相似案件的案件
            no_sim_mask = (sim_count <= 1).float()  # [batch_size]

            # 对于有相似案件的案件，使用相似度加权结果,逐元素相乘
            new_h_sim = simzhi * u_t_all[:, t, :] + pre_h[t]  # [batch_size, hidden_size]

            # 对于没有相似案件的案件，使用前一时刻的隐藏状态
            new_h_no_sim = pre_h[t]  # [batch_size, hidden_size]

            # 组合结果
            new_h_has_sim = new_h_sim * has_sim_mask.unsqueeze(1)
            """
            unsqueeze(1)广播后的 has_sim_mask = [
            [1., 1., 1.],  # 第0行: [1.] → [1., 1., 1.]
            [0., 0., 0.],  # 第1行: [0.] → [0., 0., 0.]
            [1., 1., 1.],  # 第2行: [1.] → [1., 1., 1.]
            [1., 1., 1.]   # 第3行: [1.] → [1., 1., 1.]
            ]
            """
            new_h_no_sim_part = new_h_no_sim * no_sim_mask.unsqueeze(1)

            new_h[:, t, :] = new_h_has_sim + new_h_no_sim_part

        return new_h

    def __init__(self, config, input_size, hidden_size):
        super(XSLstm, self).__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.gates = nn.ModuleList([
            # 输入相关的权重
            nn.Linear(input_size, hidden_size),  # W_ii
            nn.Linear(input_size, hidden_size),  # W_if
            nn.Linear(input_size, hidden_size),  # W_ig
            nn.Linear(input_size, hidden_size),  # W_io

            # 隐藏状态相关的权重
            nn.Linear(hidden_size, hidden_size),  # W_hi
            nn.Linear(hidden_size, hidden_size),  # W_hf
            nn.Linear(hidden_size, hidden_size),  # W_hg
            nn.Linear(hidden_size, hidden_size),  # W_ho
        ])

        self.linear = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, x, hidden_state, siminfo):
        batch_size, seq_len, input_size = x.shape
        h_t, c_t = hidden_state

        temp_outputs = []
        pre_h = []  # 保存的每一时刻的隐藏层状态
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步的输入

            # 输入门
            i_t = torch.sigmoid(self.gates[0](x_t) + self.gates[4](h_t))

            # 遗忘门
            f_t = torch.sigmoid(self.gates[1](x_t) + self.gates[5](h_t))

            # 细胞候选值
            g_t = torch.tanh(self.gates[2](x_t) + self.gates[6](h_t))

            # 输出门
            o_t = torch.sigmoid(self.gates[3](x_t) + self.gates[7](h_t))

            # 更新状态
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # 保存隐藏层状态
            pre_h.append(h_t.squeeze(0))
            temp_outputs.append(h_t.permute(1, 0, 2))  # [batch_size, 1, hidden_size]

        temp_output = torch.cat(temp_outputs, dim=1)  # 是每个案件所有时间步的隐藏表示吧[batch_size, seq_Len, hidden_size]
        # print(f"temp_output:{temp_output.shape}")

        new_h = self.xsoperator(x, pre_h, temp_output, siminfo)

        output = new_h

        return output, (new_h[:, -1, :], c_t)
