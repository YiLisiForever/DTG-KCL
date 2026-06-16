import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.task_name = task_name
        self.task_num = len(task_name)
        self.batch_size = config.getint("data", "batch_size")
        self.fuu_inter = nn.Linear(4 * self.hidden_dim, 2 * self.hidden_dim)


        # self.weight_list = nn.Parameter(torch.ones(self.batch_size, self.batch_size))  # 可学习的权重矩阵

        # 动态权重生成器：根据源节点和目标节点的特征生成边权重
        self.edge_weight_net = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0,1] 之间的权重
        )
        """"""
        # 节点更新网络
        self.node_update = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh()
        )

    def forward(self, hidden_list):
        weight_mat = []
        new_hidden_list = [None]
        for a in range(1, len(self.task_name) + 1):
            # 收集所有邻居的加权表示
            weighted_neighbors = []
            #weights_for_a = []
            for b in range(1, len(self.task_name) + 1):
                if a != b:
                    # 拼接源节点和目标节点特征
                    edge_input = torch.cat([hidden_list[a], hidden_list[b]], dim=-1)# [batch, hidden * 2]
                    """"""
                    # 方法1: 基于节点特征的动态权重
                    edge_weight = self.edge_weight_net(edge_input)  # [batch, 1]
                    weighted_neigh = edge_weight * self.fuu_inter(edge_input)  # [batch, hidden]
                    weight_mat.append(edge_weight.squeeze())
                    weighted_neighbors.append(weighted_neigh)

            stacked_neighbors = torch.stack(weighted_neighbors)  # [node_num ,batch, hidden]
            # 简单聚合
            neighbor_agg = stacked_neighbors.mean(dim=0)  # [batch, hidden]

            # 更新节点表示（残差连接）
            updated = torch.tanh(self.node_update(hidden_list[a] + neighbor_agg))

            new_hidden_list.append(updated)

        return new_hidden_list, weight_mat





