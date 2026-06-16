import numpy as np
import torch
import torch.nn.functional as F

def calcu_sim(x,y):
    # return global_cosine_similarity(x, y)
    return global_euclidean_distance(x, y)+global_cosine_similarity(x, y)

#到时候可以在cos距离前乘以权重
#主要函数1
def calculate_tensor_similarities(batch_tensor):
    """返回相似矩阵"""
    tensor_list = []
    t_len = batch_tensor.shape[0]
    sim_mat = np.zeros((t_len, t_len))
    for i in range(t_len):
        tensor_list.append(batch_tensor[i])  # 一个矩阵

    for i in range(t_len):
        for j in range(i + 1, t_len):  # 避免重复计算和自比较
            similarity = calcu_sim(tensor_list[i], tensor_list[j])
            sim_mat[i][j] = similarity
    sim_mat = sim_mat + sim_mat.T
    # 将自己与自己的相似度置为最小
    for i in range(t_len):
        sim_mat[i][i] = float('-100')
    return sim_mat, t_len
#主要函数2
def group_generation(neigh_index):
    graph = []# 存储所有连通分量
    items = []
    graph_ship = {}# 节点到组ID的映射
    for i in range(len(neigh_index)):
        if len(neigh_index[i]) == 0:
            graph.append([i])
        else:
            if neigh_index[i][0] in items:
                continue
            else:
                sub_graph = neigh_index[i]
                finding = neigh_index[i]
                exchange = []
                for j in finding:
                    exchange += neigh_index[j]
                exchange = list(set(exchange))  # 去重
                finding = exchange
                exchange = []
                while (set(sub_graph) >= set(finding)) is False:
                    sub_graph = list(set(sub_graph).union(set(finding)))
                    for j in finding:
                        exchange += neigh_index[j]
                    exchange = list(set(exchange))
                    finding = exchange
                    exchange = []
                graph.append(sub_graph)
                items += sub_graph
    for i in range(len(graph)):
        graph_1 = {j: i for j in graph[i]}
        graph_ship.update(graph_1)
    graph_ship = sorted(graph_ship.items())

    return graph, graph_ship
"""
graph:
示例: [[0,1,2], [3,4], [5]]
表示三个连通分量
graph_ship:
示例: [(0,0), (1,0), (2,0), (3,1), (4,1), (5,2)]
表示每个节点所属的组ID
"""
"""
neigh_index = np.where(neigh_mat != 0)
(
    array([0, 1, 1, 2]),  # 行索引（i）
    array([1, 0, 2, 1])   # 列索引（j）
)
判定相似的阈值threshold
law_1 = list(zip(*law_1))
*law_1 - 解包操作，相当于将 law_1 的每个子列表作为单独参数传递给 zip()
zip() - 并行迭代多个可迭代对象，每次从每个可迭代对象中取一个元素组成元组
list() - 将 zip 对象转换为列表
law_1 = [
    [index1, index2, ...],        # 所有条文的序号
    [charge1, charge2, ...],      # 所有条文的罪名
    [context1, context2, ...],    # 所有条文的文本内容
    [n_words1, n_words2, ...]     # 所有条文的词数统计
]

"""
#以下函数生成全局变量
def get_case_graph(config,tuple_list):
    neigh_mat, t_len = calculate_tensor_similarities(tuple_list)
    # print(neigh_mat)
    threshold = float(config.get("data", "threshold"))
    neigh_index = {}
    for i in range(t_len):
            neigh_index[i] = []
    rows, cols = np.where(neigh_mat > threshold)
    for i, j in zip(rows, cols):
        if i != j:  # 排除自环
            neigh_index[i].append(int(j))
    """
    neigh_index = np.where(neigh_mat > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(t_len) if (i, j) in neigh_index and j != i] for i in range(t_len)}
    """
    graph_list_1, graph_membership = group_generation(neigh_index)
    return neigh_mat, graph_list_1, graph_membership, neigh_index

def get_xiangsi(config, tuple_list):
    neigh_mat, graph_list_1, graph_membership, neigh_index = get_case_graph(config, tuple_list)
    threshold = float(config.get("data", "threshold"))
    simzhi = {}  # 改为字典
    rows, cols = neigh_mat.shape
    for key, value in neigh_index.items():
        if len(value) != 0:
            sim = []
            for i in range(cols):
                if i in value and neigh_mat[key][i] > threshold:
                    sim.append(neigh_mat[key][i])
            simzhi[key] = (sum(sim) / len(sim)) if sim else 0  # 添加空列表检查
        else:
            simzhi[key] = 0
    return simzhi  # 返回字典：案件序号 → 相似值

#[[0,1,2], [3,4], [5]]
#[(0,0), (1,0), (2,0), (3,1), (4,1), (5,2)]
"""
{
    0: [1, 5, 10],   # 条文0的邻居是1,5,10
    1: [0, 7],        # 条文1的邻居是0,7
    ...,
    102: [50, 80]     # 条文102的邻居是50,80
}
"""
def global_euclidean_distance(x, y):
    dis_matrix = torch.cdist(x, y, p=2)
    mean_distance = dis_matrix.mean()  # 平均距离
    norm = F.sigmoid(mean_distance)
    return norm.item()


def global_cosine_similarity(x, y):
    result = F.cosine_similarity(x, y, dim=1)
    mean_cos = result.mean()
    return mean_cos
