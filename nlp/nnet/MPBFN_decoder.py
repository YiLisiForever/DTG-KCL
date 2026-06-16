import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from data.deve_formatter import get_num_classes

class MPBFN(nn.Module):
    def __init__(self, config, use_gpu):
        super(MPBFN, self).__init__()

        self.feature_len = config.getint("net", "hidden_size")
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.task_name = task_name
        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        self.vec_size = config.getint("data", "vec_size")

        self.outfc = []  # 为每个任务创建一个输出全连接层（nn.Linear）
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")  # law,crit,time
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

    def init_hidden(self,config, usegpu):
        pass
    def forward(self, x, config):

        return x