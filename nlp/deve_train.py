import torch
import os
from nnet import stackedLSTM
#from nnet import AutoRelatLSTM
from nnet import LSTMDecoder
from nnet import MPBFN
from nnet import ExtralossLSTM
#from nnet import AttGcnLSTM
from nnet import SuplossLSTM
from data.makedata import init_dataset, init_transformer
from deve_work import train_file
from data.utils import print_info
from config_parser import ConfigParser
from data.deve_formatter import init
from data.deal_fact import init_thulac

# 换data_train数据集同时也要换law和crit文件,还有预训练的tfidf模型

config = ConfigParser()
init(config)
init_thulac(config)
init_transformer(config)
train_dataset, test_dataset = init_dataset(config)

usegpu = True
print_info("Building net...")

print(config.getfloat("data", "max_len_f"))
print(config.getfloat("train", "epoch"))
print(config.getfloat("train", "learning_rate"))
#net = ExtralossLSTM(config, usegpu)
net = SuplossLSTM(config, usegpu)
#net = AttGcnLSTM(config, usegpu)
#net = stackedLSTM(config, usegpu)
#net = LSTMDecoder(config, usegpu)
#total_params = sum(p.numel() for p in net.parameters())
#print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# 上次没训完的可以继续,同时也得改配置中的训练起点
try:
    net.load_state_dict(
        torch.load(
            os.path.join(config.get("output", "model_path"), config.get("output", "model_name"),
                         "model-" + config.get("train", "pre_train") + ".pkl")))
except Exception as e:
    print(e)
""""""
if torch.cuda.is_available() and usegpu:
    net = net.cuda()

print_info("Net building done.")

train_file(net, train_dataset, test_dataset, usegpu, config)

print_info("Training completed successfully.")
