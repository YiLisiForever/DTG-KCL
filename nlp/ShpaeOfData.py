"""
过采样次数（同时可以检查标签出现次数）
检查每个batch内张量相似度
"""
import os
import json
import torch
import pandas as pd
from data.makedata import BatchReader
from data.makedata import init_transformer
#from data.makedata import init_transformer
from data.deve_formatter import parse, check, get_time_id, get_crit_id, get_law_id, gen_crit_rep, gen_putong_tensor, gen_sim_keyworsds,get_crit_dict
from data.case_processed import get_case_graph
from data.Word2vec import word2vec  # 导入类
from config_parser import ConfigParser
from data.deve_formatter import init, get_name
import joblib
from data.deal_fact import init_thulac
# from data.deve_formatter import get_num_classes
import math


tfidf_model = None
transformer = None

def init_tfidf(config, file):
    global tfidf_model
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    # max_doc = int(config.get("data", "max_tdidfdoc"))

    # 初始化（无需提前分词）

    try:
        tfidf_file = os.path.join(data_path, "train_tfidf_model.joblib")
        tfidf_model = joblib.load(tfidf_file)
    except FileNotFoundError:
        print("训练TF-IDF模型文件不存在")
        tfidf_model = None


def calculate_duplication_times(duplicate_list):

    # 2. 找到最大样本数
    max_count = max(duplicate_list.values())

    # 3. 计算每个类别的重复次数（向shang取整）
    duplication_times_dict = {}
    for category, count in duplicate_list.items():
        duplication_times_dict[category] = math.ceil(max_count / count) - 1

    return duplication_times_dict

def create_dataset(config, ffile):
    duplicate_list = {"crit": {}, "law1": {}, "time": {}}
    # 初始化词向量转换器（只需初始化一次）
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    transformer = word2vec(os.path.join(data_path, config.get("data", "word_dic"))
                           , os.path.join(data_path, config.get("data", "vec_path")))
    #file_path = os.path.join(data_path, config.get("data", "train_path"))#训练文件地址（绝对路径）
    init_tfidf(config, ffile)

    with open(ffile, 'r', encoding='utf-8') as file:
        cun = []
        count_sum = 0
        content = file.read()
        # print(content), 每个json对象都被读成1行
        # 使用分隔符将多个 JSON 对象分开
        # 因为不是规范json格式
        json_objects = content.strip().split('\n')
        zongdata = [json.loads(obj) for obj in json_objects]
        for data in zongdata:
            if check(data, config):
                count_sum += 1
                print(count_sum)
                tensor, shape_tensor, label = parse(data, config, transformer, tfidf_model)
                #if tensor is None:
                #    continue  # 跳过此次循环

                id1 = get_law_id(data["meta"]["relevant_articles"])
                id2 = get_crit_id(data["meta"]["accusation"])
                id3 = get_time_id(data["meta"]["term_of_imprisonment"])
                """
                # 想获取原始值的话用这个
                id1 = data["meta"]["relevant_articles"][0]
                id2 = data["meta"]["accusation"][0]
                id3 = data["meta"]["term_of_imprisonment"]["imprisonment"]
                """
                duplicate_list["law1"][id1] = duplicate_list["law1"].get(id1, 0) + 1
                duplicate_list["crit"][id2] = duplicate_list["crit"].get(id2, 0) + 1
                duplicate_list["time"][id3] = duplicate_list["time"].get(id3, 0) + 1
        print("Loading " + str(count_sum) + " data from " + ffile + " end.")

    return duplicate_list

def check_sim(config):
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    threshold = float(config.get("data", "threshold"))
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    train_file = os.path.join(data_path, config.get("data", "train_data"))
    #file_path = os.path.join(data_path, config.get("data", "train_path"))#训练文件地址（绝对路径）
    init_tfidf(config, train_file)
    transformer = word2vec(os.path.join(data_path, config.get("data", "word_dic"))
                           , os.path.join(data_path, config.get("data", "vec_path")))
    dataset = BatchReader(config, train_file, True, tfidf_model, True, 42)
    count1 = 0
    while True:
        if count1 == 50:
            break
        data = dataset.fetch_data()
        if data is None:
            break
        count1 += 1
        inputs = data[0]  # 展平张量
        """
        print(inputs.size())
        
        labels = data[2]  # 标签
        neigh_mat, graph_list_1, graph_membership, neigh_index = get_case_graph(config, inputs)
        print(len(neigh_index))
        count = (neigh_mat > threshold).sum()
        print([x for x in list(neigh_index.items()) if len(x[1]) >= 1])
        """


def count_similar_crimes(binary_matrix):
    crit_dict = get_crit_dict()
    inverted = {}
    for key, value in crit_dict.items():
        inverted[value] = key
    for index in range(binary_matrix.shape[0]):
        print("*" * 40)
        indices = torch.where(binary_matrix[index] == 1)[0]
        if len(indices) > 1:
            print(inverted[index])
            for x in indices:
                print(inverted[x.item()], end='    ')
        print("*" * 40)




"""
def init_transformer(config):
    global transformer
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    transformer = word2vec(os.path.join(data_path, config.get("data", "word_dic"))
                           , os.path.join(data_path, config.get("data", "vec_path")))

    print("Transformer init done")
"""
""""""
if __name__ == "__main__":

    config = ConfigParser()
    #init(config)
    #init_thulac(config)
    #init_transformer(config)
    cnt = 0
    tail_charge = []
    with open("C:\\Users\\ADMIN\\Desktop\\big\\crit.txt", "r", encoding="utf-8") as f:
        for line in f:
            data = line[:-1].split(" ")
            name = data[0]
            num = int(data[1])
            if 100 < num < 1200:
                cnt += 1
                tail_charge.append([name,num])
        print(cnt)
        df = pd.DataFrame(tail_charge, columns=["charge",'num'])
        df.to_excel("output_unm.xlsx", index=False)
                # charge_num[name] = num
    #gen_crit_rep(config)
    # check_sim(config)
    # check_sim(config, "C:\\Users\\ADMIN\\Desktop\\datasss\\train_json.json")
    """
    duplicate_list = create_dataset(config, "C:\\Users\\ADMIN\\Desktop\\datasss\\train_json.json")
    
    duplicate_list["law1"] = calculate_duplication_times(duplicate_list["law1"])
    duplicate_list["crit"] = calculate_duplication_times(duplicate_list["crit"])
    duplicate_list["time"] = calculate_duplication_times(duplicate_list["time"])
    # 1. 将每个子字典转换为 DataFrame
    df_law = pd.DataFrame(list(duplicate_list["law1"].items()), columns=['法条', '重复次数'])
    df_crit = pd.DataFrame(list(duplicate_list["crit"].items()), columns=['罪名', '重复次数'])
    df_time = pd.DataFrame(list(duplicate_list["time"].items()), columns=['刑期', '重复次数'])

    # 2. 保存到同一个 Excel 文件的不同工作表
    with pd.ExcelWriter('案件统计.xlsx') as writer:
        df_law.to_excel(writer, sheet_name='法条统计', index=False)
        df_crit.to_excel(writer, sheet_name='罪名统计', index=False)
        df_time.to_excel(writer, sheet_name='刑期统计', index=False)

    print("Excel 文件已生成：案件统计.xlsx")
    """
