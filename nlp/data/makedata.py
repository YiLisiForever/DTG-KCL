import json
import random
import torch
import os
import joblib
from data.deve_formatter import parse, check, get_time_id, get_crit_id, get_law_id, get_allword_rep, calculate_mask, get_crit_word_sim
from .Word2vec import word2vec



train_tfidf_model = None
test_tfidf_model = None
transformer = None
duplicate_list = {"crit": {},
                  "law1": {},
                  "time": {}
                  }  # 启用过采样的话请赋值,不赋值即默认不采用过采样

def init_transformer(config):
    global transformer
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    transformer = word2vec(os.path.join(data_path, config.get("data", "word_dic"))
                           , os.path.join(data_path, config.get("data", "vec_path")))

    print("Transformer init done")

def init_tfidf(config, file, istrain):
    global train_tfidf_model, test_tfidf_model
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    # max_doc = int(config.get("data", "max_tdidfdoc"))

    # 初始化（无需提前分词）
    if istrain:
        try:
            tfidf_file = os.path.join(data_path, "train_tfidf_model.joblib")
            train_tfidf_model = joblib.load(tfidf_file)
        except FileNotFoundError:
            print("训练TF-IDF模型文件不存在")
            train_tfidf_model = None
    else:
        try:
            tfidf_file = os.path.join(data_path, "test_tfidf_model.joblib")
            test_tfidf_model = joblib.load(tfidf_file)
        except FileNotFoundError:
            print("测试TF-IDF模型文件不存在")
            test_tfidf_model = None

class BatchReader:
    def __init__(self, config, file_path, train, tfidf, shuffle=False, seed=None):
        self.cnt_batch = 0
        self.train = train
        self.config = config
        self.file_path = file_path
        self.batch_size = config.getint("data", "batch_size")
        self.shuffle = shuffle
        self.seed = seed
        self.tfidf_model = tfidf
        # 初始化读取状态
        self.data = None
        self.current_index = 0
        self.total_items = 0
        # 加载JSON数据
        self._load_data()

    def _load_data(self):
        """加载JSON文件,初始时会打乱"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                """
                content = file.read()
                json_objects = content.strip().split('\n')
                zongdata = [json.loads(obj) for obj in json_objects]
                """
                zongdata = json.load(file) #处理后的数据采用标准json，不需要特殊处理

                self.data = zongdata
            if self.shuffle:
                if self.seed is not None:
                    random.seed(self.seed)
                random.shuffle(self.data)
            self.total_items = len(self.data)
            self.current_index = 0

        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {self.file_path} 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"文件 {self.file_path} 不是有效的JSON格式")

    def fetch_data(self):
        batch_data = []

        # 循环读取数据，直到凑够一个batch或读取完所有数据
        while len(batch_data) < self.batch_size and self.current_index < self.total_items:
            item = self.data[self.current_index]
            self.current_index += 1

            # 进行合理性检验
            if check(item, self.config):
                duplicate_time = 1
                tupleee = parse(item, self.config, transformer, self.tfidf_model)
                if tupleee[0] is None:
                    continue
                if self.train:
                    """
                    id1 = get_law_id(item["meta"]["relevant_articles"])
                    id2 = get_crit_id(item["meta"]["accusation"])
                    id3 = get_time_id(item["meta"]["term_of_imprisonment"])
                    if id1 in duplicate_list["law1"].keys():
                        duplicate_time += duplicate_list["law1"][id1]
                    if id2 in duplicate_list["crit"].keys():
                        duplicate_time += duplicate_list["crit"][id2]
                    if id3 in duplicate_list["time"].keys():
                        duplicate_time += duplicate_list["time"][id3]
                    """
                    while len(batch_data) < self.batch_size and duplicate_time > 0:
                        duplicate_time -= 1
                        batch_data.append(tupleee)
                else:
                    batch_data.append(tupleee)
        # 打乱batch内的数据顺序
        if self.train:
            random.shuffle(batch_data)
        # 检查是否凑够了完整的batch
        if len(batch_data) < self.batch_size:
            # 如果剩余数据凑不够一个batch，返回None
            return None
        # tensor, shape_tensor, torch.cat(label), crit_rep, gold_matrix, accusation_dict[data["meta"]["accusation"]]
        if batch_data and isinstance(batch_data[0], tuple):
            self.cnt_batch += 1
            """
            gen_tensor
            解包并堆叠张量数据
            item[0] -->(2, max_len_f, 200)
            item[0][0] --> (max_len_f, 200)
            torch.cat([item[0][0], item[0][1]], dim=0) --> (2 * max_len_f, 200)

            batch_tensors = torch.stack([torch.cat([item[0][0], item[0][1]], dim=0) for item in batch_data])
            """
            batch_tensors = torch.stack([item[0] for item in batch_data])  # gen_putong_tensor
            batch_shape_tensors = [item[1] for item in batch_data]
            # 一个batch的标签矩阵
            batch_labels = torch.stack([item[2] for item in batch_data])
            # 所有的关键词表示-->[allword_num,vec_size]
            batch_crit_rep = get_allword_rep()
            # 一个batch的标准onehot张量[batch,keywords_len]
            batch_gold = torch.stack([item[3] for item in batch_data])
            # 一个batch的罪名索引
            batch_crit_index = torch.stack([item[4] for item in batch_data])

            #返回相同0-1矩阵【batch-size，batch-size】，把罪名不同或关键词相似度高的作为不相同类别矩阵(值为0)
            batch_mask = calculate_mask(batch_crit_index, get_crit_word_sim())

            return batch_tensors, batch_shape_tensors, batch_labels, batch_crit_rep, batch_gold, batch_crit_index, batch_mask

    def reset(self, shuf):
        """重置读取位置到文件开头"""
        if shuf:
            random.shuffle(self.data)
        self.current_index = 0


def create_dataset(config, file_path, train, tfidf, shuffle=False, seed=None):
    return BatchReader(config, file_path, train, tfidf, shuffle, seed)


def init_train_dataset(config):
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    train_file = os.path.join(data_path, config.get("data", "train_data"))  # 应该是单个文件路径
    print("init train tfidf")
    init_tfidf(config, train_file, True)
    print("init train tfidf down")
    return create_dataset(config, train_file, True, train_tfidf_model, True, 42)


def init_test_dataset(config):
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    test_file = os.path.join(data_path, config.get("data", "test_data"))  # 应该是单个文件路径
    print("init test tfidf")
    init_tfidf(config, test_file, False)
    print("init test tfidf down")
    return create_dataset(config, test_file, False, test_tfidf_model, False, 42)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)
    return train_dataset, test_dataset

