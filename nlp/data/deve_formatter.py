# coding=utf-8
import torch
#from case_processed import get_case_graph
from .deal_fact import *
#from data.loader import get_num_classes
import os
import pickle
import torch.nn.functional as F

#建立罪名和法条名称到索引的映射,还有停用词以及关键词文件为全局变量------------------------------------开始
all_word_tensor = None
keywords_index = {}  # 关键词：索引
crit_keyword = {}  # 罪名：关键词字典
all_keywords = []  # 所有关键词
#charge_num = {} # 罪名：num字典
crit_words_sim = None
accusation_list = []
accusation_dict = {}
law_list = []
law_dict = {}
manywords = []
reducewords = []
stopwords = []
SENTENCE_SPLITTER = re.compile(r'[。！？；…]')
def init(config):#从文件加载罪名和法条数据，并过滤低频项
    global accusation_list, accusation_dict, law_list, law_dict, manywords, reducewords, stopwords, crit_keyword, all_keywords, all_word_tensor, crit_words_sim  # 声明所有要修改的全局变量
    #all_documents = []
    min_frequency = config.getint("data", "min_frequency")
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    cnt1 = 0
    with open(os.path.join(data_path, "crit.txt"), "r", encoding="utf-8") as f:
        for line in f:
            data = line[:-1].split(" ")
            name = data[0]
            num = int(data[1])
            if num > min_frequency:
                cnt1 += num
                accusation_list.append(name)
                accusation_dict[name] = len(accusation_list) - 1
                #charge_num[name] = num

    cnt2 = 0
    sum_dict = {}  # 临时存储法条编号对应的频次总和
    with open(os.path.join(data_path, "law.txt"), "r", encoding="utf-8") as f:
        # 拼接数据文件路径，格式为 {data_path}/{dataset}/law.txt
        # 累加相同法条编号的频次
        for line in f:
            data = line[:-1].split(" ")
            name = int(data[0])
            num = int(data[1])
            sum_dict[name] = num
        for law_id, total_num in sum_dict.items():
            if total_num > min_frequency:
                cnt2 += total_num
                law_list.append(law_id)
                law_dict[law_id] = len(law_list) - 1
    with open(os.path.join(data_path, "many_keyword.txt"), 'r', encoding='utf-8') as file:
        manywords = [line.strip() for line in file.readlines()]
    with open(os.path.join(data_path, "reduce_keyword.txt"), 'r', encoding='utf-8') as file:
        reducewords = [line.strip() for line in file.readlines()]
    with open(os.path.join(data_path, "stop_word.txt"), 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())  # 读取停用词并保存为集合
    with open(os.path.join(data_path, "crit_keywords.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 遍历每一行
        for line in lines:
            line = line.strip()  # 去掉首尾空白
            if '。' in line:  # 按中文句号分割罪名和解释
                parts = line.split('。', 1)  # 只分割第一个句号
                crime = parts[0].strip()
                keywords = parts[1].strip() if len(parts) > 1 else ''
                split_word = keywords.split('，')
                # print(len(split_word))
                for x in split_word:
                    if x not in keywords_index:
                        all_keywords.append(x)
                        keywords_index[x] = len(all_keywords) - 1
                crit_keyword[crime] = split_word
        quchong = set(all_keywords)
        all_keywords = list(quchong)
    all_word_tensor = gen_crit_rep(config)
    print("load crit_words_sim_begin")
    crit_words_sim = gen_sim_keyworsds(config)
    print("load crit_words_sim_finish")
    print("manywords:"+str(len(manywords)))
    print("reducewords:"+str(len(reducewords)))
    print("stopwords:"+str(len(stopwords)))
    print("accusation_list:"+str(len(accusation_list)), cnt1)
    print(accusation_list)
    print("law_list:"+str(len(law_list)), cnt2)
    print(law_list)
    print("crit_keyword:"+str(len(crit_keyword)))
    print("all_keywords:" + str(len(all_keywords)))
    print(all_keywords)
    print(keywords_index)
#建立罪名和法条名称到索引的映射,还有停用词以及关键词文件为全局变量------------------------------------结束
def get_num_classes(s):#获得法条，罪名，刑期类别数
    if s == "crit":
        return len(accusation_list)
    if s == "law":
        return len(law_list)
    if s == "time":
        return 11


def get_name(s, num):#获得法条，罪名，刑期对应名字
    if s == "crit":
        return accusation_list[num]
    if s == "law":
        return law_list[num]
    if s == "time":
        map_list = {
            0: "死刑或无期",
            1: "十年以上",
            2: "七到十年",
            3: "五到七年",
            4: "三到五年",
            5: "二到三年",
            6: "一到二年",
            7: "九到十二个月",
            8: "六到九个月",
            9: "零到六个月",
            10: "没事"
        }

        return map_list[num]

#检查数据有效性并生成标准数据------------------------------------开始
#只选单指控案件
def check_crit(data):#传入值是罪名名称
    cnt = 0
    for x in data:
        if x in accusation_dict.keys():
            cnt += 1
        else:
            return False
    return cnt == 1
    #return cnt >= 1


def check_law(data):#传入值是法条号
    valid_ids = {int(x) for x in data
                 if 102 <= int(x) <= 452 and int(x) in law_dict}
    return len(valid_ids) == 1
    #return len(valid_ids) >= 1


def get_crit_id(data):
    for x in data:
        if x in accusation_dict.keys():
            return accusation_dict[x]
"""
从输入数据 data 中查找第一个匹配的罪名ID。
使用全局字典 accusation_dict 进行罪名匹配。
"""
def get_law_id(data):
    for x in data:
        if x in law_dict.keys():
            return law_dict[x]

"""
从输入数据 data 中查找第一个匹配的法条ID。
使用全局字典 law_dict 进行法条匹配。
"""
"""
life_imprisonment无期false
death_penalty死刑false
imprisonmen:number
"""
def get_time_id(data):#传入的是["term"]["term_of_imprisonment"]
    wuqi = data["life_imprisonment"]
    sixing =data["death_penalty"]
    zhikong =data["imprisonment"]#month
    if wuqi or sixing:
        return 0
    elif zhikong > 120:
        return 1
    elif zhikong > 84:
        return 2
    elif zhikong > 60:
        return 3
    elif zhikong > 36:
        return 4
    elif zhikong > 24:
        return 5
    elif zhikong > 12:
        return 6
    elif zhikong > 9:
        return 7
    elif zhikong > 6:
        return 8
    elif zhikong > 0:
        return 9
    else:
        return 10


#将原始数据中的罪名、法条和刑期信息转换为多标签的one-hot编码形式
def analyze_crit(data):#data得是列表
    res = torch.from_numpy(np.zeros(get_num_classes("crit")))
    for x in data:
        if x in accusation_dict.keys():
            res[accusation_dict[x]] = 1
    return res


def analyze_law(data):
    res = torch.from_numpy(np.zeros(get_num_classes("law")))
    for x in data:
        if int(x) in law_dict.keys():
            res[law_dict[int(x)]] = 1
    return res


def analyze_time(data):
    res = torch.from_numpy(np.zeros(get_num_classes("time")))
    opt = get_time_id(data)
    res[opt] = 1
    return res

def gen_sentence(data):#传入整个文章，返回句子列表
    slist = [sentences for sentences in re.split(r'[，。！？；]', data)]
    return slist

def check_sentence(data, config):#传进来的data是句子列表
    if len(data) > config.getint("data", "sentence_num"):
        return False
    for x in data:
        if len(x) > config.getint("data", "sentence_len"):
            return False
    return True

def gen_tensor(data, config, transformer, tfidf_model):#传入一整个fact文章生成张量
    vector_fact =[]
    vector_special=[]
    max_len_f = int(config.get("data","max_len_f"))  # tf-idf提取出的最大词数，即最多词汇数
    #max_len_f = 100
    cut_temp = []#一般事实分词
    cut_temp_s = []#特殊事实分词
    fact = []#一般事实句子
    special = []#特殊情况句子
    # 筛选句子阶段开始
    for sentences in re.split(SENTENCE_SPLITTER, data):
        flag = True
        # print(sentences)
        for word in manywords:
            if (word in sentences):
                special.append(sentences)
                flag = False
        for word in reducewords:
            if (word in sentences):
                special.append(sentences)
                flag = False
        if (flag == True):
            fact.append(sentences)
    # 筛选句子阶段结束，得到fact和special
    # 分词阶段开始

    for sentence in fact:
        cut_temp.extend(filter_and_tokenize(config, sentence, stopwords))  # 分词，去除停用词，数字，字母。。。

    if len(cut_temp) > max_len_f:
        top_words_per_document = tfidf(cut_temp, max_len_f, tfidf_model)  # 词数多的话，获得tf-idf觉得每篇文章重要的词构成的列表(一般事实)
    else:
        top_words_per_document = putong_cut(cut_temp, max_len_f)  # 词数少就直接用

    # print("top_words_per_document", len(top_words_per_document))
    if len(top_words_per_document) == 0:
        #print("tfidf没识别到common模块中的词汇")
        return None, None
    """"""
    for word in top_words_per_document:
        vector_fact.append(transformer.load(word))
    fillvector_fact = fill_list_with_average(vector_fact, max_len_f)# 填充向量平均值

    if(len(special)!=0):
        for sentence in special:
            cut_temp_s.extend(filter_and_tokenize_s(config, sentence, stopwords))  # 分词，去除停用词，数字，字母
        if len(cut_temp_s) > max_len_f:
            top_words_per_special = tfidf(cut_temp_s, max_len_f, tfidf_model)  # 获得tf-idf觉得每篇文章重要的词构成的列表(特殊情况)
        else:
            top_words_per_special = putong_cut(cut_temp_s, max_len_f)  # 词数少就直接用

        if len(top_words_per_special) == 0:
            # print("tfidf没识别到special模块中的词汇")
            return None, None
        # print("top_words_per_special", top_words_per_special)
        """"""
        for word in top_words_per_special:
            vector_special.append(transformer.load(word))
        fillvector_special = fill_list_with_average(vector_special, max_len_f)  # 填充向量平均值
    else:
        fillvector_special = fill_list_with_average_s(vector_fact, max_len_f)  # 将空的特殊模块填充一般事实的向量平均值
        # fillvector_special = fillvector_fact# 将空的特殊模块填充一般事实向量

    vector_np = torch.tensor(np.array(fillvector_fact))
    vector_np_s = torch.tensor(np.array(fillvector_special))
    tensor_new = torch.stack([vector_np, vector_np_s], dim=0)
    #tensor_new.shape -->(2, max_len_f, 200)
    #print("shape1", tensor_new.shape)
    return tensor_new, tensor_new.shape
"""
def gen_ciju_tensor(data, config, transformer):
    all_vec = []
    max_len_f = int(config.get("data", "max_len_f"))  # tf-idf提取出的最大词数，即最多词汇数
    max_sen_len = int(config.get("data", "max_sen_len"))  # tf-idf提取出的最大句数
    all_sentence = re.split(SENTENCE_SPLITTER, data)
    all_sentence = [sen for sen in all_sentence if sen != '']

    if len(all_sentence) > max_sen_len:
        return None, None
    for sentences in all_sentence:
        vector_fact = []
        words = filter_and_tokenize_num(config, sentences, stopwords)
        if len(words) > max_len_f:
            return None, None
        for word in words:
            vector_fact.append(transformer.load(word))
        words_vec = fill_list_with_average(vector_fact, max_len_f)
        all_vec.append(words_vec)

    all_vec = fill_list_with_average_juzi(all_vec, max_sen_len)
    tensor_fact = torch.tensor(all_vec)

    return tensor_fact, tensor_fact.shape
"""
def gen_putong_tensor(data, config, transformer,tfidf_model):
    all_vec = []
    vector_fact = []
    max_len_f = int(config.get("data", "max_len_f"))  # tf-idf提取出的最大词数，即最多词汇数
    all_sentence = re.split(SENTENCE_SPLITTER, data)
    all_sentence = [sen for sen in all_sentence if sen != '']

    for sentences in all_sentence:
        #words = filter_and_tokenize_num(config, sentences, stopwords) #保留数字
        words = filter_and_tokenize_num(config, sentences, stopwords)
        all_vec.extend(words)

    if len(all_vec) > max_len_f:
        # print(f"yuan : {all_vec[:50]}")
        top_words = tfidf(all_vec, max_len_f, tfidf_model)  # 词数多的话，获得tf-idf觉得每篇文章重要的词构成的列表(一般事实)
        # print(f"tfidf : {top_words[:50]}")
    else:
        top_words = putong_cut(all_vec, max_len_f)  # 词数少就直接用

    for word in top_words:
        vector_fact.append(transformer.load(word))

    if len(vector_fact) < max_len_f:
        vector_fact = fill_list_with_average(vector_fact, max_len_f)

    tensor_fact = torch.tensor(np.array(vector_fact))

    return tensor_fact, tensor_fact.shape

def gen_processed_putong_tensor(data, config, transformer):
    # 这个函数里传入的data是wordid列表,最多为max_len_f个词id
    vector_fact = []
    max_len_f = int(config.get("data", "max_len_f"))  # tf-idf提取出的最大词数，即最多词汇数

    for word_id in data:
        #words = filter_and_tokenize_num(config, sentences, stopwords) #保留数字
        words_vec = transformer.load_from_id(word_id) # BLANK返回None，非BLANK返回向量
        if words_vec is not None:
            vector_fact.append(words_vec)

    if len(vector_fact) > max_len_f:
       print("error")

    if len(vector_fact) < max_len_f:
        vector_fact = fill_list_with_average(vector_fact, max_len_f)

    tensor_fact = torch.tensor(np.array(vector_fact))

    return tensor_fact, tensor_fact.shape
def parse(data, config, transformer, tfidf_model):#传入一整块数据，返回张量和（三个标签堆叠成的张量）
    label_list = config.get("data", "type_of_label").replace(" ", "").split(",")
    label = []
    """
    data["meta"]["accusation"]
    data["meta"]["relevant_articles"]
    data["meta"]["term_of_imprisonment"]
    """
    for x in label_list:
        if x == "crit":
            label.append(analyze_crit(data["accu_label"]))
        if x == "law":
            label.append(analyze_law(data["law_label"]))
        if x == "time":
            label.append(analyze_time(data["term"]))
    gold_matrix = gen_gold_keyword(config, data["accu_label"])
    #tensor, shape_tensor = gen_putong_tensor(data["fact"], config, transformer, tfidf_model)
    tensor, shape_tensor = gen_processed_putong_tensor(data["fact"], config, transformer)
    crit_index = torch.tensor(accusation_dict[data["accu_label"][0]])
    #print(data["meta"]["relevant_articles"][0])
    #law_index = torch.tensor(law_dict[int(data["meta"]["relevant_articles"][0])])

    return tensor, shape_tensor, torch.cat(label), gold_matrix, crit_index


def check(data, config):
    """
    if not (check_sentence(gen_sentence(data["fact"]), config)):
        return False
    """

    if len(data["criminal"]) != 1:#单一罪犯
        return False

    if len(data["accu_label"]) != 1 or len(data["law_label"]) != 1:#单一罪名法条
        return False
    """"""
    """
    law = int(data["meta"]["relevant_articles"][0])
    #   之所以要去掉这些法条，是因为他们大多是和<100的罪名一起出现，导致大多都被筛去
    
    if law in [151, 307, 130]:
        return False
    """
    if not (check_crit(data["accu_label"])):
        return False
    if not (check_law(data["law_label"])):
        return False

    return True

def gen_crit_rep(config):
    # batch_size = int(config.get("data", "batch_size"))
    vec_size = int(config.get("data", "vec_size"))
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    word_dic = os.path.join(data_path, config.get("data", "word_dic"))
    vec_path = os.path.join(data_path, config.get("data", "vec_path"))
    rep_torch = torch.zeros(len(all_keywords), vec_size)
    with open(word_dic, 'rb') as f:
        word2id = pickle.load(f)
        vec = np.load(vec_path)
        for item in keywords_index.items():
            keywords = item[0]
            index = item[1]
            try:
                rep_torch[index] = torch.from_numpy(vec[word2id[keywords]].astype(dtype=np.float32))
            except:
                #print(keywords)
                rep_torch[index] = torch.from_numpy(vec[word2id["UNK"]].astype(dtype=np.float32))
    return rep_torch

def gen_gold_keyword(config, crit):  # 生成该案件指控对应关键词的位置编码,可以对应多个词
    res = torch.from_numpy(np.zeros(len(all_keywords)))
    for x in crit:
        keywords = crit_keyword[x]
        for word in keywords:
            if word in keywords_index.keys():
                res[keywords_index[word]] = 1
    return res

def get_allword_rep():
    return all_word_tensor
def get_allword_len():
    if crit_keyword and all_keywords:
        return len(all_keywords)

def gen_sim_keyworsds(config):
    threshold = float(config.get("data", "keywords_threshold"))
    max_len = 0
    for x in crit_keyword.values():
        if len(x) > max_len:
            max_len = len(x)
    rep_torch = get_allword_rep()  # [all_keywords, vec_size]
    crit_keyword_vec = torch.zeros(len(accusation_list), max_len, rep_torch.shape[-1])  # 先生成[crit_num,max_keyword_num,keyword_size]罪名-关键词向量矩阵
    for crit in crit_keyword.items():
        crit_name = crit[0]
        crit_keywords = crit[1]
        crit_index = accusation_dict[crit_name]
        count = 0
        for word in crit_keywords:
            index = keywords_index[word]
            crit_keyword_vec[crit_index][count] = rep_torch[index]
            count += 1
        if count < max_len:
            mean_vec = crit_keyword_vec[crit_index].mean(dim=0)
            for num in range(count,max_len):
                crit_keyword_vec[crit_index][num] = mean_vec
    print(crit_keyword_vec.shape)
    crim_key_meanrep = crit_keyword_vec.mean(dim=1)  # [crit_num,keyword_size]
    similarity_matrix = torch.mm(F.normalize(crim_key_meanrep, dim=1),
                                 F.normalize(crim_key_meanrep, dim=1).T)
    # 大于等于阈值的位置设为1，小于阈值的位置设为0
    binary_matrix = torch.where(similarity_matrix >= threshold,
                                torch.tensor(1.0),
                                torch.tensor(0.0))
    print(binary_matrix.shape)
    #[crit_num, crit_num]对角线为1
    return binary_matrix

def calculate_mask(label_matrix, sim_matrix):
    """
    label_matrix是类别向量，维度为【batch-size】，值就是crit的索引 [crit0,crit1,crit2,crit3,crit1,crit2]
    sim_matrix是0-1相似矩阵，维度为【crit_size，crit_size】，行和列都是crit的索引
    """
    label_mat = label_matrix.view(-1, 1)
    same_class_matrix = (label_mat == label_mat.T).int() # 类别相同为1，不同为0【batch-size，batch-size】
    same_class_matrix = 1 - same_class_matrix # 类别相同为0，不同为1【batch-size，batch-size】
    # print(same_class_matrix)
    # print(f"mask.sum(1): {same_class_matrix.sum(1)}")
    sample_similarity = sim_matrix[label_matrix]  # [batch-size,crit_size] 每个案件罪名对应的相似罪名（位置索引是对应相似罪名索引）
    sample_pair_similarity = sample_similarity[:, label_matrix] # 0 不相似，1 相似
    sample_pair_similarity_contra = 1 - sample_pair_similarity  # 0 相似，1 不相似
    # print(sample_pair_similarity)
    # 罪名相同为正样本
    sim_labels = sample_pair_similarity * same_class_matrix #  罪名不同且关键词相似度高为负样本(为1)

    labels = torch.zeros_like(same_class_matrix)
    labels[same_class_matrix == 0] = 1  #  罪名相同正样本(为1)
    labels[sim_labels == 1] = -1  #  罪名不同且关键词相似度高为难负样本(为-1)
    labels[(sample_pair_similarity_contra * same_class_matrix) == 1] = 0  #  罪名不同但关键词相似度不高为不重要样本(为0)
    """
    sample_pair_similarity = sample_similarity[:, label_matrix]
    # 等价于：
    for i in range(batch_size):
        for j in range(batch_size):
            sample_pair_similarity[i, j] = sample_similarity[i, label_matrix[j]]
    crit0:[1,1,0,0]
    crit1:[1,1,1,1]
    crit2:[0,1,1,1]
    crit3:[0,1,1,1]
    sample_similarity
    batch0:[1,1,0,0]   0   --->crit0,crit1--->label_matrix对应位置索引,0,1即batch0和batch1
    batch1:[1,1,1,1]   1   --->crit0,crit1,crit2,crit3--->
    batch2:[0,1,1,1]   2   --->crit1,crit2,crit3--->
    batch3:[0,1,1,1]   3   ...
    batch4:[1,1,0,0]   1   ...
    batch5:[1,1,1,1]   2   ...
    这里返回相同0-1矩阵【batch-size，batch-size】，把罪名不同或关键词相似度高的作为不相同类别矩阵(值为0)
    """
    return labels

def get_crit_dict():
    return accusation_dict
def get_crit_word_sim():
    return crit_words_sim


