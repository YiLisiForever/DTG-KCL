import re
import numpy as np
import thulac
import os
DIGIT_PATTERN = re.compile(r'\d')
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')
CHINESE_PATTERN = re.compile(r'^[\u4e00-\u9fa5]+$')
#已经在别的地方加载过jieba词典

cutter = None
def init_thulac(config):
    global cutter
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    dict_path = os.path.join(data_path, config.get("data", "dict_path"))
    model_path = os.path.join(data_path, config.get("data", "model_path"))
    cutter = thulac.thulac(user_dict=dict_path, model_path=model_path, seg_only=True, filt=False)

def filter_and_tokenize(config, sentence, stopwords):
    """优化后的过滤函数"""
    words = cutter.cut(sentence)  # 直接使用全局thulac
    words = [item[0] for item in words if item[0] != ' ']
    return [
        word for word in words
        if (word not in stopwords and
            not DIGIT_PATTERN.search(word) and
            not ENGLISH_PATTERN.search(word))
    ]

def filter_and_tokenize_s(config, sentence, stopwords):
    words = cutter.cut(sentence)
    words = [item[0] for item in words if item[0] != ' ']
    return [word for word in words
            if (word not in stopwords and
                CHINESE_PATTERN.fullmatch(word))
            ]

def filter_and_tokenize_num(config, sentence, stopwords):
    words = cutter.cut(sentence)
    words = [item[0] for item in words if item[0] != ' ']
    return [word for word in words
            if (word not in stopwords
                and not ENGLISH_PATTERN.search(word))
            ]

def thulac_cut(word):
    return cutter.cut(word)
def putong_cut(word_list,top_k):
    return word_list[:top_k]
def tfidf(cut_fact,top_k,tfidf_model):#  tfidf筛选top词，返回列表（包含文章词汇列表）

    # 将词汇列表拼接成TF-IDF预期的输入格式（空格分隔的字符串）
    text = " ".join(cut_fact)

    # 获取TF-IDF权重向量（稀疏矩阵）
    tfidf_vec = tfidf_model.transform([text])

    # 如果没有匹配的特征词
    if tfidf_vec.nnz == 0:
        # 返回原始词汇的前top_k个作为备选
        return cut_fact[:top_k]

    # 提取特征词和对应权重
    feature_names = tfidf_model.get_feature_names_out()
    sorted_indices = np.argsort(tfidf_vec.toarray()[0])[::-1]  # 权重降序排序

    # 返回前top_k个词（过滤掉零权重的词）
    top_words = [
                    feature_names[i]
                    for i in sorted_indices
                    if tfidf_vec[0, i] > 0
                ][:top_k]

    return top_words


def fill_list_with_average(vectors, target_length):  # 填充向量平均值, target_length 是目标最大长度，即max_len_f ，input_list是输入的词向量列表

    # 计算已有向量的平均值
    average_value = np.sum(vectors, axis=0)/ len(vectors)

    # 计算需要填充的元素个数
    elements_to_add = target_length - len(vectors)

    # 直接使用 extend 来避免创建临时列表
    vectors.extend([average_value] * elements_to_add)

    return vectors


def fill_list_with_average_s(vectors, target_length):  # 将空的特殊模块填充一般事实的向量平均值, target_length 是目标最大长度，即max_len_s ，input_list是输入的词向量列表
    new_vectors= []

    # 计算已有向量的平均值
    average_value = np.sum(vectors, axis=0)/ len(vectors)

    # 直接使用 extend 来避免创建临时列表
    new_vectors.extend([average_value] * target_length)

    return new_vectors

def fill_list_with_average_juzi(vectors, target_length):
    elements_to_add = target_length - len(vectors)

    all_vec = np.array(vectors)

    average_value = np.mean(all_vec, axis=0)

    print(average_value.shape)

    for x in range(elements_to_add):
        all_vec = np.append(all_vec, [average_value], axis=0)

    return all_vec
