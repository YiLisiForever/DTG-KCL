import pickle as pk
import json
import os
from data.deal_fact import *
import re
import joblib
from config_parser import ConfigParser

SENTENCE_SPLITTER = re.compile(r'[。！？；…]')

train_tfidf_model = None
test_tfidf_model = None

config = ConfigParser()
init_thulac(config)

data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
with open(os.path.join(data_path, "stop_word.txt"), 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())  # 读取停用词并保存为集合

with open(os.path.join(data_path, config.get("data", "word_dic")), 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()


try:
    tfidf_file = os.path.join(data_path, "train_tfidf_model.joblib")
    train_tfidf_model = joblib.load(tfidf_file)
except FileNotFoundError:
    print("训练TF-IDF模型文件不存在")
    train_tfidf_model = None
try:
    tfidf_file = os.path.join(data_path, "test_tfidf_model.joblib")
    test_tfidf_model = joblib.load(tfidf_file)
except FileNotFoundError:
    print("测试TF-IDF模型文件不存在")
    test_tfidf_model = None

train_file = os.path.join(data_path, config.get("data", "train_data"))  # 应该是单个文件路径
test_file = os.path.join(data_path, config.get("data", "test_data"))  # 应该是单个文件路径
file_list = [train_file,test_file]


for i in range(len(file_list)):
    all_content = []
    with open(file_list[i], 'r', encoding='utf-8') as f:
        content = f.read()
        json_objects = content.strip().split('\n')
        zongdata = [json.loads(obj) for obj in json_objects]
        idx = 0
        for line in zongdata:
            if idx % 1000 == 0:  # 每1000条打印一次
                print(f"已处理 {idx} 条数据")
            idx += 1
            fact = line["fact"]
            law_label = line["meta"]["relevant_articles"]
            accu_label = line["meta"]["accusation"]
            term = line["meta"]["term_of_imprisonment"]
            criminal = line["meta"]["criminals"]
            id_list = []
            all_vec = []
            vector_fact = []
            max_len_f = int(config.get("data", "max_len_f"))  # tf-idf提取出的最大词数，即最多词汇数
            all_sentence = re.split(SENTENCE_SPLITTER, fact)
            all_sentence = [sen for sen in all_sentence if sen != '']

            for sentences in all_sentence:
                words = filter_and_tokenize_num(config, sentences, stopwords)
                all_vec.extend(words)

            if len(all_vec) > max_len_f:
                if file_list[i] == train_file:
                    top_words = tfidf(all_vec, max_len_f, train_tfidf_model)  # 训练集用训练tfidf
                else:
                    top_words = tfidf(all_vec, max_len_f, test_tfidf_model)  # 测试集用测试tfidf
            else:
                top_words = putong_cut(all_vec, max_len_f)  # 词数少就直接用

            for word in top_words:
                if word in word2id_dict:
                    id_list.append(int(word2id_dict[word]))
                else:
                    id_list.append(int(word2id_dict['UNK']))

            while len(id_list) < max_len_f:
                id_list.append(int(word2id_dict['BLANK']))


            all_content.append({
                "fact" : id_list,
                "law_label" : law_label,
                "accu_label" : accu_label,
                "criminal" : criminal,
                "term" : term
            })

    data_dict = all_content

    if file_list[i] == train_file:
        output_file = os.path.join(data_path, "processed_train.json")
    else:
        output_file = os.path.join(data_path, "processed_test.json")  # 测试集用测试tfidf

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=None)

    print(f"数据已保存到 {output_file}")
