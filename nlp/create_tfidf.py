"""
生成train_tfidf和test_tfidf
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
import json
import thulac
from config_parser import ConfigParser

cutter = None
def init_thulac(config):
    global cutter
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    dict_path = os.path.join(data_path, config.get("data", "dict_path"))
    model_path = os.path.join(data_path, config.get("data", "model_path"))
    cutter = thulac.thulac(user_dict=dict_path, model_path=model_path, seg_only=True, filt=False)

def create_tf(file, config):
    max_doc = int(config.get("data", "max_tdidfdoc"))
    all_documents = []
    with open(file, 'r', encoding='utf-8') as file:
        content = file.read()
        # print(content), 每个json对象都被读成1行
        # 使用分隔符将多个 JSON 对象分开
        # 因为不是规范json格式
        json_objects = content.strip().split('\n')
        zongdata = [json.loads(obj) for obj in json_objects]
        for data in zongdata:
            all_documents.append(data["fact"])
        tokenized_documents = [' '.join([item[0] for item in cutter.cut(doc)]) for doc in all_documents]
       # 初始化（无需提前分词）
        tfidf_model = TfidfVectorizer(
            max_features=max_doc,
        ).fit(tokenized_documents)  # 输入原始文本列表
    return tfidf_model


if __name__ == "__main__":
    config = ConfigParser()
    init_thulac(config)
    print("train_begin")
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    train_file = os.path.join(data_path, config.get("data", "train_data"))  # 应该是单个文件路径
    train_tfidf_model = create_tf(train_file, config)
    joblib.dump(train_tfidf_model, 'train_tfidf_model.joblib')
    print("train_finish")
    print("test_begin")
    test_file = os.path.join(data_path, config.get("data", "test_data"))  # 应该是单个文件路径
    test_tfidf_model = create_tf(test_file, config)
    joblib.dump(test_tfidf_model, 'test_tfidf_model.joblib')
    print("test_finsih")