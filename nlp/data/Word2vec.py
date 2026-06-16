# coding:utf-8
import numpy as np
import pickle
class word2vec:
    word2id = {}
    vec = None
    def __init__(self, word_dic,
                 vec_path):
        print("begin to load word embedding")
        with open(word_dic, 'rb') as f:
            self.word2id = pickle.load(f)
            self.vec = np.load(vec_path)
        print("load word embedding succeed")
    def load(self, word):
        try:
            return self.vec[self.word2id[word]].astype(dtype=np.float32)
        except:
            return self.vec[self.word2id['UNK']].astype(dtype=np.float32)
    def load_from_id(self, word_id):
        if word_id != self.word2id['BLANK']:
            return self.vec[word_id].astype(dtype=np.float32)
        else:
            return None

"""
if __name__ == "__main__":
    a = word2vec()
    print(a.load("其系"))
"""