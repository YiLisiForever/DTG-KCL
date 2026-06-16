"""
从数据集统计法条，罪名出现次数
输出到txt文件
law frequent
crit frequent
"""
import json
import pandas as pd

def createfile(ffile1):
    law_dic = {}
    crit_dic = {}
    cnt = 0
    with open(ffile1, 'r', encoding='utf-8') as file:
        content = file.read()
        # print(content), 每个json对象都被读成1行
        # 使用分隔符将多个 JSON 对象分开
        # 因为不是规范json格式
        json_objects = content.strip().split('\n')
        zongdata = [json.loads(obj) for obj in json_objects]
        for data in zongdata:
            cnt += 1
            accusation = data["meta"]["accusation"]
            article = data["meta"]["relevant_articles"]
            if len(data["meta"]["criminals"]) != 1:  # 单一罪犯
                continue
            if len(accusation) == 0 or len(article) == 0:  # 罪名法条非空
                continue

            if len(accusation) > 1 or len(article) > 1:
                continue
            """"""
            acc_num = accusation[0].replace("[", "").replace("]", "")
            crit_dic[acc_num] = crit_dic.get(acc_num, 0) + 1

            art_num = article[0]
            law_dic[art_num] = law_dic.get(art_num, 0) + 1
    return law_dic, crit_dic

def print_txt(dict, filename='output.txt'):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in dict.items():
            file.write(f"{key} {value}\n")

def print_excel(dict, file):
    df = pd.DataFrame(dict, index=[0])
    df.to_excel(file, index=False)  # index=False不保存行索引
    print("数据已保存到 output.xlsx")

law_dic, crit_dic = createfile("C:\\Users\\ADMIN\\Desktop\\big\\test_json.json")

#print_excel(law_dic, 'output_law.xlsx')
#print_excel(crit_dic, 'output_crit.xlsx')
print_txt(law_dic, 'law.txt')
print_txt(crit_dic, 'crit.txt')
""""""