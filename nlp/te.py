import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def draw_bar(data):

    sort_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    print(sort_data)
    ids = list(sort_data.keys())
    values = list(sort_data.values())
    print(values)
    values = [x for x in values if x < 600]
    print(range(len(values)))
    print(values)
    # 画柱状图
    plt.bar(range(len(values)), values)

    # 添加标签和标题
    plt.xlabel("Charge Category ID")
    plt.ylabel("Case Number")

    plt.show()
accusation_list = []
accusation_dict = {}
acc_num = {}
law_list = []
law_dict = {}
law_num = {}
data_path = "C:\\Users\\ADMIN\\Desktop\\big"
cnt1 = 0
min_frequency = 100
with open(os.path.join(data_path, "crit.txt"), "r", encoding="utf-8") as f:
    for line in f:
        data = line[:-1].split(" ")
        name = data[0]
        num = int(data[1])
        if num > min_frequency:
            cnt1 += num
            accusation_list.append(name)
            acc_num.update({(len(accusation_list) - 1): num})
            accusation_dict[name] = len(accusation_list) - 1

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
        law_num.update({int(name): num})
    for law_id, total_num in sum_dict.items():
        if total_num > min_frequency:
            cnt2 += total_num
            law_list.append(law_id)
            law_dict[law_id] = len(law_list) - 1

draw_bar(acc_num)
