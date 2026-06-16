import os
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
from loss import cross_entropy_loss, one_cross_entropy_loss, log_regression
from data.utils import calc_accuracy, gen_result, print_info
from data.deve_formatter import get_num_classes, get_crit_dict
from nnet import stackedLSTM
from nnet import AutoRelatLSTM
from nnet import ExtralossLSTM
from nnet import AttGcnLSTM
from nnet import SuplossLSTM
from nnet import LSTMDecoder
from nnet import MPBFN
from draw_loss import plot_multi_losses, plot_losses_subplots
from draw_gram import drawgram
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def test_file(net, test_dataset, usegpu, config, epoch):
    batch_count = 0
    net.eval()
    test_dataset.reset(False)# 不打乱
    conv_weight_list = []
    running_acc = []
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    task_loss_type = config.get("data", "type_of_loss").replace(" ", "").split(",")
    test_result_path = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"))
    if not (os.path.exists(test_result_path)):
        os.makedirs(test_result_path)
    test_result_path = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"), str(epoch))
    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    with torch.no_grad():  # 添加no_grad上下文管理器
        while True:
            batch_count += 1
            print("****************")
            data = test_dataset.fetch_data()
            print("测试集拿了一个batch")
            print("****************")
            if data is None:
                break
            #return 0index, 1tensor, 2shape_tensor, 3torch.cat(label), 4simzhi

            inputs = data[0] #展平张量
            labels = data[2] #标签
            explain_rep = data[3]
            gold_matrix = data[4]
            batch_crit_index = data[5]
            batch_mask = data[6]
            net.init_hidden(config, usegpu)
            if torch.cuda.is_available() and usegpu:
                inputs, labels, explain_rep, gold_matrix, batch_crit_index, batch_mask = \
                    Variable(inputs.cuda()), Variable(labels.cuda()), Variable(explain_rep.cuda()), Variable(gold_matrix.cuda()), Variable(batch_crit_index.cuda()), Variable(batch_mask.cuda())
            else:
                inputs, labels, explain_rep, gold_matrix, batch_crit_index, batch_mask = \
                    Variable(inputs), Variable(labels), Variable(explain_rep), Variable(gold_matrix), Variable(batch_crit_index), Variable(batch_mask)
            # isinstance(net, stackedLSTM) or isinstance(net, LSTMDecoder)
            if isinstance(net, stackedLSTM) or isinstance(net, AutoRelatLSTM) or isinstance(net, MPBFN) or isinstance(net, LSTMDecoder):
                outputs = net.forward(inputs, config)
            elif isinstance(net, ExtralossLSTM) or isinstance(net, AttGcnLSTM):
                outputs, _ = net.forward(inputs, config, explain_rep, gold_matrix)
            elif isinstance(net, SuplossLSTM) :
                outputs, _, _, mat_for_tsne, conv_weight = net.forward(inputs, config, explain_rep, gold_matrix, batch_crit_index, batch_mask)
                conv_weight_list.extend(conv_weight)
            else:
                outputs = net.forward(inputs, config)

            #将连接的标签向量按任务分割成独立标签
            reals = []
            accumulate = 0
            for a in range(0, len(task_name)):
                num_class = get_num_classes(task_name[a])
                reals.append(labels[:, accumulate:accumulate + num_class])
                accumulate += num_class

            labels = reals
            for a in range(0, len(task_name)):
                running_acc[a] = calc_accuracy(outputs[a], labels[a], task_loss_type[a], running_acc[a])

            # draw_tsne(config, mat_for_tsne, batch_crit_index.data, batch_mask.data, batch_count)
        # draw_kernel(config, conv_weight_list)
    net.train()

    print_info('Test result:')
    for a in range(0, len(task_name)):
        print("%s result:" % task_name[a])
        try:
            gen_result(running_acc[a], True, file_path=test_result_path + "-" + task_name[a], class_name=task_name[a])
        except Exception as e:
            pass
    print("")


def train_file(net, train_dataset, test_dataset, usegpu, config):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("data", "batch_size")
    learning_rate = config.getfloat("train", "learning_rate")
    momemtum = config.getfloat("train", "momentum")
    weight_decay = config.getfloat("train", "weight_decay")

    output_time = config.getint("output", "output_time")
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    task_loss_type = config.get("data", "type_of_loss").replace(" ", "").split(",")
    optimizer_type = config.get("train", "optimizer")

    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    test_time = config.getint("output", "test_time")

    criterion = []
    for a in range(0, len(task_name)):
        if task_loss_type[a] == "multi_classification":
            criterion.append(cross_entropy_loss)
        elif task_loss_type[a] == "single_classification":
            criterion.append(one_cross_entropy_loss)
        elif task_loss_type[a] == "log_regression":
            criterion.append(log_regression)
    # 换优化器时要调learning_rate和weight_decay
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)

    total_loss = []
    first = True
    try:
        epps = config.get("train", "pre_train")
        epps = int(epps) - 1
    except Exception as e:
        epps = -1

    print_info("Training begin")
    #这里并没有pre_train所以还是range(0, epoch)
    for epoch_num in range(epps + 1, epoch):
        conv_weight_list = []
        all_loss = []  # 总结loss并画图
        all_gram = []  # 总结梯度范数并画图
        print("第", epoch_num + 1, "个epoch")
        running_loss = 0
        running_acc = []
        total_acc = []
        for a in range(0, len(task_name)):
            running_acc.append([])
            total_acc.append([])
            for b in range(0, get_num_classes(task_name[a])):
                running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
                total_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        cnt = 0
        idx = 0
        train_dataset.reset(True)
        while True:
            # print_info("One round begin, waiting for data...")
            data = train_dataset.fetch_data()
            if data is None:
                break
            idx += batch_size
            cnt += 1
            print("第", cnt + 1, "个train_batch")
            inputs = data[0]  # 展平张量
            labels = data[2]  # 标签
            explain_rep = data[3]
            gold_matrix = data[4]
            batch_crit_index = data[5]
            batch_mask = data[6]
            # print(batch_crit_index)
            if torch.cuda.is_available() and usegpu:
                inputs, labels, explain_rep, gold_matrix, batch_crit_index, batch_mask = \
                    Variable(inputs.cuda()), Variable(labels.cuda()), Variable(explain_rep.cuda()), Variable(gold_matrix.cuda()), Variable(batch_crit_index.cuda()), Variable(batch_mask.cuda())
            else:
                inputs, labels, explain_rep, gold_matrix, batch_crit_index, batch_mask = \
                    Variable(inputs), Variable(labels), Variable(explain_rep), Variable(gold_matrix), Variable(batch_crit_index), Variable(batch_mask)

            # print_info("Data fetch done, forwarding...")

            net.init_hidden(config, usegpu)
            optimizer.zero_grad()
            word_loss = 0
            contrast_loss = 0
            # isinstance(net, stackedLSTM) or isinstance(net, LSTMDecoder)
            #start = time.time()
            if isinstance(net, stackedLSTM) or isinstance(net, AutoRelatLSTM) or isinstance(net, MPBFN) or isinstance(net, LSTMDecoder):
                outputs = net.forward(inputs, config)
            elif isinstance(net, ExtralossLSTM) or isinstance(net, AttGcnLSTM):
                outputs, word_loss = net.forward(inputs, config, explain_rep, gold_matrix)
            elif isinstance(net, SuplossLSTM):
                outputs, word_loss, contrast_loss, _, _ = net.forward(inputs, config, explain_rep, gold_matrix, batch_crit_index, batch_mask)
                # conv_weight_list.extend(conv_weight)
            else:
                outputs = net.forward(inputs, config)
            #end = time.time()
            #elapsed = end - start
            #print(f"耗时 {elapsed:.3f} 秒")
            reals = []
            accumulate = 0
            #标签切分(law,crit,time)->law crit time
            for a in range(0, len(task_name)):
                num_class = get_num_classes(task_name[a])
                reals.append(labels[:, accumulate:accumulate + num_class])
                accumulate += num_class

            labels = reals

            # print_info("Forward done, lossing...")
            # print(labels)
            # print(outputs)
            loss = 0
            # 添加各任务损失监控
            task_losses = []
            for a in range(0, len(task_name)):
                task_loss = criterion[a](outputs[a], labels[a].float())
                task_losses.append(task_loss.item())
                loss = loss + task_loss
                running_acc[a] = calc_accuracy(outputs[a], labels[a], task_loss_type[a], running_acc[a])
            all_loss.append(task_losses)
            # print_info("Loss done, backwarding...")
            if isinstance(net, ExtralossLSTM) or isinstance(net, AttGcnLSTM) or isinstance(net, LSTMDecoder):
                loss = loss + word_loss
            if isinstance(net, SuplossLSTM):
                loss = loss + word_loss
                loss = loss + 0.05 * contrast_loss
            loss.backward()


            # ============ 添加梯度监控 ============
            clip_grad_norm_(net.parameters(), max_norm=30.0)

            # 计算梯度范数（不实际裁剪，只计算）
            total_norm = clip_grad_norm_(net.parameters(), max_norm=float('inf'))
            all_gram.append(total_norm)
            # 恢复原来的梯度（因为上面裁剪时设了无限大阈值，所以不会改变梯度）
            """
            print(f"梯度范数: {total_norm:.4f}")
            """
            # ====================================

            optimizer.step()

            running_loss += loss.item()
            #输出每个任务的损失

            print(f"all Loss: {loss.item()}")
            for a, task_loss in enumerate(task_losses):
                print(f"Task {task_name[a]} loss: {task_loss:.4f}")
            print(f"Word_select loss: {word_loss:.4f}")
            print(f"contrast loss: {0.05 * contrast_loss:.4f}")

            # 每output_time个batch输出一次

            if cnt % output_time == 0:
                print_info("Current res:")
                print('[%d, %5d, %5d] loss: %.3f' %
                      (epoch_num + 1, cnt, idx, running_loss / output_time))
                for a in range(0, len(task_name)):
                    print("%s result:" % task_name[a])
                    gen_result(running_acc[a])
                print("")

                total_loss.append(running_loss / output_time)
                running_loss = 0.0
                for a in range(0, len(running_acc)):
                    for b in range(0, len(running_acc[a])):
                        total_acc[a][b]["TP"] += running_acc[a][b]["TP"]
                        total_acc[a][b]["FP"] += running_acc[a][b]["FP"]
                        total_acc[a][b]["FN"] += running_acc[a][b]["FN"]
                        total_acc[a][b]["TN"] += running_acc[a][b]["TN"]

                running_acc = []
                for a in range(0, len(task_name)):
                    running_acc.append([])
                    for b in range(0, get_num_classes(task_name[a])):
                        running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        # draw_kernel(config, conv_weight_list)
        # plot_multi_losses(config, all_loss, epoch_num)
        # plot_losses_subplots(config, all_loss, epoch_num)
        # drawgram(config, all_gram, epoch_num)
        """
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch_num} learning rate: {current_lr:.2e}")
        """
        if not (os.path.exists(model_path)):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))
        # 这里好像是在用测试集

        if (epoch_num + 1) % test_time == 0:
            print("我进行了测试集测试")
            test_file(net, test_dataset, usegpu, config, epoch_num + 1)

        for a in range(0, len(task_name)):
            # 先构建目录路径
            output_dir = os.path.join(config.get("output", "test_path"),
                                      config.get("output", "model_name"))
            # 确保目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            gen_result(total_acc[a], True,
                       file_path=os.path.join(output_dir, "total") + "-" + task_name[a] + '_train_' + str(epoch_num),
                       class_name=task_name[a])

    print_info("Training done")

    test_file(net, test_dataset, usegpu, config, 0)
    torch.save(net.state_dict(), os.path.join(model_path, "model.pkl"))

def draw_tsne(config, x, y, batch_mask, num):
    # num 是第几个batch
    # 传入特征矩阵，标签向量
    # X: shape (n_samples, n_features)
    # y: shape (n_samples,)
    mask = batch_mask.cpu().numpy() if isinstance(x, torch.Tensor) else batch_mask

    valid_indices = (mask == -1)  # 获取有效样本的索引

    row_indices = np.where(valid_indices)[0]

    if np.sum(row_indices) == 0:  # 如果没有有效样本
        print('没有有效样本')
        return

    crit_dict = get_crit_dict()  # 罪名：索引
    reverse_dict = {value: key for key, value in crit_dict.items()}  # 索引：罪名
    # 转换为 numpy 数组
    x_numpy = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_numpy = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    # print(x.shape)
    # print(y.shape)
    # 只选择有效样本进行可视化
    x_change = x_numpy[row_indices]
    y_change = y_numpy[row_indices]

    row_min = x_change.min(axis=1, keepdims=True)
    row_max = x_change.max(axis=1, keepdims=True)
    x_change = (x_change - row_min) / (row_max - row_min)

    n_classes = np.unique(y_change)
    if len(n_classes) >= 22:
        print('类别太多')
        return

    n_samples = x_change.shape[0]
    perplexity = min(30, n_samples - 1)
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    x_tsne = tsne.fit_transform(x_change)
    # 可视化
    plt.figure(figsize=(8, 8))
    colors = ['lightcoral','red', 'maroon','orange','green','gold',
              'y', 'yellow','yellowgreen','greenyellow','forestgreen','lime', 'blue'
               , 'cyan', 'deepskyblue','slateblue','blueviolet','violet',
              'purple','fuchsia','crimson','pink'
              ]
    for i in n_classes:
        plt.scatter(x_tsne[y_change == i, 0],
                    x_tsne[y_change == i, 1],
                    c=colors[i % len(colors)],
                    )
    """
    for i in n_classes:
        plt.scatter(x_tsne[y_change == i, 0],
                    x_tsne[y_change == i, 1],
                    c=colors[i % len(colors)],
                    label=f'{reverse_dict[i]}')
    """
    # plt.legend()
    output_dir = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"))
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "total") + "-tsne-" + str(num) + ".png"

    plt.savefig(file_path, dpi=600, bbox_inches='tight', format="png")
    # 显示图形
    plt.close()

def draw_kernel(config, con_weight):
    con_weight = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in con_weight]
    result_1 = [con_weight[i] for i in range(0, len(con_weight), 6)]
    result_2 = [con_weight[i] for i in range(1, len(con_weight), 6)]
    result_3 = [con_weight[i] for i in range(2, len(con_weight), 6)]
    result_4 = [con_weight[i] for i in range(3, len(con_weight), 6)]
    result_5 = [con_weight[i] for i in range(4, len(con_weight), 6)]
    result_6 = [con_weight[i] for i in range(5, len(con_weight), 6)]
    result_1 = np.array(result_1).flatten()
    result_2 = np.array(result_2).flatten()
    result_3 = np.array(result_3).flatten()
    result_4 = np.array(result_4).flatten()
    result_5 = np.array(result_5).flatten()
    result_6 = np.array(result_6).flatten()
    print(result_1.shape)
    print(result_2.shape)
    print(result_3.shape)
    print(result_4.shape)
    print(result_5.shape)
    print(result_6.shape)
    weight_list = [[result_1, result_3], [result_2, result_5], [result_4, result_6]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=False)
    axes = axes.flatten()
    name = ['law<->charge', 'law<->time', 'charge<->time']
    legend_name = [['law->charge', 'charge->law'], ['law->time', 'time->law'], ['charge->time', 'time->charge']]
    color = ['steelblue', 'salmon']
    count = 0
    for ax, values, name in zip(axes, weight_list, name):
        cnt = 0
        label_cnt = 0
        label_name = legend_name[count]
        for value in values:
            sns.kdeplot(value, ax=ax, fill=True, color=color[cnt], alpha=0.5, linewidth=2, label=f'{label_name[label_cnt]}')
            cnt += 1
            label_cnt += 1
        count += 1
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    plt.tight_layout()

    output_dir = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"))
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "total") + "-kernel-" + ".png"

    plt.savefig(file_path, dpi=800, bbox_inches='tight', format="png")

    plt.close()
