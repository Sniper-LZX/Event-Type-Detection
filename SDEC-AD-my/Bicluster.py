from sklearn.cluster import KMeans
import numpy as np
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

with open('reconstruction_loss/unknown_text_scores.json', 'r', encoding='utf-8') as anom:
    datas = json.load(anom)

    num_of_r = dict()
    for data in datas:
        num = round(data, 4)
        if num not in num_of_r:
            num_of_r[num] = 1
        else:
            num_of_r[num] += 1
    tmp = 0
    for o in num_of_r:
        if num_of_r[o] > 27:
            tmp += num_of_r[o]

    X = list(num_of_r.keys())
    Y = list(num_of_r.values())
    label_font = {"fontname": "SimHei", "fontsize": 14}

    fig, ax1 = plt.subplots()

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax1.set_xlim([-0.006, 0.09])
    ax1.set_ylim([-5, 55])
    ax1.set_xlabel('重构误差', labelpad=1, **label_font)
    ax1.set_ylabel('事件数量（个）', labelpad=1, **label_font)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_position(("data", -0.006))
    ax1.spines['bottom'].set_position(("data", -5))
    plt.tick_params(labelsize=14)

    plt.scatter(X, Y, c='c', s=25, edgecolors='r')
    plt.plot([0, 0.08], [27, 27])
    plt.plot([0.0005, 0.01, 0.01, 0.0005, 0.0005], [27.2, 27.2, 51, 51, 27.2], color='r', linestyle='dashed')
    plt.scatter([0.0038, 0.0065], [27, 27], s=25)
    plt.annotate('(0.0038,27)', xy=(0.0038, 27), xytext=(0.013, 31), fontsize=14)
    plt.plot([0.0038, 0.013], [27, 31], color='#1F77B4', linestyle='dashed')
    plt.annotate('(0.0065,27)', xy=(0.0065, 27), xytext=(0.013, 22), fontsize=14)
    plt.plot([0.0065, 0.013], [27, 23], color='#1F77B4', linestyle='dashed')

    plt.show()
    # fig.savefig('bert_text(abnormal)中文.png', format='png')

    label_font2 = {"fontname": "Times New Roman", "fontsize": 14}
    ax1.set_xlabel('Reconstruction_loss', labelpad=1, **label_font2)
    ax1.set_ylabel('Event Count', labelpad=1, **label_font2)
    plt.show()
    # fig.savefig('bert_text(abnormal).png', format='png')


