# from keras_sdec import DeepEmbeddingClustering
from models.SDEC_AD import DeepEmbeddingClustering
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scriptss.load_data import load_data, load_anomalous_synsets
import numpy as np
import json


def plot_roc_pr(fpr_sdec, tpr_sdec, roc_thresholds_sdec, roc_auc_sdec,
                prec_sdec, recall_sdec, pr_thresholds_sdec, pr_auc_sdec):
    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 10))  # outward by 10 points
                spine.set_smart_bounds(True)
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    label_font = {"fontname": "Times New Roman", "fontsize": 13}
    plt.rcParams['xtick.direction'] = 'in'      # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'      # 将y轴的刻度线方向设置向内
    # plt.xticks(fontsize=5)
    # plt.yticks(fontsize=5)

    fig, ax1 = plt.subplots()
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)  # 刻度字体大小13
    ax1.set_xlabel('False Positive Rate', labelpad=0.8, **label_font)
    ax1.set_ylabel('True Positive Rate', labelpad=1.2, **label_font)

    # plot ROC results
    ax1.plot(fpr_sdec, tpr_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='Incremental Discovery Model(AUC = %0.2f)' % roc_auc_sdec)
    adjust_spines(ax1, ['left', 'bottom'])

    # display only a left and bottom box border in matplotlib
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.05, -0.025), handlelength=3.5, borderpad=1, labelspacing=1,
               prop={"family": "Times New Roman", "size": 14})
    # fig.savefig('roc(bert_triple).eps', format='eps')
    fig.savefig('roc(bert_text3).png', format='png')
    plt.close(fig)

    ### plot PRC
    fig, ax2 = plt.subplots()
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)  # 刻度字体大小13
    ax2.set_xlabel('Recall', labelpad=0.8, **label_font)
    ax2.set_ylabel('Precision', labelpad=1.2, **label_font)

    # plot PRC results
    ax2.plot(recall_sdec, prec_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='Incremental Discovery Model (AUC = %0.2f)' % pr_auc_sdec)
    adjust_spines(ax2, ['left', 'bottom'])

    ax2.legend(loc="upper right", handlelength=3.5, borderpad=1.2, labelspacing=1.2,
               prop={"family": "Times New Roman", "size": 14})
    # fig.savefig('prc(bert_text).eps', format='eps')
    fig.savefig('prc(bert_text3).png', format='png')
    plt.close(fig)

    # -------------------------中文------------------------
    fig, ax3 = plt.subplots()
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)  # 刻度字体大小13
    ax3.set_xlabel('False Positive Rate', labelpad=0.8, **label_font)
    ax3.set_ylabel('True Positive Rate', labelpad=1.2, **label_font)

    # plot ROC results
    ax3.plot(fpr_sdec, tpr_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='增量发现模型(AUC = %0.2f)' % roc_auc_sdec)
    adjust_spines(ax3, ['left', 'bottom'])

    # display only a left and bottom box border in matplotlib
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.legend(loc="lower right", bbox_to_anchor=(1.05, -0.025), handlelength=3.5, borderpad=1, labelspacing=1,
               prop={"family": "SimHei", "size": 14})
    # fig.savefig('roc(bert_triple).eps', format='eps')
    fig.savefig('roc(bert_text中文).png', format='png')
    plt.close(fig)

    ### plot PRC
    fig, ax4 = plt.subplots()
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)  # 刻度字体大小13
    ax4.set_xlabel('Recall', labelpad=0.8, **label_font)
    ax4.set_ylabel('Precision', labelpad=1.2, **label_font)

    # plot PRC results
    ax4.plot(recall_sdec, prec_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='增量发现模型(AUC = %0.2f)' % pr_auc_sdec)
    adjust_spines(ax4, ['left', 'bottom'])

    ax4.legend(loc="upper right", handlelength=3.5, borderpad=1.2, labelspacing=1.2,
               prop={"family": "SimHei", "size": 14})
    # fig.savefig('prc(bert_text).eps', format='eps')
    fig.savefig('prc(bert_text中文).png', format='png')
    plt.close(fig)


if __name__ == "__main__":
    fn17_embedding_file = "expand_data/Bert_Text/text_bert_first.p"
    fnplus_embedding_file = "expand_data/Bert_Text/text_bert_second.p"
    # anomalous_synset_embedding_filename = "expand_data/Bert_Triple/a(Triple).p"
    anomalous_synset_embedding_filename = "expand_data/Bert_Text/text_bert_anomalies.p"
    dict1_file = "dicts/text_num_type.json"
    weights = r"trained_SDEC_AD(DuEE-fin_ChinFinAnn)/trained_SDEC_AD(bert_text)/SDEC_AD_bcubed_fscore_0.74868.h5"
    with open(dict1_file, 'r', encoding='utf-8') as df1:
        dict1 = json.loads(df1.readline())
        df1.close()

    newX, newY, num_frames, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file, dict1)
    c = DeepEmbeddingClustering(n_clusters=num_frames,
                                input_dim=768,
                                encoders_dims=[7500, 1000])

    # print("Train Autoencoder...")
    # c.initialize(newX[:cut_off],
    #              y=newY[:cut_off],
    #              finetune_iters=100000,
    #              layerwise_pretrain_iters=50000,
    #              save_autoencoder=True)
    #
    # print("Clustering...")
    # L = c.cluster(newX, y=newY, cut_off=cut_off, iter_max=1e6)

    print("Train Decoder...")
    # use the saved model when running the clustering (c.cluster)
    c.train_decoders(SDEC_trained_weights=weights,
                     X=newX,
                     finetune_iters=50000,
                     layerwise_pretrain_iters=25000)

    print("Anomaly Detection...")
    # 获得异常LU的tensor（anom_X）以及异常LU（anom_Y）
    anom_X, anom_Y = load_anomalous_synsets(anomalous_synset_embedding_filename)
    # 正常 LUs 的 tensor 与异常 LUs 的 tensor 拼接
    combined_X = np.concatenate((newX, anom_X), axis=0)
    # 0矩阵（正常LUs的维度）拼接 1矩阵（异常LUs的维度）
    combined_Y = np.concatenate((np.zeros(shape=newX.shape[0]), np.ones(shape=anom_X.shape[0])), axis=0)

    # 重构误差
    norm_scores = c.reconstruction_loss(newX, individual=True)
    anom_scores = c.reconstruction_loss(anom_X, individual=True)
    # 误差拼接
    scores = np.concatenate((norm_scores, anom_scores), axis=0)
    # 0矩阵（正常LUs的维度）拼接 1矩阵（异常LUs的维度）
    y_labels = np.concatenate((np.zeros(shape=newX.shape[0]), np.ones(shape=anom_X.shape[0])), axis=0)

    # sklearn.metrics.roc_curve()函数绘制 ROC曲线
    fpr, tpr, roc_thresholds = roc_curve(y_labels, scores, pos_label=1)
    # 绘制 precision-recall曲线
    prec, recall, pr_thresholds = precision_recall_curve(y_labels, scores, pos_label=1)
    '''
        sklearn.metrics.auc函数的输入是FPR和TPR的值，
        即ROC曲线中的真阳性率（true positive rate）和假阳性率（false positive rate）。
        得到的输出结果是一个float格式的数值，代指ROC曲线下的面积（AUC的值）。
    '''
    roc_auc_sdec = auc(fpr, tpr)
    pr_auc_sdec = auc(recall, prec)
    plot_roc_pr(fpr, tpr, roc_thresholds, roc_auc_sdec, prec, recall, pr_thresholds, pr_auc_sdec)



