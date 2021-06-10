import numpy as np
import matplotlib.pyplot as plt
import time

def detect(scores, threshold):
    return [1 if score>threshold else 0 for score in scores]


def binary_confusion(true, pred):
    # pred: predicted class labels (0 normal, 1 anomaly)
    # true: true class labels
    assert len(pred)==len(true), "Number of predictions must match number of labels."
    p = sum(pred)
    n = len(pred) - p
    tp = sum(np.multiply(pred, true))
    fp = p - tp
    fn = sum(true) - tp
    tn = n - fn
    return tn, fp, fn, tp


def rates(true, pred):
    # Returns FPR, TPR
    tn, fp, fn, tp = binary_confusion(true, pred)
    return tp/(tp + fn), fp/(fp + tn)


def get_fpr(scores, threshold, true):
    pred = detect(scores, threshold)
    tpr, fpr = rates(true, pred)
    return fpr


def auc(scores, true, n=1000, plot=False):
    # Returns the AUC of the detector
    # n: number of points to trial threshold
    fprs = [1]
    tprs = [1]
    ts = np.linspace(min(scores),max(scores),n)
    for threshold in ts:
        pred = detect(scores, threshold)
        tpr, fpr = rates(true, pred)
        fprs.append(fpr)
        tprs.append(tpr)

    # Ensuring the plot cover all points for AUC calculation
    fprs.append(0)
    tprs.append(0)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fprs, tprs, color='k',linewidth=2)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_aspect('equal','box')
        ax.set_xlabel("FPR", fontsize=14)
        ax.set_ylabel("TPR", fontsize=14)
        plt.show()

    # Simple integration to find AUC
    return np.trapz(np.flip(tprs), x=np.flip(fprs))


def plot_nyc_results(data_test, ts_test, scores, threshold, anomaly_ts, save_fig=False):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(13.5,5) ,sharex='all')
    ax[0].plot(ts_test, data_test, color='k')
    for anomaly in anomaly_ts:
        ax[0].axvline(anomaly, color='r', linestyle='--')
    ax[1].plot(ts_test, scores, color='r')
    ax[1].axhline(threshold, color='k', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig("results/nyc_plot.png", dpi=200)
    plt.show()

def save_nyc_results(true, true_corrected, scores, threshold, alpha):
    t = time.localtime()
    current_time = time.strftime("%d/%m/%y %H:%M:%S", t)

    f = open("results/nyc_results.txt", "a")
    f.write("------- NYC results -------\n")
    f.write("---- {} ----\n".format(current_time))

    f.write('Desired FPR: {:.2%}\n'.format(alpha))
    f.write('Realised FPR: {:.2%}\n'.format(get_fpr(scores, threshold, true)))
    f.write('Corrected FPR: {:.2%}\n'.format(get_fpr(scores, threshold, true_corrected)))
    f.write('AUC: {:.3}\n'.format(auc(scores, true, plot=False)))
    f.write("\n")
    f.close()

    a_file = open("results/nyc_results.txt", "r")
    lines = a_file.readlines()
    last_lines = lines[-7:]
    for line in last_lines:
        print(line)