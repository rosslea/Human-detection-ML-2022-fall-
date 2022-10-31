import numpy as np
import matplotlib.pyplot as plt

def T_or_F(scores, phase, th=0):
    mask = scores < th
    P = mask.sum()
    N = len(mask.flatten()) - P
    if phase == 'neg':
        TP, FP, TN, FN = (0, P, N, 0)
    elif phase == 'pos':
        TP, FP, TN, FN = (P, 0, 0, N)
    else:
        assert False, 'unknown phase'
    return TP, FP, TN, FN


def get_basic(neg_s, pos_s, th=0):
    TP, FP, TN, FN = T_or_F(neg_s, phase='neg', th=th)
    TP_2, FP_2, TN_2, FN_2 = T_or_F(pos_s, phase='pos', th=th)
    TP, FP, TN, FN = TP+TP_2, FP+FP_2, TN+TN_2, FN+FN_2
    return dict(TP=TP, FP=FP, TN=TN, FN=FN)



def get_rates(basic):
    TPR = basic['TP'] / (basic['TP']+basic['FN'])
    FPR = basic['FP'] / (basic['TN']+basic['FP'])
    return dict(TPR=TPR, FPR=FPR)

def get_ROC(num=100):
    neg_scores = np.load('data/neg_scores_logistic.npy')
    pos_scores = np.load('data/pos_scores_logistic.npy')
    threshold = map(lambda x:x/10, range(-num, num))
    coord = []
    for th in threshold:
        basic = get_basic(neg_scores, pos_scores,th=th)
        rate = get_rates(basic)
        coord.append(np.array([rate['TPR'], rate['FPR']]))
    coord = np.array(coord)
    X = coord[:,1]
    Y = coord[:,0]
    return X, Y

def plot_roc():
    X, Y = get_ROC(num=100)
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(X,Y, label='ROC')
        ax.plot(X, X, label='Ref line')
        ax.legend(title='')
        # ax.autoscale(tight=True)
        pparam = dict(xlabel='FPR', ylabel='TPR')
        ax.set(**pparam)
        # fig.savefig('figs/fig1.pdf')
        fig.savefig('figs/roc_logistic.jpg', dpi=300)
    # AUC
    delta = np.zeros(Y.shape)
    n = len(Y)
    delta[0] = (X[1]-X[0])/2
    delta[n-1] = (X[n-1]-X[n-2])/2
    for i in range(1,n-1):
        delta[i] = (X[i+1] - X[i-1])/2
    s = Y *delta
    AUC = s.sum()
    print("AUC=", AUC)
    return AUC

def plot_roc_minus():
    X, Y = get_ROC(num=100)
    Y = 1 - Y
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(X,Y, label='Miss Vs. FPR')
        ax.legend(title='')
        # ax.autoscale(tight=True)
        pparam = dict(xlabel='$Miss_{neg}$', ylabel='$Miss_{pos}$')
        ax.set(**pparam)
        # fig.savefig('figs/fig1.pdf')
        fig.savefig('figs/roc_minus_logistic.jpg', dpi=300)
    # AUC
    delta = np.zeros(Y.shape)
    n = len(Y)
    delta[0] = (X[1]-X[0])/2
    delta[n-1] = (X[n-1]-X[n-2])/2
    for i in range(1,n-1):
        delta[i] = (X[i+1] - X[i-1])/2
    s = Y *delta
    AUC = s.sum()
    print("AUC=", AUC)
    return AUC

AUC_1 = plot_roc_minus()
AUC_2 = plot_roc()