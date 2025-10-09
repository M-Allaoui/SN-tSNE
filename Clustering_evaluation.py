from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import numpy as np

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)  # Hungarian algorithm
    best_fit = [col_ind[y_pred[i]] for i in range(y_pred.size)]
    return best_fit, list(zip(row_ind, col_ind)), w

def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def Accuracy(y, y_pred):
    acc = np.round(cluster_acc(y, y_pred), 5)
    return acc

def get_tptnfpfn(preds, y):
    #preds = clf.predict(data['x_test'])

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for real_line, pred_line in zip(y, preds):
        for real, pred in zip(real_line, pred_line):
            if pred == 1:
                if real == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if real == 1:
                    fn += 1
                else:
                    tn += 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

def get_accuracy(data):
    sum_ = (data['tp'] + data['tn'] + data['fp'] + data['fn'])
    return float(data['tp'] + data['tn']) / sum_

def NMI(y, y_pred):
    return normalized_mutual_info_score(y, y_pred)

from scipy.spatial.distance import cdist, pdist, squareform
def DB(X, y):
    """
    Davies-Bouldin index is an internal evaluation method for
    clustering algorithms. Lower values indicate tighter clusters that
    are better separated.
    """
    # get unique labels
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    uniqlbls = np.unique(y)
    n = len(uniqlbls)
    # pre-calculate centroid and sigma
    centroid_arr = np.empty((n, X.shape[1]))
    sigma_arr = np.empty((n,1))
    dbi_arr = np.empty((n,n))
    mask_arr = np.invert(np.eye(n, dtype='bool'))
    for i,k in enumerate(uniqlbls):
        Xk = X[np.where(y==k)[0],...]
        Ak = np.mean(Xk, axis=0)
        centroid_arr[i,...] = Ak
        sigma_arr[i,...] = np.mean(cdist(Xk, Ak.reshape(1,-1)))
    # compute pairwise centroid distances, make diagonal elements non-zero
    centroid_pdist_arr = squareform(pdist(centroid_arr)) + np.eye(n)
    # compute pairwise sigma sums
    sigma_psum_arr = squareform(pdist(sigma_arr, lambda u,v: u+v))
    # divide
    dbi_arr = np.divide(sigma_psum_arr, centroid_pdist_arr)
    # get mean of max of off-diagonal elements
    dbi_arr = np.where(mask_arr, dbi_arr, 0)
    dbi = np.mean(np.max(dbi_arr, axis=1))
    return dbi

def SC(d, y_pred):
    SC = metrics.silhouette_score(d, y_pred, metric="euclidean")
    return SC

def CH(d, y_pred):
    CH=metrics.calinski_harabasz_score(d, y_pred)
    return CH
