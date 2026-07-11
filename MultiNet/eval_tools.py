import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mean_squared_error
from scipy.stats import pearsonr
nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def clustering_metric(y_true, y_pred):
    return nmi(y_true, y_pred), ari(y_true, y_pred)


# IMPUTATION METRICS
def pcc(X_mean, X, i=None, j=None, ix=None):
    if i is None or j is None or ix is None:
        x = np.reshape(X_mean, -1)
        y = np.reshape(X, -1)
    else:
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]

    return pearsonr(x, y)

def imputation_error(X_mean, X, i=None, j=None, ix=None):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    if i is None or j is None or ix is None:
        x = np.reshape(X_mean, -1)
        y = np.reshape(X, -1)
    else:
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]

    # L1-distance
    l1_distance = np.median(np.abs(x - y))

    # Cosine similarity
    cosine_similarity = np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-10)

    # RMSE
    rmse = np.sqrt(mean_squared_error(x, y))

    return {"imputation_l1":l1_distance,
            "imputation_cosine":cosine_similarity,
            "imputation_rmse":rmse}


def dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    i, j = np.nonzero(X_zero)

    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)

    # choice number 2, focus on a few but corrupt binomially
    # ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    # X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix
