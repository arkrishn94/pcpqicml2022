import numpy as np
from tqdm.auto import tqdm 

# Max dot product is one line in python with numpy...
def brute_force_mips(X, q, k):
    return np.argpartition(X@q, -k)[-k:]

#Returns the recall as a sorted array 
def test_score_recall(X, Q, exact_scores, k, N, index, eps_scaling, *indexqueryargs): #exact_scores needs to be sorted.
    assert(exact_scores.shape[0] == Q.shape[0]) #|exact_scores| = |Q| .
    assert(exact_scores.shape[1] >= k) #Should have at least k scores for eahc query. Sorted order!
    recalls = []  
    for i in range(exact_scores.shape[0]):
        approx = index.query(X, Q[i], *indexqueryargs)
        recall = np.count_nonzero(eps_scaling*(X[np.sort(approx)]@(-Q[i])) <= exact_scores[i][k-1] )/k
        recalls.append(recall)
    recalls.sort()
    return recalls

def test_recall(X, Q, k, index):
    recalls = []
    for i in range(Q.shape[0]):
        exact = set(brute_force_mips(X, Q[i], k=k))
        approx = set(index.query(Q[i], k))
        recall = len(exact.intersection(approx))/k
        recalls.append(recall)
    recalls.sort()
    return recalls

def test_ips_error(X, Q, index): 
    recall_errs = np.zeros((X.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]): 
        exact = X@Q[i]
        approx = index.query_angular_dists(Q[i])
        recall_errs[:, i] = np.abs(1. - (approx/exact)) #Make safe to NaN
    return recall_errs