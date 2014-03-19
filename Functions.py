#coding:latin_1
import numpy as np
import pywt
from math import sqrt, copysign


######### MATCHING PURSUIT, NO CRITERION #########

def mp_arbitrary_criterion(phi, phi_t, y, n_iter):
    x = np.zeros(phi_t.dot(y).shape[0])
    err = [np.linalg.norm(y)]

    for k in xrange(n_iter):
        c = phi_t.dot(y - phi.dot(x))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)
        x[i_0] += c[i_0]

        err.append(np.linalg.norm(y - phi.dot(x)))

    z = phi.dot(x)
    return z, err, n_iter


######### MATCHING PURSUIT, STATISTICAL CRITERION #########

def mp_stat_criterion(phi, phi_t, y, sigma):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    # 1st MP STEP
    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    dic = [phi.dot(np.identity(p)[:, i_0])]  # Temporary dictionary containing evaluated atoms
    dic_arr = np.asarray(dic).T
    t, q = (2, 1)

    while t > q:

        sparse[i_0] += c[i_0]
        err.append(np.linalg.norm(y - phi.dot(sparse)))

        #MP
        c = phi_t.dot(y - phi.dot(sparse))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(np.linalg.norm(x)-x.T.dot(h).dot(x)))

        s = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1])-1, 50)
        q = np.percentile(s, 95)

        dic.append(x)
        dic_arr = np.asarray(dic).T
        #print Dic_Arr.shape, i_0, C[i_0], err[-1]

    return phi.dot(sparse), err, dic_arr.shape[1]


######### ORTHOGONAL MATCHING PURSUIT, STAT CRITERION #########

def omp_stat_criterion(phi, phi_t, y, sigma):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T
    t, q = (2, 1)

    while t > q:

        sparse[i_0] += c[i_0]
        index = np.where(sparse)[0]
        sparse[index] = np.linalg.pinv(phi.dot(np.identity(p)[:, index])).dot(y)  # OMP projection
        err.append(np.linalg.norm(y - phi.dot(sparse)))

        c = phi_t.dot(y - phi.dot(sparse))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)

        x = phi.dot(np.identity(p)[:, i_0])

        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(np.linalg.norm(x)-x.T.dot(h).dot(x)))

        s = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1])-1, 50)
        q = np.percentile(s, 95)

        dic.append(x)
        dic_arr = np.asarray(dic).T
        #print Dic_Arr.shape, i_0, C[i_0], err[-1]

    return phi.dot(sparse), err, dic_arr.shape[1]


######### MEDIAN ABSOLUTE DEVIATION ######### 

#consistent estimator of the std deviation
#where hat(Sigma) a consistent estimator of the std dev. of the finest level detail coefficients.

def mad(data):
    return np.ma.median(np.abs(data - np.ma.median(data)))


######### HARD THRESHOLDING #########
#a) N-largest Hard Thresholding : verifies same sparsity constraint as the Matching Pursuit solution.

def hard_thresholding(phi, phi_t, y, n_thresh):

    c = phi_t.dot(y)
    thr = sorted(np.abs(c), reverse=True)[n_thresh-1]
    c = pywt.thresholding.hard(c, thr)
    #print np.flatnonzero(c).shape[0]
    return phi.dot(c)

#b) Hard Thresholding, using MP on pre-experiment noise :
# - We select atoms describing noise implementing a MP on y1 (signal pre-experiment)
# - we operate a hard thresholding by removing those from our explanatory model

#ONLY WORKS FOR TWO SIGNALS OF IDENTICAL SIZES.


def hard_thresholding_early_experiment(phi, phi_t, early_signal, currnt_signal):
    if early_signal.shape[0] == currnt_signal.shape[0]:

        p = phi_t.dot(early_signal).shape[0]
        sigma = float(sqrt(np.var(early_signal)))
        sparse = np.zeros(p)
        indexes = []

        c = phi_t.dot(early_signal)
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)
        indexes.append(i_0)

        dic = [phi.dot(np.identity(p)[:, i_0])]
        dic_arr = np.asarray(dic).T
        t, q = (2, 1)

        while t > q:

            sparse[i_0] += c[i_0]

            c = phi_t.dot(early_signal - phi.dot(sparse))
            abs_c = np.abs(c)
            i_0 = np.argmax(abs_c)
            indexes.append(i_0)

            x = phi.dot(np.identity(p)[:, i_0])
            h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
            t = np.abs(x.T.dot(early_signal-h.dot(early_signal)))/(sigma*sqrt(np.linalg.norm(x)-x.T.dot(h).dot(x)))

            s = np.random.standard_t(int(early_signal.shape[0]-dic_arr.shape[1])-1, 50)
            q = np.percentile(s, 95)

            dic.append(x)
            dic_arr = np.asarray(dic).T

        indexes.pop()
        return currnt_signal-phi.dot(sparse)

    else:
        return "Shapes dot not match. The present algorithm cannot be implemented."