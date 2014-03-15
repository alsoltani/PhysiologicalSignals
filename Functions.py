#coding:latin_1
import numpy as np
import pywt
from math import sqrt


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
        t = np.abs(x.T.dot(y))/(sigma*sqrt(np.linalg.norm(x)-x.T.dot(h).dot(x)))

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
        t = np.abs(x.T.dot(y))/(sigma*sqrt(np.linalg.norm(x)-x.T.dot(h).dot(x)))

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

#NB : equiv. to OMP when using orthogonal dictionary.
#We can use the following threshold :
#thresh = sigma*np.sqrt(2*np.log(len(y)))

#a) Hard Thresholding using Wavelet decomposition

def hard_thresholding_wavedec(y, name, lvl, thresh):

    c = pywt.wavedec(y, name, level=lvl)
    denoised = c[:]
    denoised[1:] = (pywt.thresholding.hard(i, value=thresh) for i in denoised[1:])
    return pywt.waverec(denoised, name)


#b) Hard Thresholding using Classes

def hard_thresholding_operator(phi, phi_t, y, thresh):

    c = phi_t.dot(y)
    c = pywt.thresholding.hard(c, value=thresh)
    return phi.dot(c)
