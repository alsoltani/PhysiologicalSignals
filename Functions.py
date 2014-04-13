#coding:latin_1
from math import sqrt, log
import numpy as np
import pywt
import Classes
from Utility import gram_schmidt


#--------------------------------------------------------------------------------------------------------------#
#--------------------------------------------   SHANNON ENTROPY   ---------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def shannon(vect_unit):
    p = 0
    for j in xrange(vect_unit.shape[0]):
        if vect_unit[j] != 0:
            p += -vect_unit[j]*log(vect_unit[j], 2)
    return p

#--------------------------------------------------------------------------------------------------------------#
#----------------------------------------   SINGLE-CHANNEL PURSUITS   -----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#


#---------------------------------   MATCHING PURSUIT, ARBITRARY CRITERION   ----------------------------------#

def mp_arbitrary_criterion(phi, phi_t, y, n_iter):
    x = np.zeros(phi_t.dot(y).shape[0])
    err = [np.linalg.norm(y)]

    atoms_list = []

    for k in xrange(n_iter):
        c = phi_t.dot(y - phi.dot(x))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)
        x[i_0] += c[i_0]

        atoms_list.append(i_0)

        err.append(np.linalg.norm(y - phi.dot(x)))

    z = phi.dot(x)
    return x, z, err, atoms_list


#--------------------------------   MATCHING PURSUIT, STATISTICAL CRITERION   ------------------------------#

def mp_stat_criterion(phi, phi_t, y, sigma):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    atoms_list = []

    # 1st MP step.
    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    atoms_list.append(i_0)

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

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

        s = np.random.normal(0, 1, 50)
        q = np.percentile(s, 95)

        dic.append(x)
        dic_arr = np.asarray(dic).T
        #print Dic_Arr.shape, i_0, C[i_0], err[-1]

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


#----------------------------   ORTHOGONAL MATCHING PURSUIT, STATISTICAL CRITERION   -----------------------#

def omp_stat_criterion(phi, phi_t, y, sigma):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    atoms_list = []

    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    atoms_list.append(i_0)

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

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])

        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

        s = np.random.normal(0, 1, 50)
        q = np.percentile(s, 95)

        dic.append(x)
        dic_arr = np.asarray(dic).T
        #print Dic_Arr.shape, i_0, C[i_0], err[-1]

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


#-----------------------------   MP & OMP, STAT CRITERION, /W ENDOGENOUS VARIANCE   ------------------------#

def mp_stat_endogen_var(phi, phi_t, y):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    atoms_list = []

    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    atoms_list.append(i_0)

    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T

    t, q = (2, 1)

    while t > q:

        sparse[i_0] += c[i_0]
        err.append(np.linalg.norm(y - phi.dot(sparse)))

        c = phi_t.dot(y - phi.dot(sparse))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)

        #Estimating residual std. deviation, using OLS variance estimator s²
        s = sqrt(y.T.dot(y-h.dot(y))/(y.shape[0]-dic_arr.shape[1]))

        if x.T.dot(x)-x.T.dot(h).dot(x) > 0:
            t = np.abs(x.T.dot(y-h.dot(y)))/(s*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

            t_dist = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1]), 50)
            q = np.percentile(t_dist, 95)

        else:
            break

        #print y.shape[0], dic_arr.shape[1], s, t, q

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


def omp_stat_endogen_var(phi, phi_t, y):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    atoms_list = []

    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    atoms_list.append(i_0)

    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T
    t, q = (2, 1)

    while t > q:

        sparse[i_0] += c[i_0]
        index = np.flatnonzero(sparse).tolist()
        sparse[index] = np.linalg.pinv(phi.dot(np.identity(p)[:, index])).dot(y)  # OMP projection
        err.append(np.linalg.norm(y - phi.dot(sparse)))

        c = phi_t.dot(y - phi.dot(sparse))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)

        #Estimating residual std. deviation, using OLS variance estimator s²
        sigma_est = sqrt(y.T.dot(y-h.dot(y))/(y.shape[0]-dic_arr.shape[1]))

        if x.T.dot(x)-x.T.dot(h).dot(x) > 0:
            t = np.abs(x.T.dot(y-h.dot(y)))/(sigma_est*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

            t_dist = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1]), 50)
            q = np.percentile(t_dist, 95)

        else:
            break

        #print y.shape[0], dic_arr.shape[1], sigma_est, t, q

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


#--------------------------------------------------------------------------------------------------------------#
#-------------------------------------------   HARD THRESHOLDING   --------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

# N-largest Hard Thresholding : verifies same sparsity constraint as the Matching Pursuit solution.

def hard_thresholding(phi, phi_t, y, n_thresh):

    c = phi_t.dot(y)
    thr = sorted(np.abs(c), reverse=True)[n_thresh-1]
    c = pywt.thresholding.hard(c, thr)
    return phi.dot(c)


#--------------------------------------------------------------------------------------------------------------#
#--------------------------------------------   GOODNESS OF FIT   ---------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def classic_r2(level, algorithm, y, sigma, name_1, orth="no", name_2=None):
    results = []

    dic_t = Classes.DictT(level=level, name=name_1)
    mat_t_1 = dic_t.dot(np.identity(y.size))
    if orth == "yes":
        mat_t_1 = gram_schmidt(mat_t_1.T).T
    mat_t_1 /= np.sqrt(np.sum(mat_t_1 ** 2, axis=0))

    z_1, err_1, n_1 = algorithm(mat_t_1.T, mat_t_1, y, sigma)

    r2_1 = np.var(z_1)/np.var(y)
    results.append(r2_1)

    if name_2 is not None:
        dic_t = Classes.DictT(level=level, name=name_2)
        mat_t_2 = dic_t.dot(np.identity(y.size))
        if orth == "yes":
            mat_t_2 = gram_schmidt(mat_t_2.T).T
        mat_t_2 /= np.sqrt(np.sum(mat_t_2 ** 2, axis=0))

        z_2, err_2, n_2 = algorithm(mat_t_2.T, mat_t_2, y, sigma)

        r2_2 = np.var(z_2)/np.var(y)
        results.append(r2_2)

    return results


def ajusted_r2(level, algorithm, y, sigma, name_1, orth="no", name_2=None):
    results = []

    dic_t = Classes.DictT(level=level, name=name_1)
    mat_t_1 = dic_t.dot(np.identity(y.size))
    if orth == "yes":
        mat_t_1 = gram_schmidt(mat_t_1.T).T
    mat_t_1 /= np.sqrt(np.sum(mat_t_1 ** 2, axis=0))

    z_1, err_1, n_1 = algorithm(mat_t_1.T, mat_t_1, y, sigma)

    r2_1 = np.var(z_1)/np.var(y)
    results.append(r2_1 - (1 - r2_1)*len(n_1)/(y.size-len(n_1)-1))

    if name_2 is not None:
        dic_t = Classes.DictT(level=level, name=name_2)
        mat_t_2 = dic_t.dot(np.identity(y.size))
        if orth == "yes":
            mat_t_2 = gram_schmidt(mat_t_2.T).T
        mat_t_2 /= np.sqrt(np.sum(mat_t_2 ** 2, axis=0))

        z_2, err_2, n_2 = algorithm(mat_t_2.T, mat_t_2, y, sigma)

        r2_2 = np.var(z_2)/np.var(y)
        results.append(r2_2 - (1 - r2_2)*len(n_2)/(y.size-len(n_2)-1))

    return results


#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   DICTIONARY TESTING   --------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def dictionary_testing(level, name_1, name_2, algorithm, y, sigma, orth="no"):

    dic_t = Classes.DictT(level=level, name=name_1)
    mat_t_1 = dic_t.dot(np.identity(y.size))
    if orth == "yes":
        mat_t_1 = gram_schmidt(mat_t_1.T).T
    mat_t_1 /= np.sqrt(np.sum(mat_t_1 ** 2, axis=0))

    dic_t = Classes.DictT(level=level, name=name_2)
    mat_t_2 = dic_t.dot(np.identity(y.size))
    if orth == "yes":
        mat_t_2 = gram_schmidt(mat_t_2.T).T
    mat_t_2 /= np.sqrt(np.sum(mat_t_2 ** 2, axis=0))

    z_1, err_1, n_1 = algorithm(mat_t_1.T, mat_t_1, y, sigma)
    z_2, err_2, n_2 = algorithm(mat_t_2.T, mat_t_2, y, sigma)

    print "R2, Model 1 :" + str(np.var(z_1)/(np.var(y)))
    print "R2, Model 2 :" + str(np.var(z_2)/(np.var(y)))
    print "Number of parameters :" + str(len(n_1)) + " (Model 1), " + str(len(n_2)) + " (Model 2)."

    if len(n_1) == len(n_2):
        print "Model have same the same number of parameters.\n"
        print "Null hypothesis : first model fits better the data.\n"
        f_test = pow(err_1[-1], 2)/pow(err_2[-1], 2)
        s = np.random.f(y.size-len(n_1), y.size-len(n_2), 50)
        q = np.percentile(s, 95)
        print "F-statistic : " + str(f_test)
        print "95th-quantile : " + str(q)

        return f_test, q

    if len(n_2) > len(n_1):
        print "Model 2 has a larger number of parameters than 1.\n"
        print "Null hypothesis : most complex model fits better the data.\n"
        f_test = (pow(err_1[-1], 2) - pow(err_2[-1], 2))*(y.size-len(n_2))
        f_test /= pow(err_2[-1], 2)*(len(n_2)-len(n_1))

        s = np.random.f(len(n_2)-len(n_1), y.size-len(n_2), 50)
        q = np.percentile(s, 95)
        print "F-statistic : " + str(f_test)
        print "95th-quantile : " + str(q)

        return f_test, q

    else:
        print "Model 1 has a larger number of parameters than 2."
        print "Null hypothesis : most complex model fits better the data.\n"
        f_test = (pow(err_2[-1], 2) - pow(err_1[-1], 2))*(y.size-len(n_1))
        f_test /= pow(err_1[-1], 2)*(len(n_1)-len(n_2))

        s = np.random.f(len(n_1)-len(n_2), y.size-len(n_1), 50)
        q = np.percentile(s, 95)
        print "F-statistic : " + str(f_test)
        print "95th-quantile : " + str(q)

        return f_test, q


#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------------   MULTI-CHANNEL PURSUITS   -----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

#--------------------------------------------   MULTI-CHANNEL MP   -----------------------------------------#

def multi_channel_mp(phi, phi_t, matrix_y, vect_sigma):

    j = matrix_y.shape[1]  # Number of noisy signals

    p = phi_t.dot(matrix_y).shape[0]  # Number of sparse matrix rows
    matrix_sparse = np.zeros((p, j))
    err = [np.linalg.norm(matrix_y)]  # Frobenius norm

    atoms_list = []

    # 1st MP step.
    matrix_c = phi_t.dot(matrix_y)
    corr_list = np.zeros(p)
    for i in xrange(p):
        corr_list[i] = np.linalg.norm(matrix_c[i, :])

    i_0 = np.argmax(np.abs(corr_list))

    atoms_list.append(i_0)

    dic = [phi.dot(np.identity(p)[:, i_0])]  # Temporary dictionary containing evaluated atoms
    dic_arr = np.asarray(dic).T
    k, q = (2, 1)

    while k > q:

        matrix_sparse[i_0, :] += matrix_c[i_0, :]
        err.append(np.linalg.norm(matrix_y - phi.dot(matrix_sparse)))

        #MP
        matrix_c = phi_t.dot(matrix_y - phi.dot(matrix_sparse))
        corr_list = np.zeros(p)
        for i in xrange(p):
            corr_list[i] = np.linalg.norm(matrix_c[i, :])

        i_0 = np.argmax(np.abs(corr_list))

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)
        t = np.zeros(j)
        for i in xrange(len(t)):
            t[i] = (np.abs(x.T.dot(matrix_y[:, i]-h.dot(matrix_y[:, i])))
                    / (vect_sigma[i]*sqrt(pow(np.linalg.norm(x), 2)-x.T.dot(h).dot(x))))

        k = pow(np.linalg.norm(t), 2)
        s = np.random.chisquare(j, 50)
        q = np.percentile(s, 95)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return matrix_sparse, phi.dot(matrix_sparse), err, atoms_list