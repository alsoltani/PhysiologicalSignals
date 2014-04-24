#coding:latin_1
from math import sqrt, log, floor
import numpy as np
import pywt
import Classes
from Utility import gram_schmidt


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

def mp_stat_criterion(phi, phi_t, y, sigma, significance_level=5):
    p = phi_t.dot(y).shape[0]
    sparse = np.zeros(p)
    err = [np.linalg.norm(y)]

    atoms_list = []

    # Compute 1st MP step.
    c = phi_t.dot(y)
    abs_c = np.abs(c)
    i_0 = np.argmax(abs_c)

    atoms_list.append(i_0)

    # Temporary dictionary containing evaluated atoms.
    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T
    t, q = (2, 1)

    while t > q:

        sparse[i_0] += c[i_0]
        err.append(np.linalg.norm(y - phi.dot(sparse)))

        # Compute MP step.
        c = phi_t.dot(y - phi.dot(sparse))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)

        atoms_list.append(i_0)

        x = phi.dot(np.identity(p)[:, i_0])
        h = dic_arr.dot(np.linalg.inv(dic_arr.T.dot(dic_arr))).dot(dic_arr.T)

        if x.T.dot(x)-x.T.dot(h).dot(x) <= 0:
            break

        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

        s = np.random.normal(0, 1, 1e3)
        q = np.percentile(s, 100-significance_level/2)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


#----------------------------   ORTHOGONAL MATCHING PURSUIT, STATISTICAL CRITERION   -----------------------#

def omp_stat_criterion(phi, phi_t, y, sigma, significance_level=5):
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

        if x.T.dot(x)-x.T.dot(h).dot(x) <= 0:
            break

        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))

        s = np.random.normal(0, 1, 1e3)
        q = np.percentile(s, 100-significance_level/2)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


#-----------------------------   MP & OMP, STAT CRITERION, /W ENDOGENOUS VARIANCE   ------------------------#

def mp_stat_endogen_var(phi, phi_t, y, significance_level=5):
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

        if x.T.dot(x)-x.T.dot(h).dot(x) <= 0:
            break

        t = np.abs(x.T.dot(y-h.dot(y)))/(s*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))
        t_dist = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1]), 1e3)
        q = np.percentile(t_dist, 100 - significance_level/2)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return sparse, phi.dot(sparse), err, atoms_list


def omp_stat_endogen_var(phi, phi_t, y, significance_level=5):
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

        if x.T.dot(x)-x.T.dot(h).dot(x) <= 0:
            break

        t = np.abs(x.T.dot(y-h.dot(y)))/(sigma_est*sqrt(x.T.dot(x)-x.T.dot(h).dot(x)))
        t_dist = np.random.standard_t(int(y.shape[0]-dic_arr.shape[1]), 1e3)
        q = np.percentile(t_dist, 100 - significance_level/2)

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
#-----------------------------------------   MULTI-CHANNEL PURSUITS   -----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

#--------------------------------------------   MULTI-CHANNEL MP   -----------------------------------------#

def multi_channel_mp(phi, phi_t, matrix_y, vect_sigma, significance_level=5):

    j = matrix_y.shape[1]  # Number of noisy signals

    p = phi_t.dot(matrix_y).shape[0]  # Number of sparse matrix rows
    matrix_sparse = np.zeros((p, j))
    err = [np.linalg.norm(matrix_y)]  # Frobenius norm

    atoms_list = []

    matrix_c = phi_t.dot(matrix_y)
    corr_list = np.zeros(p)
    for i in xrange(p):
        corr_list[i] = np.linalg.norm(matrix_c[i, :])

    i_0 = np.argmax(np.abs(corr_list))

    atoms_list.append(i_0)

    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T
    k, q = (2, 1)

    while k > q:

        matrix_sparse[i_0, :] += matrix_c[i_0, :]
        err.append(np.linalg.norm(matrix_y - phi.dot(matrix_sparse)))

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
        s = np.random.chisquare(j, 1e3)
        q = np.percentile(s, 100-significance_level/2)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return matrix_sparse, phi.dot(matrix_sparse), err, atoms_list


def multi_channel_omp(phi, phi_t, matrix_y, vect_sigma, significance_level=5):

    j = matrix_y.shape[1]

    p = phi_t.dot(matrix_y).shape[0]
    matrix_sparse = np.zeros((p, j))
    err = [np.linalg.norm(matrix_y)]

    atoms_list = []

    matrix_c = phi_t.dot(matrix_y)
    corr_list = np.zeros(p)
    for i in xrange(p):
        corr_list[i] = np.linalg.norm(matrix_c[i, :])

    i_0 = np.argmax(np.abs(corr_list))

    atoms_list.append(i_0)

    dic = [phi.dot(np.identity(p)[:, i_0])]
    dic_arr = np.asarray(dic).T
    k, q = (2, 1)

    while k > q:

        matrix_sparse[i_0, :] += matrix_c[i_0, :]
        index = np.where(matrix_sparse[:, 0])[0]
        matrix_sparse[index] = np.linalg.pinv(phi.dot(np.identity(p)[:, index])).dot(matrix_y)

        err.append(np.linalg.norm(matrix_y - phi.dot(matrix_sparse)))

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
        s = np.random.chisquare(j, 1e3)
        q = np.percentile(s, 100-significance_level/2)

        dic.append(x)
        dic_arr = np.asarray(dic).T

    del atoms_list[-1]

    return matrix_sparse, phi.dot(matrix_sparse), err, atoms_list


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

    x_1, z_1, err_1, n_1 = algorithm(mat_t_1.T, mat_t_1, y, sigma)

    r2_1 = np.var(z_1)/np.var(y)
    results.append(r2_1)

    if name_2 is not None:
        dic_t = Classes.DictT(level=level, name=name_2)
        mat_t_2 = dic_t.dot(np.identity(y.size))
        if orth == "yes":
            mat_t_2 = gram_schmidt(mat_t_2.T).T
        mat_t_2 /= np.sqrt(np.sum(mat_t_2 ** 2, axis=0))

        x_2, z_2, err_2, n_2 = algorithm(mat_t_2.T, mat_t_2, y, sigma)

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

    x_1, z_1, err_1, n_1 = algorithm(mat_t_1.T, mat_t_1, y, sigma)

    r2_1 = np.var(z_1)/np.var(y)
    results.append(r2_1 - (1 - r2_1)*len(n_1)/(y.size-len(n_1)-1))

    if name_2 is not None:
        dic_t = Classes.DictT(level=level, name=name_2)
        mat_t_2 = dic_t.dot(np.identity(y.size))
        if orth == "yes":
            mat_t_2 = gram_schmidt(mat_t_2.T).T
        mat_t_2 /= np.sqrt(np.sum(mat_t_2 ** 2, axis=0))

        x_2, z_2, err_2, n_2 = algorithm(mat_t_2.T, mat_t_2, y, sigma)

        r2_2 = np.var(z_2)/np.var(y)
        results.append(r2_2 - (1 - r2_2)*len(n_2)/(y.size-len(n_2)-1))

    return results

#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   DICTIONARY TESTING   --------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#


def cross_val_procedure(basis, basis_t, database_1, database_2, vect_sigma, significance_level=5):
    cv_out = 0

    # ----------------- PHASE 1 - Training : database_1, Testing : database_2. ----------------- #

    # 1. Select significant atoms for database_1.
    x_train_1, z_train_1, err_train_1, n_train_1 = \
        multi_channel_omp(basis, basis_t, database_1, vect_sigma, significance_level)

    # 2. Explain database_2 with same atoms. (HT)
    z_temp_1 = basis_t.dot(database_2)
    z_temp_1[list(set(xrange(database_2.shape[0]))-set(n_train_1))] = 0
    z_temp_1 = basis_t.T.dot(z_temp_1)

    # 3. Explain database_2 using OMP procedure.
    x_test_1, z_test_1, err_test_1, n_test_1 = \
        multi_channel_omp(basis, basis_t, database_2, vect_sigma, significance_level)

    cv_out += np.linalg.norm(z_test_1-z_temp_1)

    # ----------------- PHASE 2 - Training : database_2, Testing : database_1. ----------------- #

    x_train_2, z_train_2, err_train_2, n_train_2 = \
        multi_channel_omp(basis, basis_t, database_2, vect_sigma, significance_level)

    z_temp_2 = basis_t.dot(database_1)
    z_temp_2[list(set(xrange(database_1.shape[0]))-set(n_train_2))] = 0
    z_temp_2 = basis_t.T.dot(z_temp_2)

    x_test_2, z_test_2, err_test_2, n_test_2 = \
        multi_channel_omp(basis, basis_t, database_1, vect_sigma, significance_level)

    cv_out += np.linalg.norm(z_test_2-z_temp_2)

    return cv_out


def cross_validation(matrix_signal, vect_sigma, wavelet_family, significance_level=5):
    matrix_signal_sh = np.copy(matrix_signal)
    #np.random.shuffle(matrix_signal_sh)

    cross_val = []
    wavelet_indexes = []

    training_database = matrix_signal_sh[:floor(matrix_signal.shape[0]/2)]
    testing_database = matrix_signal_sh[floor(matrix_signal.shape[0]/2):]

    for i in xrange(len(pywt.wavelist(wavelet_family))):
        dictionary_t = Classes.DictT(level=None, name=wavelet_family+str(i+1))
        ortho_basis_t = gram_schmidt(dictionary_t.dot(np.identity(training_database.shape[0])).T).T
        ortho_basis_t /= np.sqrt(np.sum(ortho_basis_t ** 2, axis=0))

        cross_val.append(cross_val_procedure(ortho_basis_t.T, ortho_basis_t,
                                             training_database, testing_database, vect_sigma, significance_level))

        wavelet_indexes.append(i+1)

    return cross_val, wavelet_indexes


def bic_criterion(matrix_signal, vect_sigma, wavelet_family, ortho='no', significance_level=5):
    bic_values = []
    wavelet_indexes = []

    for i in xrange(len(pywt.wavelist(wavelet_family))):
        dictionary_t = Classes.DictT(level=None, name=wavelet_family+str(i+1))
        ortho_basis_t = dictionary_t.dot(np.identity(matrix_signal.shape[0]))
        if ortho == 'yes':
            ortho_basis_t = gram_schmidt(ortho_basis_t.T).T
        ortho_basis_t /= np.sqrt(np.sum(ortho_basis_t ** 2, axis=0))

        matrix_x, matrix_z, err, n = multi_channel_omp(ortho_basis_t.T, ortho_basis_t, matrix_signal, vect_sigma,
                                                       significance_level)

        bic_values.append(pow(np.linalg.norm(matrix_signal-matrix_z), 2)/matrix_signal.shape[0]
                          + len(n)*np.log(matrix_signal.shape[0])/matrix_signal.shape[0])
        wavelet_indexes.append(i+1)

    return bic_values, wavelet_indexes


#--------------------------------------------------------------------------------------------------------------#
#--------------------------------------------   ENTROPY FUNCTIONS   ---------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def shannon(vect_unit):
    p = 0
    for j in xrange(vect_unit.shape[0]):
        if vect_unit[j] != 0:
            p += -pow(vect_unit[j], 2)*log(pow(vect_unit[j], 2))
    return p


def sum_squares(vect_unit):
    p = 0
    for j in xrange(vect_unit.shape[0]):
        if vect_unit[j] != 0:
            p += pow(vect_unit[j], 2)
    return p

