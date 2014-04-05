import numpy as np
from numpy.linalg import inv
from math import sqrt

    
#--------------------------------   GRAM-SCHMIDT ALGORITHM   ------------------------------#

#Takes as input a set of vectors, stored row-wise.


def gram_schmidt(vecs, row_wise_storage=True):
    
    vecs = np.asarray(vecs)  # transform to array if list of vectors
    m, n = vecs.shape
    basis = np.array(vecs.T)
    eye = np.identity(n).astype(float)
    
    basis[:, 0] /= sqrt(np.dot(basis[:, 0], basis[:, 0]))
    for i in range(1, m):
        v = basis[:, i]/sqrt(np.dot(basis[:, i], basis[:, i]))
        u = basis[:, :i]
        p = eye - np.dot(u, np.dot(inv(np.dot(u.T, u)), u.T))
        basis[:, i] = np.dot(p, v)
        basis[:, i] /= sqrt(np.dot(basis[:, i], basis[:, i]))

    return basis.T if row_wise_storage else basis
