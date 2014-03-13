import numpy as np
    
######### GRAM-SCHMIDT ALGORITHM #########

#Takes as input a set of vectors, stored row-wise.

def Gram_Schmidt(vecs, row_wise_storage=True):

    from numpy.linalg import inv
    from math import sqrt
    
    vecs = np.asarray(vecs)  # transform to array if list of vectors
    m, n = vecs.shape
    basis = np.array(vecs.T)
    eye = np.identity(n).astype(float)
    
    basis[:,0] /= sqrt(np.dot(basis[:,0], basis[:,0]))
    for i in range(1, m):
        v = basis[:,i]/sqrt(np.dot(basis[:,i], basis[:,i]))
        U = basis[:,:i]
        P = eye - np.dot(U, np.dot(inv(np.dot(U.T, U)), U.T))
        basis[:, i] = np.dot(P, v)
        basis[:, i] /= sqrt(np.dot(basis[:, i], basis[:, i]))

    return basis.T if row_wise_storage else basis
    