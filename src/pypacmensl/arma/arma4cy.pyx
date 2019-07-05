# distutils: language = c++
cimport pypacmensl.arma.arma4cy as arma
cimport numpy as cnp

cdef public arma.Mat[double] MakeDoubleMat(cnp.ndarray np_mat):
    cdef int n_rows
    if np_mat.ndim == 1:
        n_rows = 1
    else:
        n_rows = np_mat.shape[1]
    cdef arma.Mat[double] mat = arma.Mat[double](<double*> np_mat.data, n_rows, np_mat.shape[0])
    return mat

cdef public arma.Mat[int] MakeIntMat(cnp.ndarray np_mat):
    cdef int n_rows
    if np_mat.ndim == 1:
        n_rows = 1
    else:
        n_rows = np_mat.shape[1]
    cdef arma.Mat[int] mat = arma.Mat[int](<int*> np_mat.data, n_rows, np_mat.shape[0])
    return mat

cdef public arma.Row[double] MakeDoubleRow(cnp.ndarray np_vec):
    cdef arma.Row[double] y = arma.Row[double](<double*> np_vec.data, np_vec.size)
    return y

cdef public arma.Row[int] MakeIntRow(cnp.ndarray np_vec):
    cdef arma.Row[int] y = arma.Row[int](<int*> np_vec.data, np_vec.size)
    return y

cdef public arma.Col[double] MakeDoubleCol(cnp.ndarray np_vec):
    cdef arma.Col[double] y = arma.Col[double](<double*> np_vec.data, np_vec.size)
    return y

cdef public arma.Col[int] MakeIntCol(cnp.ndarray np_vec):
    cdef arma.Col[int] y = arma.Col[int](<int*> np_vec.data, np_vec.size)
    return y