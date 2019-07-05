from libcpp cimport bool
cimport numpy as cnp

cdef extern from "armadillo" namespace "arma":
    cdef cppclass Mat[T]:
        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero
        # constructors
        Mat()
        Mat(int n_rows, int n_cols)
        Mat(T* aux_mem, int aux_nrows, int aux_ncols)
        Mat(T* aux_mem, int aux_nrows, int aux_ncols, const bool copy, const bool strict)
        # methods
        T* memptr()
        T* colptr()

    cdef cppclass Col[T]:
        # attributes
        int n_elem
        # constructors
        Col()
        Col(int n_elem)
        Col(T* aux_mem, int n_elem)
        Col(T* aux_mem, int n_elem, const bool copy, const bool strict)
        # methods
        T* memptr()

    cdef cppclass Row[T]:
        # attributes
        int n_elem
        # constructors
        Row()
        Row(int n_elem)
        Row(T* aux_mem, int n_elem)
        Row(T* aux_mem, int n_elem, const bool copy, const bool strict)
        # methods
        T* memptr()

cdef Mat[double] MakeDoubleMat(cnp.ndarray np_mat)

cdef Mat[int] MakeIntMat(cnp.ndarray np_mat)

cdef Row[double] MakeDoubleRow(cnp.ndarray np_vec)

cdef Row[int] MakeIntRow(cnp.ndarray np_vec)

cdef Col[double] MakeDoubleCol(cnp.ndarray np_vec)

cdef Col[int] MakeIntCol(cnp.ndarray np_vec)