from libcpp cimport bool
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