# distutils: language = c++
from cpython.ref cimport PyObject

cdef public int call_py_prop_obj(const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args)

cdef public int call_py_t_fun_obj (double t, int num_coefs, double* outputs, void* args)

cdef public int call_py_constr_obj(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args)

cdef public int call_py_weight_fun(int num_species, int* x, int nout, double* fout, void* args)