# from libcpp.functional cimport function

# ctypedef function[int (const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args)] PropXFun
# ctypedef function[int (double t, int num_coefs, double* outputs, void* args)] PropTFun
# ctypedef function[int (int num_species, int num_constr, int n_states, int *states, int *outputs, void *args)] ConstrFun

ctypedef int(*PropXFun)(const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args)
ctypedef int(*PropTFun)(double t, int num_coefs, double* outputs, void* args)
ctypedef int(*ConstrFun)(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args)
ctypedef int(*WeightFun)(int num_species, int* x, int nout, double* wx, void* args)