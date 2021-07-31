# from libcpp.functional cimport function

ctypedef int(*PropXFun)(const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args)
ctypedef int(*PropTFun)(double t, int num_coefs, double* outputs, void* args)
ctypedef int(*DPropXFun)(const int parameter_idx, const int reaction_idx, const int num_species, const int num_states, const int* states, double* outputs, void* args)
ctypedef int(*DPropTFun)(const int parameter_idx, const double t, int num_coefs, double *outputs, void *args)

ctypedef int(*ConstrFun)(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args)
ctypedef int(*WeightFun)(int num_species, int* x, int nout, double* wx, void* args)