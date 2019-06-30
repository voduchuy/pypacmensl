cimport numpy as cnp
import numpy as np

cdef public int call_py_prop_obj(object, const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args) except -1:
    cdef int[:,::1] state_view = <int[:num_states,:num_species]> states
    cdef double[::1] out_view = <double[:num_states]> outputs
    state_np = np.asarray(state_view)
    out_np = np.asarray(out_view)
    cdef double propensity
    try:
        object(reaction, state_np, out_np)
    except:
        return -1
    return 0

cdef public int call_py_t_fun_obj (object, double t, int num_coefs, double* outputs, void* args) except -1:
    cdef double[::1] out_view = <double[:num_coefs]> outputs
    out_np = np.asarray(out_view)
    try:
        object(t, out_np)
    except:
        return -1
    return 0