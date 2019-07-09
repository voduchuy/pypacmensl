# distutils: language = c++
import numpy as np

cdef int call_py_prop_obj(const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args):
    cdef PyObject* obj = <PyObject*> args
    if args == NULL:
        print('Null object passed to call_py_prop_obj.')
        return -1
    cdef int[:,::1] state_view = <int[:num_states,:num_species]> states
    cdef double[::1] out_view = <double[:num_states]> outputs
    state_np = np.asarray(state_view)
    out_np = np.asarray(out_view)
    try:
        (<object>obj)(reaction, state_np, out_np)
    except:
        return -1
    return 0

cdef int call_py_t_fun_obj (double t, int num_coefs, double* outputs, void* args):
    cdef PyObject* obj = <PyObject*> args
    if args == NULL:
        print('Null object passed to call_py_prop_obj.')
        return -1
    cdef double[::1] out_view = <double[:num_coefs]> outputs
    out_np = np.asarray(out_view)
    try:
        (<object>obj)(t, out_np)
    except:
        return -1
    return 0

cdef int call_py_constr_obj(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args):
    cdef PyObject* obj = <PyObject*> args
    if args == NULL:
        print('Null object passed to call_py_prop_obj.')
        return -1
    cdef int[:,::1] states_view = <int[:n_states, :num_species]> states
    cdef int[:,::1] outputs_view = <int[:n_states, :num_constr]> outputs
    states_np = np.asarray(states_view)
    outputs_np = np.asarray(outputs_view)
    try:
        (<object>obj)(states_np, outputs_np)
    except:
        return -1
    return 0