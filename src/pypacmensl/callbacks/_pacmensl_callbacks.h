/* Generated by Cython 0.29.10 */

#ifndef __PYX_HAVE__pypacmensl__include___pacmensl_callbacks
#define __PYX_HAVE__pypacmensl__include___pacmensl_callbacks


#ifndef __PYX_HAVE_API__pypacmensl__include___pacmensl_callbacks

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C int call_py_prop_obj(PyObject *, int const , int const , int const , int const *, double *, void *);
__PYX_EXTERN_C int call_py_t_fun_obj(PyObject *, double, int, double *, void *);
__PYX_EXTERN_C int call_py_constr_obj(PyObject *, int, int, int, int *, int *, void *);

#endif /* !__PYX_HAVE_API__pypacmensl__include___pacmensl_callbacks */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_pacmensl_callbacks(void);
#else
PyMODINIT_FUNC PyInit__pacmensl_callbacks(void);
#endif

#endif /* !__PYX_HAVE__pypacmensl__include___pacmensl_callbacks */