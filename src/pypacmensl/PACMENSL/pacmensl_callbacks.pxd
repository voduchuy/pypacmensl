cdef extern from "PyCallbacksWrapper.h":
    cdef cppclass PyPropWrapper:
        PyPropWrapper()
        PyPropWrapper(object)

    cdef cppclass PyTFunWrapper:
        PyTFunWrapper()
        PyTFunWrapper(object)