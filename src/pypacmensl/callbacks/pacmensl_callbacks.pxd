cdef extern from "PyCallbacksWrapper.hpp":
    cdef cppclass PyPropWrapper:
        PyPropWrapper()
        PyPropWrapper(object)

    cdef cppclass PyTFunWrapper:
        PyTFunWrapper()
        PyTFunWrapper(object)

    cdef cppclass PyConstrWrapper:
        PyConstrWrapper()
        PyConstrWrapper(object)
