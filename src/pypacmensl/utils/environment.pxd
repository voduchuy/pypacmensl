cdef extern from "Python.h":
  int Py_AtExit(void (*) ())

cdef extern from "pacmensl.h" namespace "pacmensl":
  int PACMENSLInit(int *argc, char ***argv, const char *help)
  int PACMENSLFinalize()

  cdef cppclass Environment:
    Environment()
    Environment(int *argc, char ***argv, const char *help)

cdef class pyEnvironment:
  cdef Environment* this_
