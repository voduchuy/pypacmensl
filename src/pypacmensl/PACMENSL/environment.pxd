cdef extern from "pacmensl.h" namespace "pacmensl":
  int PACMENSLInit(int *argc, char ***argv, const char *help)
  int PACMENSLFinalize()
  cdef cppclass Environment:
    Environment();
    Environment(int *argc, char ***argv, const char *help);
