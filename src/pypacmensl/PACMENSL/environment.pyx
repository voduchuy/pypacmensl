cimport environment

# cdef class _Environment:
#   cdef environment.Environment* this_
#
#   def __cinit__(self):
#     self.this_ = new environment.Environment()
#     print("PACMENSL session initialized.")
#
#   def __dealloc__(self):
#     if self.this_ != NULL:
#       del self.this_
#       print("PACMENSL session finalized.")

cdef int initialize():
  cdef int argc = 0
  cdef char** argv
  return environment.PACMENSLInit(&argc, &argv, <char*> 0)