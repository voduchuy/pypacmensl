# distutils: language = c++

import sys


cdef class _Environment:
  cdef Environment* this_

  def __cinit__(self):
    self.this_ = new Environment()
    print("PACMENSL session initialized.")

  def __dealloc__(self):
    if self.this_ != NULL:
      del self.this_
      print("PACMENSL session finalized.")
#
# cdef void pyPACMENSLFinalize():
#   cdef int ierr
#   ierr = PACMENSLFinalize()
#   if ierr:
#     raise RuntimeError("Error encountered when trying to finalize PACMENSL session.")
#
# cpdef int initialize():
#   cdef:
#     int argc = 0
#     int ierr
#
#   ierr = Py_AtExit(pyPACMENSLFinalize)
#   if ierr:
#     raise ImportError("Cannot register PACMENSLFinalize().")
#   ierr = PACMENSLInit(&argc, NULL, <char*> 0)
#   if ierr:
#     raise ImportError("PACMENSLInit returns error.")