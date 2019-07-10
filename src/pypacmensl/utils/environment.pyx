# distutils: language = c++
import atexit

cdef class pyEnvironment:

  def __cinit__(self):
    self.this_ = new Environment()
    # print("PACMENSL session initialized.")

  def __dealloc__(self):
    if self.this_ != NULL:
      del self.this_
      # print("PACMENSL session finalized.")

_PACMENSL_GLOBAL_ENV = pyEnvironment()


