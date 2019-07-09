# distutils: language = c++

import sys


cdef class pyEnvironment:
  cdef Environment* this_

  def __cinit__(self):
    self.this_ = new Environment()
    print("PACMENSL session initialized.")

  def __dealloc__(self):
    if self.this_ != NULL:
      # TODO: finalize is never called!!!!
      # del self.this_
      print("PACMENSL session finalized.")