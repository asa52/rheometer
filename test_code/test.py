"""Imports useful functions from Arduino code to enable values to be read from 
and written to the .so file. Note the .so file can only be run in Linux."""

from ctypes import cdll
from ctypes import c_int
import numpy as np

# Load the source object and import the relevant functions to enable reading
# and writing of global variables.
arduino_code = cdll.LoadLibrary("test.so")
inc_t = arduino_code.main
inc_t.restype = c_int  # Specify the return type of the function.
get_t = arduino_code.get_t
get_t.restype = c_int

for i in range(10):
    print(get_t())
    inc_t()



