from ctypes import cdll
from ctypes import c_int

hello_lib = cdll.LoadLibrary("example.so")
test_func = hello_lib.test_func
test_func.restype = c_int

print(test_func(1, 1))