from ctypes import cdll
from ctypes import c_int

hello_lib = cdll.LoadLibrary("measure_feedback.so")
main = hello_lib.main
main.restype = c_int

print(main(0))