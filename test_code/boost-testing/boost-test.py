from ctypes import cdll
from ctypes import c_int_p

hello_lib = cdll.LoadLibrary("example.so")
fact = hello_lib.Foo
fact.restype = c_int_p

print(fact)