"""Imports useful functions from Arduino code to enable values to be read from 
and written to the .so file. Note the .so file can only be run in Linux."""

from ctypes import cdll
from ctypes import c_int
import numpy as np

# Load the source object and import the relevant functions to enable reading
# and writing of global variables.
arduino_code = cdll.LoadLibrary("measure_feedback.so")
get_torque = arduino_code.main
get_torque.restype = c_int  # Specify the return type of the function.
set_k_b_primes = arduino_code.set_k_b
set_amp = arduino_code.set_amp
set_amp.restype = c_int
get_k_prime = arduino_code.get_k
get_k_prime.restype = c_int
get_b_prime = arduino_code.get_b
get_b_prime.restype = c_int
get_amp = arduino_code.get_amp
get_amp.restype = c_int
get_mu = arduino_code.get_mu
get_mu.restype = c_int
get_dmudt = arduino_code.get_dmudt
get_dmudt.restype = c_int


def main():
    #results = []
    #for t in range(120):
    #    results.append([t, get_torque(t)])
    #results = np.array(results)
    #set_k_b(3, 4)
    #print(get_k())
    #print(get_b())
    #set_amp(3)
    #print(get_amp())
    #np.savetxt('test.txt', r# esults)
    print(get_k_prime(), get_b_prime())

def to_voltage(dmless_value, res_bit=12, min_voltage=0.55, max_voltage=2.75):
    """Convert a digitised value to a voltage between min_voltage and 
    max_voltage.
    :param dmless_value: The value to convert to a voltage.
    :param res_bit: The resolution of the apparatus.
    :return: The voltage equivalent in V."""

    # TODO check that the range is between 0 and 5V not something else.
    max_val = 2**res_bit - 1
    dmless_value = _check_within_range(dmless_value, 0, max_val)
    # Note that values vary between 0 and 2^(res_bits) - 1. Map accordingly.
    voltage_range = max_voltage - min_voltage
    voltage = dmless_value/max_val * voltage_range + min_voltage
    return voltage


def _check_within_range(val, minim, maxim):
    """Check if a val is in between min and max; saturate at the appropriate 
    value if outside the range."""
    if minim > val:
        # too small
        val = minim
    elif maxim < val:
        # too large
        val = maxim
    return val


def to_dmless_val(voltage, res_bit=12, min_voltage=0.55, max_voltage=2.75):
    """Convert a voltage from 0 to 5V to a dimensionless value with the 
    specified bit resolution."""
    voltage = _check_within_range(voltage, min_voltage, max_voltage)
    max_val = 2 ** res_bit - 1
    voltage_range = max_voltage - min_voltage
    dmless_val = (voltage - min_voltage)/voltage_range * max_val
    return int(dmless_val)


if __name__ == '__main__':
    main()

