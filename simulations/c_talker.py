"""Imports useful functions from Arduino code to enable values to be read from 
and written to the .so file."""

from ctypes import cdll
from ctypes import c_int

# Load the source object and import the relevant functions to enable reading
# and writing of global variables.
arduino_code = cdll.LoadLibrary("measure_feedback.so")
get_torque = arduino_code.main
get_torque.restype = c_int  # Specify the return type of the function.
set_k_b = arduino_code.set_k_b
get_k = arduino_code.get_k
get_k.restype = c_int
get_b = arduino_code.get_b
get_b.restype = c_int
get_amp = arduino_code.get_amp
get_amp.restype = c_int
set_amp = arduino_code.set_amp
set_amp.restype = c_int


def main():
    #results = []
    #for t in range(120):
    #    results.append([t, get_torque(t)])
    #results = np.array(results)
    set_k_b(3, 4)
    print(get_k())
    print(get_b())
    set_amp(3)
    print(get_amp())
    #np.savetxt('test.txt', results)


def to_voltage(dmless_value, res_bit=12, min_voltage=0.55, max_voltage=2.75):
    """Convert a digitised value to a voltage between 0 and 5V.
    :param dmless_value: The value to convert to a voltage.
    :param res_bit: The resolution of the apparatus.
    :return: The voltage equivalent in V."""

    # TODO check that the range is between 0 and 5V not something else.
    # Note that values vary between 0 and 2^(res_bits) - 1. Map accordingly.
    max_val = 2**res_bit - 1
    voltage_range = max_voltage - min_voltage
    voltage = dmless_value/max_val * voltage_range + min_voltage
    return voltage


def to_dmless_val(voltage, res_bit=12, min_voltage=0.55, max_voltage=2.75):
    """Convert a voltage from 0 to 5V to a dimensionless value with the 
    specified bit resolution."""
    max_val = 2 ** res_bit - 1
    voltage_range = max_voltage - min_voltage
    dmless_val = (voltage - min_voltage)/voltage_range * max_val
    return dmless_val


if __name__ == '__main__':
    main()

