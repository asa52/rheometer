"""Helper functions for the rest of the code to work, which don't have much 
to do with the Physics."""
import numpy as np


def make_same_dim(*variables, ref_dim_array=np.ones(1)):
    """Return variables with the same dimension as some reference array.
    :param variables: Scalar or 1D-array variables to change the dimensions of.
    :param ref_dim_array: Array with the required dimensions."""
    n_vars = len(variables)
    variables = check_types_lengths(*variables)
    multiplier = np.ones(ref_dim_array.shape).squeeze()
    split_values = np.array(np.split(np.outer(variables, multiplier), n_vars,
                            axis=0)).squeeze()
    return split_values


def _check_iterable(variable):
    """Checks that a variable is iterable, such as tuple, list or array, 
    but is not a string."""
    return hasattr(variable, '__iter__') and type(variable) is not str


def convert_to_array(variable):
    """Checks if variable is of correct type, convert to array."""
    iterable = _check_iterable(variable)
    if iterable:
        return np.array(variable)
    else:
        return np.array([variable])


def check_types_lengths(*variables, check_lengths=True):
    converted_vars = []
    for i in range(len(variables)):
        converted = convert_to_array(variables[i])
        if i > 0 and check_lengths:
            assert len(converted) == len(converted_vars[i - 1]), \
                "Variables are of different lengths"
        converted_vars.append(converted)
    return converted_vars