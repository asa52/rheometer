"""Helper functions for the rest of the code to work, which don't have much 
to do with the Physics."""

import numpy as np
import yaml


def make_same_dim(*variables, ref_dim_array=np.ones(1)):
    """Return variables with the same dimension as some reference array.
    :param variables: Scalar or 1D-array variables to change the dimensions of.
    :param ref_dim_array: Array with the required dimensions."""
    n_vars = len(variables)
    variables = check_types_lengths(*variables)
    try:
        multiplier = np.ones(ref_dim_array.shape[1]).squeeze()
    except IndexError:
        multiplier = np.ones(1)
    split_values = np.array(np.split(np.outer(variables, multiplier), n_vars,
                            axis=0)).squeeze()
    return split_values


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


def baker(fun, args=None, kwargs=None, position_to_pass_through=(0, 0)):
    """Returns an object given by the function 'fun' with its arguments,
    known as a curried function or closure. These objects can be passed into
    other functions to be evaluated.

    :param fun: The function object without any arguments specified.
    :param args: A list of the positional arguments. Put any placeholder in
    the index that will not be baked into the function.
    :param kwargs: A list of keyword arguments.
    :param position_to_pass_through: A tuple specifying the index of
    positional arguments for the function 'fun' that will be skipped in
    baking. For example, (1,3) will skip positional arguments 1 through to
    3, so that the baked arguments in function 'fun' will be:
        fun(baked, unbaked, unbaked, unbaked, baked...).
    If a single position is to be skipped, enter an integer for this
    argument. For example, entering 1 will result in:
        fun(baked, unbaked, baked...).
    NOTE: Ensure the result can fill in the exact number of missing
    positional arguments!
    :return: The object containing the function with its arguments."""

    # Defaults.
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if type(position_to_pass_through) is int:
        position_to_pass_through = (position_to_pass_through,
                                    position_to_pass_through)
    elif type(position_to_pass_through) is not tuple:
        raise TypeError('The variable \'position_to_pass_through\' must be a '
                        'tuple or int.')

    def wrapped(*result):
        """Parameter position_to_pass_through specifies the index of the
        parameter 'result' in sequence of positional arguments for 'fun'."""
        return fun(*(args[:position_to_pass_through[0]] + list(result) + args[(
            position_to_pass_through[1]+1):]), **kwargs)

    return wrapped


def yaml_read(yaml_file):
    """Parses a YAML file and returns the resulting dictionary.
    :param yaml_file: the YAML file path, ending in .yaml."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.load(f)

    for key in config_dict:
        config_dict[key] = convert_to_array(config_dict[key])
    return config_dict


def _check_iterable(variable):
    """Checks that a variable is iterable, such as tuple, list or array, 
    but is not a string."""
    return hasattr(variable, '__iter__') and type(variable) is not str


def combine_quantities(quants, errs=None, operation='mean', axis=None):
    """Calculate a quantity and its error given the quantities quants with 
    error errs (1D arrays). Operation can be: 'add' for addition, 'subtract' 
    for subtraction, 'mean' for weighted mean. Can also specify axis for 
    'add' or 'mean'."""
    if errs is None and (operation == 'add' or operation == 'subtract'):
        errs = np.zeros(quants.shape)
    if axis is None:
        axis = -1

    if operation == 'add':
        quantity = np.sum(quants, axis=axis)
        err = np.sqrt(np.sum(errs ** 2, axis=axis))
    elif operation == 'subtract':
        # Final minus initial quantity. There can only be two values in this
        # case.
        assert len(quants) == len(errs) == 2, \
            "Quantities and errors can only by 1D arrays of length 2 for the " \
            "'subtract' operation."
        quantity = np.ediff1d(quants)[0]
        err = np.sqrt(np.sum(errs ** 2))
    elif operation == 'mean':
        if errs is None:
            errs = np.ones(quants.shape)
            err = np.std(quants, ddof=1) / np.sqrt(quants.shape[axis])
        else:
            err = np.sqrt(1 / np.sum(1 / errs ** 2, axis=axis)) / np.sqrt(
                errs.shape[axis])
        quantity = np.average(quants, weights=1 / errs ** 2, axis=axis)
    else:
        raise ValueError('Invalid operation.')
    return np.array([quantity, err]).squeeze()