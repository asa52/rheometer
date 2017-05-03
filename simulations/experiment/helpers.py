"""Helper functions for the rest of the code to work, which don't have much 
to do with the Physics."""

import time
from genericpath import isfile
from os import listdir
from os.path import join

import numpy as np
import yaml


class CodeTimer:
    """Generic class to time code execution."""

    def __init__(self, name=None):
        if name is None:
            self.name = ''
        self.start = None  # Timer not yet started.
        self.created = time.time()
        # Keep track of how many times this timer has measured the elapsed time.
        self.n_called = 0
        self.checkpoints = []
        return

    def start_timer(self):
        self.start = time.time()
        print("Timer {} started.".format(self.name))

    def checkpoint(self, checkp_name=None, verbose=False):
        """Create a checkpoint for which elapsed time will be printed. Give 
        the checkpoint a name string to make it recognisable."""
        self.n_called += 1
        if checkp_name is None:
            checkp_name = self.n_called
        time_here = self.get_elapsed()
        self.checkpoints.append([time_here, checkp_name, self.n_called])
        if verbose:
            print("Reached {} at {}s after start.".format(checkp_name,
                                                          time_here))
        return [time_here, checkp_name, self.n_called]

    def see_checkpoints(self):
        """Print all timing checkpoints so far."""
        self.checkpoint('End of experiment')
        checkpoints = np.array(self.checkpoints)
        print("Total runtime: {}".format(self.get_elapsed()))
        return checkpoints

    def get_elapsed(self):
        return time.time() - self.start


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


def baker(fun, args=None, kwargs=None, pos_to_pass_through=(0, 0)):
    """Returns an object given by the function 'fun' with its arguments,
    known as a curried function or closure. These objects can be passed into
    other functions to be evaluated.

    :param fun: The function object without any arguments specified.
    :param args: A list of the positional arguments. Put any placeholder in
    the index that will not be baked into the function.
    :param kwargs: A dictionary of keyword arguments.
    :param pos_to_pass_through: A tuple specifying the index of
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

    if type(pos_to_pass_through) is int:
        pos_to_pass_through = (pos_to_pass_through,
                               pos_to_pass_through)
    elif type(pos_to_pass_through) is not tuple:
        raise TypeError('The variable \'position_to_pass_through\' must be a '
                        'tuple or int.')

    def wrapped(*result):
        """Parameter position_to_pass_through specifies the index of the
        parameter 'result' in sequence of positional arguments for 'fun'."""
        return fun(*(args[:pos_to_pass_through[0]] + list(result) + args[(
            pos_to_pass_through[1] + 1):]), **kwargs)

    return wrapped


def all_combs(func, *args, deconstruct=False):
    """Iterate over a function multiple times using different input values.
    :param func: A baked function object with the correct number of remaining 
    arguments to be filled by args.
    :param args: A list of arrays for argument 1, 2, etc. for the function. E.g.
    If args = [[1,2], [3,4]], the function will be called thus:
    func(1,3), func(1,4), func(2,3), func(2,4). All these values will be 
    returned as a list of results.
    :param deconstruct: Whether or not to deconstruct the output from the 
    function."""
    results = []
    for arg in np.array(np.meshgrid(*args)).T.reshape(-1, len(args)):
        if deconstruct:
            results.append([*arg, *func(*arg)])
            pass
        else:
            results.append([*arg, func(*arg)])
    return results


def yaml_read(yaml_file):
    """Parses a YAML file and returns the resulting dictionary.
    :param yaml_file: the YAML file path, ending in .yaml."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.load(f)

    for key in config_dict:
        config_dict[key] = convert_to_array(config_dict[key])
    return config_dict


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


def stack_cols(*arrs):
    """Stack column arrays side by side."""
    main = np.array(arrs[0])
    add_on = list(arrs[1:])
    for i in range(len(add_on)):
        add_on[i] = convert_to_array(add_on[i])
        if len(add_on[i].shape) == 1 and len(arrs[0].shape) != 1:
            # 1D array, need to introduce extra dimension.
            add_on[i] = np.expand_dims(add_on[i], 1)

    return np.hstack((main, *add_on))


def find_consec_indices(arr):
    """Given an array, find the indices of all the elements that are equal to 
    the last element and also have consecutive indices. Note arr must be 1D."""
    all_matches = np.where(arr[:] == arr[-1])[0]
    index_lists = np.split(all_matches, np.where(np.diff(all_matches) != 1)[0]
                           + 1)
    for i in range(len(index_lists)):
        if len(arr) - 1 in index_lists[i]:
            # Find the set that contains the index of the last element.
            return index_lists[i]
        elif i == len(index_lists) - 1:
            # Not found anywhere
            raise Exception('Panic.')


def find_files(path, check_type='.csv'):
    """Find all the files of the specified type, non-recursively, in the 
    directory specified. Returns a list of the file names."""
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if check_type is not None:
        onlyfiles = [f for f in onlyfiles if f[-len(check_type):] == check_type]
    return onlyfiles


def time_for_name():
    """Get the current time for use in a filename."""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


def _check_iterable(variable):
    """Checks that a variable is iterable, such as tuple, list or array, 
    but is not a string."""
    try:
        len(variable)
        return hasattr(variable, '__iter__') and type(variable) is not str
    except TypeError:
        return False
