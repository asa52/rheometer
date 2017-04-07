"""Code separate from the main code, for testing purposes (such as timing)."""

import time
import numpy as np


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
