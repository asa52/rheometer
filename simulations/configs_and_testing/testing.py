"""Code separate from the main code, for testing purposes (such as timing)."""

import time
import numpy as np


class CodeTimer:
    """Generic class to time code execution."""

    def __init__(self):
        self.start = None  # Timer not yet started.
        self.created = time.time()
        # Keep track of how many times this timer has measured the elapsed time.
        self.n_called = 0
        self.checkpoints = []
        print("Timer created but not started.")
        return

    def start(self):
        self.start = time.time()
        print("Timer started.")

    def checkpoint(self, name, verbose=True):
        """Create a checkpoint for which elapsed time will be printed. Give 
        the checkpoint a name string to make it recognisable."""
        self.n_called += 1
        time_here = self.get_elapsed()
        self.checkpoints.append([self.n_called, name, time_here])
        if verbose:
            print("Reached {} at {}s after start.".format(name, time_here))

    def see_checkpoints(self):
        """Print all timing checkpoints so far."""
        print(np.array(self.checkpoints))
        print("Total runtime so far: {}".format(self.get_elapsed()))

    def get_elapsed(self):
        return time.time() - self.start
