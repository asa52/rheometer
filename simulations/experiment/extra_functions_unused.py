"""Extra functions created for measurement that are not currently being used."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg

import helpers as h


def enter_ss_times(x_axis, y_axis):
    """Enter the times at which the displacements are steady state manually 
    and return the minimum and maximum."""
    plt.plot(x_axis, y_axis)
    plt.show()

    while True:
        ss_times = input('Enter the time range over which the signal is steady '
                         'state.')
        ss_times = ss_times.split(' ')

        try:
            if ss_times[0] == 'none':
                # ss not reached.
                return False
            elif len(ss_times) == 1:
                return float(ss_times[0]), x_axis[-1]
            elif len(ss_times) == 2:
                return float(ss_times[0]), float(ss_times[1])
            else:
                print('Invalid answer.')

        except ValueError:
            # Ignore answers of the wrong format, just ask again until the
            # correct answer is given.
            pass


def norm_correlations(x_axis, y_axis, n_per_segment):
    """Get normalised correlations of consecutive y segments with n_per_segment 
    points per segment using the windowed fourier transform."""
    results = _calc_stft(x_axis, y_axis, n_per_segment)
    freqs = results[0]
    xs = results[1][1:]
    amplitudes = results[2]

    # Calculate normalised amplitudes by subtracting mean. Necessary for
    # normalised correlation calculation.
    mean_amps = np.mean(amplitudes, axis=0)
    means = np.outer(np.ones(len(freqs)), mean_amps)
    norm_amps = amplitudes - means

    prev_col = np.roll(norm_amps, 1, axis=1)
    norm_by = norm_amps.shape[0] * np.std(norm_amps, ddof=1, axis=0) * \
        np.std(prev_col, ddof=1, axis=0)

    # Calculate consecutive normalised correlations.
    correlations = []
    for i in range(norm_amps.shape[1]):
        if i != 0:
            try:
                correlations.append(np.absolute(sg.correlate(
                    prev_col[:, i], norm_amps[:, i], mode='valid') /
                                                norm_by[i]))
            except RuntimeWarning:
                # When trying to divide by zero.
                correlations.append(0)
    return xs, np.array(correlations).squeeze()


def _calc_stft(x_axis, y_axis, n_per_segment):
    """Calculate the windowed FFT of y across its range, with 1/2 overlap 
    between one window and the next one, using a top hat window, 
    with n_per_segment points per window."""
    x_axis, y_axis = h.check_types_lengths(x_axis, y_axis)
    fs = 1 / (x_axis[1] - x_axis[0])
    return sg.stft(y_axis, fs, window='boxcar', nperseg=n_per_segment)
