"""Functions not being used right now because they MAY NOT WORK and/or are 
not currently needed."""

import numpy as np
import re

from simulations.experiment import helpers as h


def padder(arr, stop_at=0.9):
    """Returns a 2D array of values from arr, shifted and padded to look at 
    the whole range, then sub-select the later range, and so on. This helps 
    identify the steady state in a sequence of displacement values.
    :param arr: 1D array of y-values to FFT later.
    :param stop_at: The fraction of the length of arr that the final set to 
    be FFT'd is.
    :return: 2D array of [arr, arr[0: n / 50], arr[n / 50 + 1: 2 * n / 50], 
                          ..., arr[m * n / 50 + 1: stop_at * n]], 
             the ends of the rows padded with zeros, where n = len(arr)."""
    n = len(arr)
    fft_range = np.arange(0, int(stop_at * n), int(np.ceil(n / 50.)))
    padded = np.zeros((n, len(fft_range)))
    for index, i in np.ndenumerate(fft_range):
        padded[:, index[0]] = np.pad(arr, (0, i), mode='constant')[i:]
    return np.array(padded).T


def calc_amplitudes(x, y, res_freqs, full_Y):
    """Calculate the amplitudes of the waves in y over time x given the 
    resonant frequencies and their bandwidths of the signal in y."""
    n = len(x)
    k = np.arange(n)
    dt = x[1] - x[0]
    # errors not taken into account yet in terms of bandwidth
    trig_coeffs = []
    for freq in res_freqs[0, :]:
        b = freq * dt - 2 * np.pi * k / n
        b_prime = freq * dt + 2 * np.pi * k / n
        sines = ((1 - np.exp(n * b * 1j)) / (1 - np.exp(b * 1j)) -
                 (1 - np.exp(-n * b_prime * 1j)) / (
                     1 - np.exp(-b_prime * 1j))) / 2j
        cosines = ((1 - np.exp(n * b * 1j)) / (1 - np.exp(b * 1j)) +
                   (1 - np.exp(-n * b_prime * 1j)) / (
                       1 - np.exp(-b_prime * 1j))) / 2
        trig_coeffs.append(sines)
        trig_coeffs.append(cosines)
    a = np.array(trig_coeffs).T
    print(a)

    # avs = np.zeros(4)
    # i = 0
    # while True:
    #    try:
    #        avs = np.add(avs, np.linalg.lstsq(a[5 * i:5 * i + 4, :],
    #                                          full_Y[5 * i:5*i+4])[0])#[0]
    #        i += 1
    #    except ValueError:
    #        break
    # print(avs/i)
    print(np.dot(np.linalg.pinv(a), full_Y))
    return


def find_peaks(freqs, ffts, n_expected=1):
    """Find the peak and bandwidth of a signal given the number of peaks 
    expected."""
    diffs = np.ediff1d(ffts)
    signs = np.sign(diffs)
    all_peaks_pos = np.ediff1d(signs) < 0
    peaks = []
    look_for_rise = True
    for index, pos in enumerate(all_peaks_pos):
        if pos:
            peak = tuple(signs[index:index + 2])
            if peak == (1, 0) and look_for_rise:
                try:
                    peaks.append([])
                    peaks[-1].append(ffts[index: index + 2])
                    look_for_rise = False
                except IndexError:
                    pass
            elif peak == (1, -1):
                peaks.append([])
                peaks[-1].append(ffts[index: index + 2])
                peaks[-1].append(ffts[index + 1: index + 3])
            elif peak == (0, -1) and not look_for_rise:
                try:
                    peaks[-1].append(ffts[index + 1: index + 3])
                    look_for_rise = True
                except IndexError:
                    pass

    peak_diffs = np.array([])
    print("peaks1", peaks)
    for index, i in enumerate(peaks):
        if len(i) != 2:
            peaks.pop(index)
    print("fin", peaks)
    peaks = np.array(peaks)
    for i in range(len(peaks)):
        rise = np.absolute(peaks[i, 0, 1] - peaks[i, 0, 0])
        fall = np.absolute(peaks[i, 1, 1] - peaks[i, 1, 0])
        peak_diffs = np.append(peak_diffs, np.mean([rise, fall]))
    ind = np.argpartition(peak_diffs, -n_expected)[-n_expected:]
    peaks = peaks[ind, :, :]
    res_freqs = []
    b_widths = []
    for i in range(len(peaks)):
        min_freq = freqs[np.where(ffts == peaks[i, 0, 0])]
        mean_freq = freqs[np.where(ffts == peaks[i, 1, 0])]
        res_freqs.append(mean_freq.squeeze())
        max_freq = freqs[np.where(ffts == peaks[i, 1, 1])]
        b_widths.append((max_freq - min_freq).squeeze())
    res_freqs = np.array([res_freqs, b_widths])
    return res_freqs, ffts


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0, n_maxima=None,
               n_minima=None, first_is_peak=False):
    """Detects local maxima and minima in a signal, by searching for values 
    which are surrounded by lower or larger values for maxima and minima 
    respectively. Sourced from: https://gist.github.com/sixtenbe/1178136. 
    Based on MATLAB script at: http://billauer.co.il/peakdet.html.
    :param y_axis: A list containing the signal over which to find peaks.
    :param x_axis: List whose values correspond to the y_axis list and is used 
    in the return to specify the position of the peaks. If omitted an index of 
    the y_axis is used.
    :param lookahead: Distance to look ahead from a peak candidate to determine 
    if it is the actual peak. '(samples / period) / f' where '4 >= f >= 1.25' 
    might be a good value.
    :param delta: Specifies a minimum difference between a peak and the 
    following points, before a peak may be considered a peak. Useful to hinder 
    the function from picking up false peaks towards to end of the signal. To 
    work well, should be set to delta >= RMSnoise * 5. When omitted, delta 
    function causes a 20% decrease in speed. When used correctly, it can 
    double the speed of the function.
    :param n_maxima: The number of maxima to return. The sharpest ones are 
    returned.
    :param n_minima: The number of minima to return.
    :param first_is_peak: True if there is a possibility that the very first 
    value is a peak (e.g. for testing noiseless signals; all 'real' signals 
    won't have this normally if time resolution is fine enough).
    :return Two lists [max_peaks, min_peaks] containing the positive and 
    negative peaks respectively. Each cell of the lists contains a tuple of: 
    (position, peak_value). To get the average peak value do: np.mean(
    max_peaks, 0)[1] on the results. To unpack one of the lists into x, 
    y coordinates do: x, y = zip(*max_peaks)."""

    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # store data length for later use
    length = int(len(y_axis))
    # check input data
    if x_axis is None:
        x_axis = range(length)
    x_axis, y_axis = h.check_types_lengths(x_axis, y_axis)

    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # Maxima and minima candidates are temporarily stored in mx and mn
    # respectively.
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis, y_axis)):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        # Look for maxima.
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                # dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                # if index + lookahead >= length:
                #    # end is within lookahead no more peaks can be found
                #    break
                continue

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                # dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                # if index + lookahead >= length:
                #    # end is within lookahead no more peaks can be found
                #    break

    # Remove the false hit on the first value of the y_axis
    # try:
    #    if not first_is_peak:
    #        if dump[0]:
    #            max_peaks.pop(0)
    #        else:
    #            min_peaks.pop(0)
    #        del dump
    # except IndexError:
    #    pass    # No maxima/minima found.

    max_peaks = np.array(max_peaks)
    min_peaks = np.array(min_peaks)
    if n_maxima is not None:
        ind = np.argpartition(max_peaks[:, 1], -n_maxima)[-n_maxima:]
        max_peaks = max_peaks[ind, :]
    if n_minima is not None:
        ind = np.argpartition(min_peaks[:, 1], -n_minima)[-n_minima:]
        min_peaks = min_peaks[ind, :]
    return [max_peaks, min_peaks]


# regex pattern to find numbers of any kind in a string.
numeric_const_pattern = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"


def nr_torque(t, omega_d, amplitude, phase, b_prime, k_prime):
    """Get the full torque for the oscillator at time t.
    :param t: A single time value in seconds.
    :param omega_d: Angular frequency of analytic part of torque.
    :param amplitude: Amplitude of ""
    :param phase: Phase of "".
    :param b_prime: Coefficient of d(theta)/dt (measured).
    :param k_prime: Coefficient of theta (measured).
    :return: The torque value at time t."""

    # First run code
    run = 0
    talk.set_k_b_primes(k_prime, b_prime)  # todo check value and dimensions of each
    # todo set amplitude, phase and omega_d also
    theta_sim = []
    omega_sim = []
    nr_func = []

    # torque = analytic_torque(t, omega_ds, amplitudes, phases)

    # The DAC works by feeding true voltages from 0 to 4095 for 12-bit
    # resolution. Various frequencies are handled by calling the get_torque
    # function in the Arduino script with the step size of dt = T/120,
    # where T is the desired period of the sine waveform. One facet of this C
    # code is that measurement is only performed 120 times every wave,
    # rather than being separate from the analytic torque's period. So calling
    # this generator at this rate would allow the pendulum motion to be
    # simulated. Note that the internal step size for the ode iterator is
    # different from the set dt.
    # TODO make sure that varying dt does not affect accuracy of the iterator!

    # Connect to Arduino code here! Measure and return the value of theta and
    # omega at this time. Consider changing this function to a generator to
    # retain previous values of measured theta and omega.
    while True:
        # get mu and dmudt and func. convert to actual values.
        run += 1
        func = talk.get_torque  # change
        mu = 0  # change
        dmudt = 0  # change
        theta_sim.append(mu)
        omega_sim.append(dmudt)
        nr_func.append(func)
        yield nr_func  # change to reflect some time delay.