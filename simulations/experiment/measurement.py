"""Functions to perform measurements on the simulated data."""

import numpy as np
import pyfftw
import scipy.signal as sg

import simulations.experiment.helpers as h


def calc_fft(x, y):
    """Calculate the FFT of y over domain x."""

    n = y.shape[-1]  # length of the signal
    # Get frequencies, normalise by sampling rate and shift to be centred at 0.
    freqs = np.fft.fftshift(np.fft.fftfreq(n)) / (x[1] - x[0])

    # calculate FFT using FFTW module. Then, shift and normalise.
    fft = pyfftw.builders.fft(y, overwrite_input=False,
                              planner_effort='FFTW_ESTIMATE',
                              threads=2, auto_align_input=False,
                              auto_contiguous=False, avoid_copy=True)
    full_fft_y = np.fft.fftshift(2 * fft() / n)
    return freqs, np.absolute(full_fft_y)


def _calc_stft(x, y, n_per_segment):
    """Calculate the windowed FFT of y across its range, with 1/2 overlap 
    between one window and the next one, using a top hat window, 
    with n_per_segment points per window."""
    x, y = h.check_types_lengths(x, y)
    fs = 1 / (x[1] - x[0])
    return sg.stft(y, fs, window='boxcar', nperseg=n_per_segment)


def _norm_correlations(x, y, n_per_segment):
    """Get normalised correlations of consecutive y segments with 
    n_per_segment points per segment."""
    results = _calc_stft(x, y, n_per_segment)
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


def identify_ss(x, y, n_per_segment=None, min_lim=0.95, tol=0.005,
                max_increase=0.0001):
    """Calculate the x range over which the FFT of y is in the steady state, 
    which means the normalised correlation between consecutive data segments is 
    greater than min_lim and within tol of the previous. x should be 
    uniformly spaced."""

    if n_per_segment is None:
        n_per_segment = int(len(x) / 400)

    # Get consecutive correlations.
    xs, correlations = _norm_correlations(x, y, n_per_segment)

    # To work out the range of times that correspond to good steady state
    # values, require that the correlation exceeds 'min_lim' and the gradient
    # varies by no more than 'tol' either side. Find the range of times where
    # this is obeyed.
    exceeds_min = correlations >= min_lim

    # within_tol checks that the differences between consecutive values are
    # small, but there can still be a small net increase. Need to ensure that
    # the differences are within some range around 0. 10**(-4) is the default.
    diffs = np.ediff1d(correlations)
    within_tol = np.absolute(diffs) <= tol
    within_tol = np.insert(within_tol, 0, [False])
    valid_corr = correlations[exceeds_min * within_tol]
    assert np.absolute(np.mean(valid_corr - np.mean(valid_corr))) < \
           max_increase, "Steady state not reached - correlations are changing."
    ss_points = xs[exceeds_min * within_tol]
    return min(ss_points), max(ss_points)


def get_peak_pos(max_peaks, x):
    """Calculate the positions and bandwidths of the resonant peaks.
    :param max_peaks: 2D array of [index, peak_value] entries for each peak, 
    as from peak detect function.
    :param x: the range of positions, for example time.
    :return: Bandwidths for each of the resonant signals, calculated from the 
    width of the range over which the peak increases then drops."""
    positions = []
    bandwidths = []
    for index in max_peaks[:, 0]:
        positions.append(x[int(index)])
        if index == 0:
            bandwidths.append(x[int(index) + 1] - x[int(index)])
        elif index == np.max(max_peaks[:, 0]):
            bandwidths.append(x[int(index)] - x[int(index) - 1])
        else:
            bandwidths.append(x[int(index) + 1] - x[int(index) - 1])
    return np.array([positions, bandwidths])


def calc_one_amplitude(y, bins=None):
    """Calculate the mean amplitude of time-domain signal y."""
    if bins is None:
        bins = int(len(y) / 10)

    # Get a histogram of the displacement values and find the peaks here,
    # which should occur at the endpoints for a single frequency. This is
    # where the system is slowest (stationary), hence the amplitude.
    results = np.histogram(y, bins=bins)
    num = results[0]
    displacements = results[1]
    bin_centres = displacements[:-1] + np.ediff1d(displacements)

    # Find the 2 biggest maxima and find their difference to get the
    # peak-to-peak displacement, which is then halved with an error based on the
    # width of the bins.
    maxima = peakdetect(num, n_maxima=2)[0]
    max_disps = get_peak_pos(maxima, bin_centres)
    return np.absolute(combine_quantities(max_disps[0, :] / 2.,
                                          max_disps[1, :] / 2.,
                                          operation='subtract'))


def calc_freqs(full_fft_y, freqs, n_peaks=1):
    """Calculate the top n_peaks frequencies in a FFT'd signal full_fft_y."""
    # Get the peak positions and their bandwidths.
    max_peaks = peakdetect(full_fft_y, n_maxima=n_peaks * 2)[0]

    # We always expect the spectrum to be symmetric as the signal is real. So
    #  there will be an even number of peaks.
    peak_pos = get_peak_pos(max_peaks, freqs)
    len_array = len(peak_pos[0, :])
    assert len_array % 2 == 0, "Odd number of peaks - input signal is not real!"

    # Rearrange the array into a form that can be used to average over the
    # positive and negative frequencies in the spectrum.
    num_freqs = int(len_array / 2)
    average_over = np.absolute(np.array([peak_pos[:, :num_freqs],
                                         np.flip(peak_pos[:, num_freqs:], 1)]))
    return combine_quantities(np.absolute(average_over[:, 0, :]),
                              average_over[:, 1, :], operation='mean', axis=0)


def combine_quantities(quants, errs, operation='mean', axis=None):
    """Calculate a quantity and its error given the quantities quants with 
    error errs (1D arrays). Operation can be: 'add' for addition, 'subtract' 
    for subtraction, 'mean' for weighted mean. Can also specify axis for 
    'add' or 'mean'."""
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
        quantity = np.average(np.absolute(quants), weights=1 / errs ** 2,
                              axis=axis)
        err = np.sqrt(1 / np.sum(1 / errs ** 2, axis=axis)) / np.sqrt(
            errs.shape[axis])
    else:
        raise ValueError('Invalid operation.')
    return np.array([quantity, err]).squeeze()


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0, n_maxima=None,
               n_minima=None):
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
    :return Two lists [max_peaks, min_peaks] containing the positive and 
    negative peaks respectively. Each cell of the lists contains a tuple of: 
    (position, peak_value). To get the average peak value do: np.mean(
    max_peaks, 0)[1] on the results. To unpack one of the lists into x, 
    y coordinates do: x, y = zip(*max_peaks)."""

    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    length = int(len(y_axis))
    if x_axis is None:
        x_axis = range(length)
    x_axis, y_axis = h.check_types_lengths(x_axis, y_axis)
    # store data length for later use

    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    max_peaks = np.array(max_peaks)
    min_peaks = np.array(min_peaks)
    if n_maxima is not None:
        ind = np.argpartition(max_peaks[:, 1], -n_maxima)[-n_maxima:]
        max_peaks = max_peaks[ind, :]
    if n_minima is not None:
        ind = np.argpartition(min_peaks[:, 1], -n_minima)[-n_minima:]
        min_peaks = min_peaks[ind, :]
    return [max_peaks, min_peaks]


if __name__ == '__main__':
    t = np.linspace(0, 10000, 100000)
    y = np.sin(10 * t) + 0.1 * np.random.rand(100000)
    # If another frequency present, perhaps filter the signal first?
    ss_times = identify_ss(t, y)
    frq, full_Y = calc_fft(t[(t >= ss_times[0]) * (t <= ss_times[1])],
                           y[(t >= ss_times[0]) * (t <= ss_times[1])])
    print(calc_freqs(full_Y, frq))
    print(calc_one_amplitude(y[(t >= ss_times[0]) * (t <= ss_times[1])]))
