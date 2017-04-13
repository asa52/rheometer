"""Functions to process and perform measurements on the simulated data."""

import numpy as np
import pyfftw
import scipy.signal as sg
import matplotlib.pyplot as plt

import helpers as h


def calc_fft(x_axis, y_axis):
    """Calculate the FFT of y over domain x."""
    n = y_axis.shape[-1]  # length of the signal
    # Get frequencies, normalise by sampling rate and shift to be centred at 0.
    freqs = np.fft.fftshift(np.fft.fftfreq(n)) / (x_axis[1] - x_axis[0])
    # calculate FFT using FFTW module. Then, shift and normalise.
    fft = pyfftw.builders.fft(y_axis, overwrite_input=False,
                              planner_effort='FFTW_ESTIMATE',
                              threads=2, auto_align_input=False,
                              auto_contiguous=False, avoid_copy=True)
    full_fft_y = np.fft.fftshift(2 * fft() / n)
    return freqs, full_fft_y


def identify_ss(x_axis, y_axis, n_per_segment=None, min_lim=0.95, tol=0.005,
                max_increase=0.0001):
    """Calculate the x range over which the FFT of y is in the steady state, 
    which means the normalised correlation between consecutive data segments is 
    greater than min_lim and within tol of the previous. x should be 
    uniformly spaced."""

    if n_per_segment is None:
        n_per_segment = int(len(x_axis) / 400)

    # Get consecutive correlations.
    xs, correlations = _norm_correlations(x_axis, y_axis, n_per_segment)

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


def get_peak_pos(max_peaks, x_axis, y_axis):
    """Calculate the positions and bandwidths of the resonant peaks.
    :param max_peaks: 2D array of [index, peak_value] entries for each peak, 
    as from peak detect function.
    :param x_axis: Array of positions, for example time.
    :param y_axis: Array of heights of all steady state data.
    :return: Half-bandwidths for each of the resonant signals, calculated from 
    the width of the range over which the peak exceeds the half-max height."""
    positions = []
    half_widths = []
    data_length = len(x_axis)
    # Better way to find the bandwidth is using the half-max rule; find the
    # width of the peak at the half-power limit, such that all positions in
    # this range have a y value greater than half-maximum for the peak.
    for i in range(len(max_peaks[:, 1])):
        half_max = max_peaks[i, 1] / 2.
        indices = np.where(y_axis >= half_max)[0]
        # Split the data according to consecutive sets and find the set that
        # contains the index of the peak.
        consecs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        correct_indices = np.array([])
        for consec_set in consecs:
            if max_peaks[i, 0] in consec_set:
                correct_indices = consec_set
                break
        minim, maxim = np.min(correct_indices), np.max(correct_indices)

        positions.append(x_axis[int(max_peaks[i, 0])])
        if int(minim) == int(maxim):
            # If peak is infinitely sharp at this resolution, take the
            # position resolution to get error.
            err = x_axis[1] - x_axis[0]
        elif minim == 0 or maxim == data_length - 1:
            err = x_axis[int(maxim)] - x_axis[int(minim)]
        else:
            err = (x_axis[int(maxim)] - x_axis[int(minim)]) / 2.
        half_widths.append(err)
    return np.array([positions, half_widths])


def peak_detect(y_values, n_maxima):
    """Find the indices of the peaks in y-values and their heights, and return 
    the sharpest n_maxima peaks. Does a calculation based on the np.gradient 
    function, which uses using second order accurate central differences in the 
    interior and either first differences or second order accurate one-sides 
    (forward or backwards) differences at the boundaries."""
    peaks = []
    first_diff = np.gradient(y_values)
    second_diff = np.gradient(first_diff)
    # Maxima will have negative curvature. Obtain the peak position by
    # finding within each consecutive set where the first derivative is
    # closest to zero.
    maxima_ranges = np.where(second_diff < 0)[0]
    consecs = np.split(maxima_ranges, np.where(np.diff(maxima_ranges) != 1)[0]
                       + 1)
    for indices in consecs:
        # Want to find the stationary point of the curve - where first_diff
        # is closest to zero for that peak. Indices is an array of indices.
        diff_set = first_diff[indices]
        pos = (np.abs(diff_set)).argmin()
        index = indices[pos]
        peaks.append([index, y_values[index]])

    peaks = np.array(peaks)
    n_maxima = int(n_maxima)
    if n_maxima is not None:
        ind = np.argpartition(peaks[:, 1], -n_maxima)[-n_maxima:]
        peaks = peaks[ind, :]
    return peaks


def calc_one_amplitude(real_disps, bins=None):
    """Calculate the mean amplitude of time-domain signal y."""
    if bins is None:
        bins = int(len(real_disps) / 10)

    # Get a histogram of the displacement values and find the peaks here,
    # which should occur at the endpoints for a single frequency. This is
    # where the system is slowest (stationary), hence the amplitude.
    results = np.histogram(real_disps, bins=bins)
    num = results[0]
    displacements = results[1]
    bin_centres = displacements[:-1] + np.ediff1d(displacements)

    # Find the 2 biggest maxima and find their difference to get the
    # peak-to-peak displacement, which is then halved with an error based on the
    # width of the bins.
    maxima = peak_detect(num, 2)
    max_disps = get_peak_pos(maxima, bin_centres, num)
    max_disps[0, :] -= np.mean(displacements)
    return np.absolute(h.combine_quantities(np.absolute(max_disps[0, :]),
                                            max_disps[1, :],
                                            operation='mean'))


def calc_freqs(full_fft_y, freqs, n_peaks=1):
    """Calculate the top n_peaks frequencies in a FFT'd signal full_fft_y."""
    # Get the peak positions and their bandwidths.
    max_peaks = peak_detect(full_fft_y, n_peaks * 2)

    # We always expect the spectrum to be symmetric as the signal is real. So
    # there will be an even number of peaks.
    peak_pos = get_peak_pos(max_peaks, freqs, full_fft_y)
    len_array = len(peak_pos[0, :])
    assert len_array % 2 == 0 or len_array == 1, \
        "Odd number of peaks that is not 1 - input signal is not real or has " \
        "multiple frequencies."

    # Rearrange the array into a form that can be used to average over the
    # positive and negative frequencies in the spectrum.
    num_freqs = int(len_array / 2) if len_array != 1 else 1
    if len_array > 1:
        average_over = np.absolute(np.array([
            peak_pos[:, :num_freqs], np.flip(peak_pos[:, num_freqs:], 1)]))
        return h.combine_quantities(
            np.absolute(average_over[:, 0, :]), average_over[:, 1, :],
            operation='mean', axis=0)
    else:
        return peak_pos.squeeze()


def calc_phase(real_resp, real_force):
    """Calculate the phase difference in radians between the displacement and 
    the torque using a Hilbert transform. Signal MUST be monochromatic, 
    so filter first if necessary.
    :param real_resp: The displacement, as an array.
    :param real_force: The analytic torque, as an array."""

    # Make sure y and torque are complex and not absolute-valued.
    hil_y = sg.hilbert(real_resp)
    hil_torque = sg.hilbert(real_force)
    phases = np.angle(hil_y / hil_torque)  # also calculate average and error.
    return h.combine_quantities(phases, operation='mean')


def _calc_stft(x_axis, y_axis, n_per_segment):
    """Calculate the windowed FFT of y across its range, with 1/2 overlap 
    between one window and the next one, using a top hat window, 
    with n_per_segment points per window."""
    x_axis, y_axis = h.check_types_lengths(x_axis, y_axis)
    fs = 1 / (x_axis[1] - x_axis[0])
    return sg.stft(y_axis, fs, window='boxcar', nperseg=n_per_segment)


def _norm_correlations(x_axis, y_axis, n_per_segment):
    """Get normalised correlations of consecutive y segments with 
    n_per_segment points per segment."""
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


def remove_one_frequency(times, theta, w_remove):
    """Remove one frequency from the real space signal.
    :param times: Time array
    :param theta: Theta array
    :param w_remove: Remove this angular frequency.
    :return: The theta array with the frequency in question filtered out."""

    n = theta.shape[-1]
    frq, fft_theta = calc_fft(times, theta)
    f_remove = w_remove / (2*np.pi)
    frq_res = frq[1] - frq[0]

    # Filter
    neg = np.logical_and(-10 * frq_res - f_remove <= frq,
                         frq <= 10 * frq_res - f_remove)
    pos = np.logical_and(-10 * frq_res + f_remove <= frq,
                         frq <= 10 * frq_res + f_remove)
    remove = np.where(np.logical_or(neg, pos))
    fft_theta[remove] = 0

    # Calculate IFFT using FFTW module. Shift and normalise.
    ifft = pyfftw.builders.ifft(
        np.fft.ifftshift(fft_theta) * n / 2., overwrite_input=False,
        planner_effort='FFTW_ESTIMATE', threads=2, auto_align_input=False,
        auto_contiguous=False, avoid_copy=True)
    return ifft()


if __name__ == '__main__':
    t = np.linspace(0, 1000, 10000)
    amp = np.sin(t) + np.sin(2 * t)
    plt.plot(t, amp)
    plt.show()
    filtered_amps = remove_one_frequency(t, amp, 1)
    plt.plot(t, np.real(filtered_amps), t, np.imag(filtered_amps))
    plt.show()
