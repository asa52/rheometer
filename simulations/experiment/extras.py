"""Functions not being used right now because they MAY NOT WORK and/or are 
not currently needed."""

import numpy as np


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
