"""Code for the simulation of the pendulum under NR."""

import time
import numpy as np
import pyfftw
import scipy.signal as sg
from scipy.integrate import ode
import matplotlib.pyplot as plt

import simulations.helpers as h
#import simulations.c_talker as talk


def f(t, y, i, g_0, b_prime, omega_sim, k_prime, theta_sim, b, k):
    """Returns the RHS of the ODE y'(t) = f(t, y).
    :param t: Time in seconds.
    :param y: A vector of [theta, omega], where omega = d(theta)/dt.
    :param i: Moment of inertia of pendulum.
    :param g_0: Analytic driving torque function object, eg. g_0(t) = sin(t).
    :param b_prime: Simulated damping coefficient.
    :param omega_sim: Simulated (measured) angular velocity.
    :param k_prime: Simulated elastic coefficient.
    :param theta_sim:
    :param b: Actual damping coefficient of system.
    :param k: Actual elastic coefficient of system."""
    return [y[1], (g_0(t) + b_prime * omega_sim + k_prime * theta_sim - b *
                   y[1] - k * y[0]) / i]


def jac(t, y, i, b, k):
    """The Jacobian J[i, j] = df[i]/dy[j] of the above f."""
    return [[0, 1], [-k/i, -b/i]]


def ode_integrator(y0, t0, i, b_prime, omega_sim, k_prime, theta_sim, b, k,
                   g_0_mags, w_ds, phases, t_fin, dt, torque_func):
    r = ode(f, jac).set_integrator('dop853')
    baked_g_0 = h.baker(torque_func, args=['', w_ds, g_0_mags, phases])
    r.set_initial_value(y0, t0).set_f_params(
        i, baked_g_0, b_prime, omega_sim, k_prime, theta_sim, b,
        k).set_jac_params(i, b, k)

    results = [[t0, *y0]]
    while r.successful() and r.t < t_fin:
        data_point = [r.t + dt, *np.real(r.integrate(r.t + dt))]
        results.append(data_point)
        # TODO recalculate the parameters and set them again!
    exp_results = np.array(results)
    return exp_results


def analytic_torque(t, omega_ds, amplitudes, phases):
    """Return the value of the analytic driving torque at time t.
    :param t: Time in seconds - a single value.
    :param omega_ds: Angular frequency of the sinusoid.
    :param amplitudes: Amplitude of sinusoid.
    :param phases: Phase of sinusoid in radians."""

    # TODO Add noise.
    amplitudes, omega_ds, phases = h.check_types_lengths(amplitudes, omega_ds,
                                                         phases)
    torque = 0
    for i in range(len(amplitudes)):
        torque += amplitudes[i] * np.sin(omega_ds[i] * t + phases[i])
    return torque


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
    talk.set_k_b(k_prime, b_prime)  # todo check value and dimensions of each
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
        func = talk.get_torque # change
        mu = 0 # change
        dmudt = 0 # change
        theta_sim.append(mu)
        omega_sim.append(dmudt)
        nr_func.append(func)
        yield nr_func   # change to reflect some time delay.


def main():
    ode_integrator([1, 0], 0, 1, 0, 0, 0, 0, 4, 1, [1, 0], [1, 0], [0, 0],
                   1, 0.001, analytic_torque)


def plot_spectrum(x, y):
    """
    Plots a Single-Sided Amplitude Spectrum of y(x)
    """
    next(t1)
    n = y.shape[-1]  # length of the signal
    s = 1/(x[1] - x[0])    # sampling rate
    k = np.arange(n)
    T = n / s
    next(t1)
    frq = k / T  # two sides frequency range
    next(t1)
    frq = frq[range(int(np.round(n/2)))]  # one side frequency range
    next(t1)
    full_Y = np.zeros_like(y)
    fft = pyfftw.builders.fft(y, overwrite_input=False,
                              planner_effort='FFTW_ESTIMATE',
                              threads=2, auto_align_input=False,
                              auto_contiguous=False, avoid_copy=True)
    full_Y = fft()
    full_Y /= n  # interfaces.numpy_fft.fft(y) / n   # fft computing
    # and normalization
    next(t1)
    Y = 2 * full_Y[range(int(np.round(n / 2)))]
    next(t1)
    return frq, np.absolute(Y) ** 2, full_Y


def _calc_stft(x, y):
    x, y = h.check_types_lengths(x, y)
    fs = 1/(x[1] - x[0])
    return sg.stft(y, fs, window='boxcar', nperseg=int(len(x) / 400))


def get_correlation(x, y, min_lim=0.95, tol=0.005):
    results = _calc_stft(x, y)
    freqs = results[0]
    times = results[1][1:]
    amplitudes = results[2]
    mean_amps = np.mean(amplitudes, axis=0)
    means = np.outer(np.ones(len(freqs)), mean_amps)
    norm_amps = amplitudes - means
    prev_col = np.roll(norm_amps, 1, axis=1)
    norm_by = norm_amps.shape[0] * np.std(norm_amps, ddof=1, axis=0) * \
              np.std(prev_col, ddof=1, axis=0)
    correlations = []
    for i in range(norm_amps.shape[1]):
        if i != 0:
            try:
                correlations.append(np.absolute(sg.correlate(
                    prev_col[:, i], norm_amps[:, i], mode='valid') /
                                                norm_by[i]))
            except RuntimeWarning:
                correlations.append(0)

    correlations = np.array(correlations).squeeze()
    # To work out the range of times that correspond to good steady state
    # values, require that the correlation exceeds 'min_lim' and the gradient
    # varies by no more than 'tol' either side. Find the range of times where
    # this is obeyed.
    exceeds_min = correlations >= min_lim
    # within_tol checks that the differences between consecutive values are
    # small, but there can still be a small net increase. Need to ensure that
    # the differences are within some range around 0.
    diffs = np.ediff1d(correlations)
    within_tol = np.absolute(diffs) <= tol
    within_tol = np.insert(within_tol, 0, [False])
    valid_corr = correlations[exceeds_min * within_tol]
    assert np.absolute(np.mean(valid_corr - np.mean(valid_corr))) < 10 ** (-4), \
        "Steady state not reached - correlations are changing consistently."
    ss_times = times[exceeds_min * within_tol]
    return min(ss_times), max(ss_times)


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


def calc_one_amplitude(y):
    plt.hist(y, bins=int(len(y) / 10))
    plt.show()
    results = np.histogram(y, bins=int(len(y) / 10))
    num = results[0]
    amps = np.array([*results[1]])
    medians = amps[:-1] + np.ediff1d(amps)
    max_disps = np.array(peakdetect(num, medians)[0])
    ind = np.argpartition(np.absolute(max_disps[:, 1]), -2)[-2:]
    print(np.absolute(max_disps[ind, 0]))
    # mean_amp = np.average(np.absolute(max_disps[0, :]),
    #                      weights=1/max_disps[1,:]**2)
    # mean_err = (1/np.sum(1/max_disps[1, :]**2))/np.sqrt(len(max_disps[1, :]))
    # return mean_amp, mean_err


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0):
    """ SOURCE: https://gist.github.com/sixtenbe/1178136
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html

    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)

    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value

    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

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
                # else:  #slows shit down this does
                #    mx = ahead
                #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

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
                    # else:  #slows shit down this does
                    #    mn = ahead
                    #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

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

    return [max_peaks, min_peaks]


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
            "Input vectors y_axis and x_axis must have same length")

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def timer():
    start = time.time()
    counter = 0
    while True:
        elapsed = time.time() - start
        counter += 1
        print(counter, elapsed)
        yield elapsed

if __name__ == '__main__':
    t1 = timer()
    t = np.linspace(0, 10000, 100000)
    y = np.sin(5 * t) + 0.05 * np.random.rand(100000)
    next(t1)
    ss_times = get_correlation(t, y)
    next(t1)
    frq, Y_sq, full_Y = plot_spectrum(t[(t >= ss_times[0]) * (t <= ss_times[
        1])], y[(t >= ss_times[0]) * (t <= ss_times[1])])
    next(t1)
    max_peaks = np.array(peakdetect(Y_sq, frq)[0])  # max peak (frq_position,
    # height of
    # peak)
    ind = np.argpartition(max_peaks[:, 1], -1)[-1:]
    # print(max_peaks[ind, :])
    # res_freqs, ffts = find_peaks(frq, Y_sq, n_expected=1)
    amplitudes = calc_one_amplitude(y[(t >= ss_times[0]) * (t <= ss_times[1])])
    #print(amplitudes)
    next(t1)
