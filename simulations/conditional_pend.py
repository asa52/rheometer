"""Code for the simulation of the pendulum under NR."""

import numpy as np
from scipy.integrate import ode
import pyfftw
import scipy.signal as sg
import matplotlib.pyplot as plt
import time

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


def bandwidth(freqs, weights):
    """Calculate the bandwidth of a signal as the standard deviation in the 
    frequencies, weighted by the weights."""
    sum_sq_weights = np.sum(weights**2, axis=1)
    widths = []
    print(weights[3, :])
    for i in range(weights.shape[0]):
        widths.append([i, np.std(freqs * weights[i, :]**2 / sum_sq_weights[i],
                                 ddof=1)])
    widths = np.array(widths)
    plt.plot(widths[:, 0], widths[:, 1])
    plt.show()
    #return width


def padder(arr, stop_at=0.9):
    """Returns a 2D array of values from arr, shifted and padded to look at 
    the whole range, then subselect the later range, and so on. This helps 
    identify the steady state in a sequence of displacement values.
    :param arr: 1D array of y-values to FFT later.
    :param stop_at: The fraction of the length of arr that the final set to 
    be FFT'd is.
    :return: 2D array of [arr, arr[0: n / 50], arr[n / 50 + 1: 2 * n / 50], 
                          ..., arr[m * n / 50 + 1: stop_at * n]], 
             the ends of the rows padded with zeros, where n = len(arr)."""
    n = len(arr)
    fft_range = np.arange(0, int(stop_at * n), int(np.ceil(n/50.)))
    padded = np.zeros((n, len(fft_range)))
    for index, i in np.ndenumerate(fft_range):
        padded[:, index[0]] = np.pad(arr, (0, i), mode='constant')[i:]
    return np.array(padded).T


def calc_stft(x, y):
    x, y = h.check_types_lengths(x, y)
    fs = 1/(x[1] - x[0])
    return sg.stft(y, fs, window='boxcar', nperseg=int(len(x) / 400))


def get_correlation(x, y, min_lim=0.95, tol=0.005):
    results = calc_stft(x, y)
    freqs = results[0]
    times = results[1][1:]
    amplitudes = results[2]
    # for amp in range(amplitudes.shape[1]):
    #    plt.plot(freqs, np.absolute(amplitudes[:, amp]), label='{}'.format(
    # amp))
    # plt.legend()
    # plt.show()
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
    # values, require that the correlation exceeds 95% and the gradient
    # varies by no more than 0.5% either side. Find the range of times where
    # this is obeyed.
    # plt.plot(correlations)
    # plt.show()
    exceeds_min = correlations >= min_lim
    print("exceeds", exceeds_min)
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


def find_peak(freqs, ffts, n_expected=1):
    """Find the peak and bandwidth of a signal given the number of peaks 
    expected."""
    diffs = np.ediff1d(ffts)
    signs = np.sign(diffs)
    all_peaks_pos = np.ediff1d(signs) < 0
    peaks = []
    for index, pos in enumerate(all_peaks_pos):
        if pos:
            peak = tuple(signs[index:index + 2])
            if peak == (1, 0):
                try:
                    peaks.append([])
                    peaks[-1].append(ffts[index: index + 2])
                except IndexError:
                    pass
            elif peak == (1, -1):
                peaks.append([])
                peaks[-1].append(ffts[index: index + 2])
                peaks[-1].append(ffts[index + 1: index + 3])
            elif peak == (0, -1):
                try:
                    peaks[-1].append(ffts[index + 1: index + 3])
                except IndexError:
                    pass

    peak_diffs = np.array([])
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


def calc_amplitudes(x, y, res_freqs, full_Y):
    """Calculate the amplitudes of the waves in y over time x given the 
    resonant frequencies and their bandwidths of the signal in y."""
    n_peaks = res_freqs.shape[1]
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
    t = np.linspace(0, 100000, 1000001)
    y = np.sin(5 * t) + np.cos(3 * t)
    next(t1)
    print("hello")
    ss_times = get_correlation(t, y, min_lim=0.75)
    next(t1)
    frq, Y_sq, full_Y = plot_spectrum(t[(t >= ss_times[0]) * (t <= ss_times[
        1])], y[(t >= ss_times[0]) * (t <= ss_times[1])])
    next(t1)
    res_freqs, ffts = find_peak(frq, Y_sq, n_expected=2)
    amplitudes = calc_amplitudes(t[(t >= ss_times[0]) * (t <= ss_times[1])],
                                 y[(t >= ss_times[0]) * (t <= ss_times[1])],
                                 res_freqs, full_Y)
    print(amplitudes)
    next(t1)
