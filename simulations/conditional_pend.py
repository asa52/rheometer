"""Code for the simulation of the pendulum under NR."""

import numpy as np
from scipy.integrate import ode
from scipy.signal import stft
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
    n = y.shape[1]  # length of the signal
    s = 1/(x[1] - x[0])    # sampling rate
    k = np.arange(n)
    T = n / s
    frq = k / T  # two sides frequency range
    frq = frq[range(int(np.round(n/2)))]  # one side frequency range
    Y = np.fft.fft(y, axis=1) / n  # fft computing and normalization
    Y = 2 * Y[:, range(int(np.round(n/2)))]

    for i in range(y.shape[0]):
        plt.plot(frq, abs(Y[i, :])**2, label=str(i))
    plt.legend()
    plt.show()
    return frq, abs(Y)
    #plt.plot(frq, abs(Y), 'r')  # plotting the spectrum
    #plt.xlabel('Freq (Hz)')
    #plt.ylabel('|Y(freq)|')
    #plt.show()


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
    fs = 1/(x[1] - x[0])
    return stft(y, fs, window='boxcar')


def get_correlation(x, y):
    results = calc_stft(x, y)
    freqs = results[0]
    amplitudes = results[2]
    mean_amps = np.mean(amplitudes, axis=0)
    means = np.outer(np.ones(len(freqs)), mean_amps)
    norm_amps = amplitudes - means
    correlations = []
    for i in range(amplitudes.shape[1]):
        #plt.plot(freqs, np.absolute(amplitudes[:, i]))
        #plt.plot(freqs, np.angle(amplitudes[:, i]))
        if i != 0:
            norm_by = len(norm_amps[:, i]) * np.std(norm_amps[:, i], ddof=1) \
                      * np.std(norm_amps[:, i - 1], ddof=1)
            correlations.append(np.correlate(
                norm_amps[:, i - 1], norm_amps[:, i]) / norm_by)
    plt.plot(np.absolute(correlations))
    plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 10000, 100000)
    y = np.sin(2*t) + 10*np.exp(-t) + 2*np.cos(t) + t**10*np.exp(-2*t) + \
        0.1*np.sin(100*np.pi*t)
    get_correlation(t, y)

