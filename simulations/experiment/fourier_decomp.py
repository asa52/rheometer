"""Fourier decomposition of the digitised analytic torque, to be evaluated 
and then solved for theoretically using theory functions."""

import numpy as np
from scipy.integrate import simps

import helpers as h


def g_s(t, angular_freq, delta_t, g_0):
    """Return the value of the digitised analytic torque at time t.
    G_s(t) = g_0 * {0                       for 0  <= t < delta_t, ...
                    sin(angular_freq * t1)  for t1 <= t < t1 + delta_t, ...
                    sin(angular_freq * (T - delta_t)) for T-delta_t <= t < T}.
    :param t: Time array in seconds.
    :param angular_freq: Angular frequency of the driving torque.
    :param delta_t: The measurement time interval. ADVISE to be integer 
    fraction of T.
    :param g_0: Magnitude of the torque."""
    # Format checks
    t = h.convert_to_array(t)
    g_0, delta_t, angular_freq = h.check_types_lengths(g_0, delta_t,
                                                       angular_freq)
    period = 2 * np.pi / angular_freq
    assert len(g_0) == 1

    # Find the position in the cycle by converting everything to 1st period.
    # Then, convert to the value at the beginning of appropriate ranges.
    t = np.fmod(t, period)
    t_floored = np.floor(t / delta_t) * delta_t
    return g_0 * np.sin(angular_freq * t_floored)


def a_formula(t, angular_freq, delta_t, g_0, index):
    """Formula to calculate 'a' coefficient."""
    return np.array([g_s(t, angular_freq, delta_t, g_0)]).T * \
        np.cos(angular_freq * np.outer(t, index))


def b_formula(t, angular_freq, delta_t, g_0, index):
    """Formula to calculate 'b' coefficient."""
    return np.array([g_s(t, angular_freq, delta_t, g_0)]).T * \
        np.sin(angular_freq * np.outer(t, index))


def calc_fourier_coeffs(indices, angular_freq, delta_t, g_0):
    """Calculate the Fourier series coefficients using Simpson's rule.
        a_m = \int_{0}^{T} 2/T * G_s(t) * cos(2*pi*m*t/T) dt, and 
        b_m = \int_{0}^{T} 2/T * G_s(t) * sin(2*pi*m*t/T) dt.
    :param indices: Indices m of the Fourier coefficients to calculate. 
    generally set to a linspace from 1 to a certain number (ideally 
    infinite).
    :param angular_freq: Angular frequency of the sine function.
    :param delta_t: Torque recalculation time.
    :param g_0: Magnitude of the sine torque function."""
    assert indices.dtype == np.int32
    period = 2*np.pi/angular_freq
    integration_t_range = np.arange(0, period, delta_t / 100)
    a_range = a_formula(integration_t_range, angular_freq, delta_t, g_0,
                        indices)
    b_range = b_formula(integration_t_range, angular_freq, delta_t, g_0,
                        indices)
    a_m = 2 / period * simps(a_range, integration_t_range, axis=0)
    b_m = 2 / period * simps(b_range, integration_t_range, axis=0)
    return np.array([indices, a_m, b_m]).T


def convert_to_mag_phis(indices, a_m, b_m):
    """Convert the a_ms and b_ms to magnitudes and phis of R*sin(angular_freq * 
    t + phi)."""
    mags = np.sqrt(a_m**2 + b_m**2)
    phis = np.arctan(a_m/b_m)
    return np.array([indices, mags, phis]).T


def get_torque(t, indices, mags, phis, ang_freq):
    """Calculate the torque in Nm given the Fourier components of the torque."""
    indices, mags, phis = h.check_types_lengths(indices, mags, phis)
    num_times = t.size
    torque_parts = np.array([mags]).T * np.sin(np.outer(
        ang_freq * indices, t) + np.outer(phis, np.ones(num_times)))
    torque = np.sum(torque_parts, axis=0)
    return torque


def get_fourier_series(n, w_d, dt, g_0):
    """Calculate first n terms of a Fourier series for the digitised sine 
    torque. Return the value of the Fourier-calculated torque at specified 
    times. NOTE this can't deal with phi being nonzero!
    :param n: Number of Fourier series terms to return.
    :param w_d: Angular frequency of sine wave.
    :param dt: Sampling time of the sine for digitisation.
    :param g_0: Amplitude of the driving torque."""
    ind = np.arange(1, n, 1, dtype=np.int32)
    ind_a_b = calc_fourier_coeffs(ind, w_d, dt, g_0)
    ind_mag_phi = convert_to_mag_phis(ind_a_b[:, 0], ind_a_b[:, 1],
                                      ind_a_b[:, 2])
    return ind_mag_phi
