"""Calculate the theoretical values for a given system."""

import numpy as np
import simulations.helpers as h
import matplotlib.pyplot as plt


def calculate_cf(time, b, k, i):
    """Calculate the coefficients of A, B for the complementary function 
    solution of the oscillator ODE.
    :param time: Time array or single value in seconds.
    :param b: Damping coefficient.
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :return: Coefficients matrix: [[theta_A, theta_B], [omega_A, omega_B]]."""

    time = h.convert_to_array(time)
    w2, gamma = find_w2_gamma(b, k, i)
    if w2 > 0:
        w = np.sqrt(w2)
        theta_coeffs = np.exp(-gamma * time) * np.array([np.exp(w * time),
                                                         np.exp(-w * time)])
        omega_coeffs = np.array([(-gamma + w) * np.exp((-gamma + w) * time),
                                 (-gamma - w) * np.exp((-gamma - w) * time)])
    elif w2 == 0:
        theta_coeffs = np.exp(-gamma * time) * np.array([1, time])
        omega_coeffs = np.exp(-gamma * time) * np.array([-gamma,
                                                         (1 - time * gamma)])
    else:  # w2 < 0
        w = np.sqrt(-w2)
        theta_coeffs = np.exp(-gamma * time) * np.array([np.cos(w * time),
                                                         np.sin(w * time)])
        omega_coeffs = np.exp(-gamma * time) * np.array(
            [(-w * np.sin(w * time) - gamma * np.cos(w * time)),
             (w * np.cos(w * time) - gamma * np.sin(w * time))])

    return np.array([theta_coeffs, omega_coeffs])


def find_w2_gamma(b, k, i):
    w2 = b ** 2 / (4 * i ** 2) - k / i
    gamma = b / (2 * i)
    return w2, gamma


def calculate_sine_pi(t, b, k, i, g_0_mags, w_ds, phis):
    """Calculate the particular integral contributions to theta and omega.
    :param t: Array or single value of time values.
    :param b: Damping coefficient
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :param g_0_mags: Amplitude of driving torques as an array.
    :param w_ds: Angular frequencies array, in same order as g_0_mags.
    :param phis: Phases array.
    :return: Theta and omega arrays at the times in t for the PI solution 
    part."""
    # Check all parameters are of correct format.
    g_0_mags, w_ds, phis = h.check_types_lengths(g_0_mags, w_ds, phis)

    # Calculate the forcing torque's amplitude.
    a_0 = g_0_mags * (k - i * w_ds ** 2) / (
        b ** 2 * w_ds ** 2 + (k - i * w_ds ** 2) ** 2)
    b_0 = -g_0_mags * b * w_ds / (b ** 2 * w_ds ** 2 + (k - i * w_ds ** 2) ** 2)

    # Use these calculated values to work out theta and omega for the
    # required times.
    time_bit = np.outer(w_ds, t).squeeze()
    a_0, b_0, phis, w_ds = h.make_same_dim(a_0, b_0, phis, w_ds,
                                           ref_dim_array=time_bit)
    theta_pi = a_0 * np.sin(time_bit + phis) + b_0 * np.cos(time_bit + phis)
    omega_pi = a_0 * w_ds * np.cos(time_bit + phis) - b_0 * w_ds * np.sin(
        time_bit + phis)
    if theta_pi.squeeze().ndim == 1 and omega_pi.squeeze().ndim == 1:
        theta_pi = np.sum(theta_pi.squeeze())
        omega_pi = np.sum(omega_pi.squeeze())
    elif theta_pi.squeeze().ndim == 2 and theta_pi.squeeze().ndim == 2:
        theta_pi = np.sum(theta_pi.squeeze(), axis=0)
        omega_pi = np.sum(omega_pi.squeeze(), axis=0)
    else:
        raise Exception("Something has gone wrong with array dimensions!")
    return np.array([theta_pi, omega_pi])


def solve_for_ics(t_i, theta_i, omega_i, b, k, i, baked_torque):
    """Calculate the coefficients for this ODE based on the initial conditions.
    :param t_i: Initial time.
    :param theta_i: Initial displacement.
    :param omega_i: Initial angular velocity.
    :param b: Damping coefficient.
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :param baked_torque: The forcing torque with the relevant parameters 
    baked in, such as amplitudes, period, phases.
    :return: The values of the coefficients A and B for the complementary 
    function."""
    pis = (baked_torque(t_i, b, k, i)).squeeze()
    matrix = calculate_cf(t_i, b, k, i).squeeze()
    rhs_vector = (np.array([[theta_i], [omega_i]]).squeeze() - pis)
    return np.linalg.solve(matrix, rhs_vector)


def calc_theory_soln(t, t_i, y_i, b, k, i, baked_torque):
    """NOTE: t must be an array with at least 2 values. So should the torque 
    amplitudes, phis and w_ds.
    :param t: Array of time values to plot for.
    :param t_i: Initial time.
    :param y_i: Initial position, velocity.
    :param b: Damping coefficient.
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :param baked_torque: The torque function to calculate the theta_ss 
    parameters with relevant parameters baked in.
    :return: Array of [t, theta, omega], each column being a different 
    variable."""
    cf_constants = solve_for_ics(t_i, y_i[0], y_i[1], b, k, i, baked_torque)
    results = np.einsum('ijk,j', calculate_cf(t, b, k, i), cf_constants)
    results += baked_torque(t, b, k, i)
    return np.array([t, results[0], results[1]]).T
