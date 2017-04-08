"""Alternative theory_calc scripts for a single sinusoidal torque."""

import numpy as np

import simulations.experiment.helpers as h


def calculate_cf_matrix(time, b, k, i):
    """Calculate the coefficients of A, B for the complementary function 
    solution of the oscillator ODE.
    :param time: Time array or single value in seconds.
    :param b: Damping coefficient.
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :return: Coefficients matrix: [[theta_A, theta_B], [omega_A, omega_B]]."""
    time = h.convert_to_array(time)
    w2, gamma = h.find_w2_gamma(b, k, i)
    print(w2, gamma)
    if w2 > 0:
        w = np.sqrt(w2)
        theta_coeffs = np.exp(-gamma * time / 2.) * np.array([np.exp(w * time),
                                                              np.exp(
                                                                  -w * time)])
        omega_coeffs = np.array(
            [(-gamma / 2. + w) * np.exp((-gamma / 2. + w) * time),
             (-gamma / 2. - w) * np.exp((-gamma / 2. - w) * time)])
    elif w2 == 0:
        theta_coeffs = np.exp(-gamma / 2. * time) * np.array(
            [np.ones(len(time)),
             time])
        omega_coeffs = np.exp(-gamma / 2. * time) * np.array(
            [-gamma / 2. * np.ones(len(time)), (1 - time * gamma / 2.)])
    else:  # w2 < 0
        w = np.sqrt(-w2)
        theta_coeffs = np.exp(-gamma / 2. * time) * np.array([np.cos(w * time),
                                                              np.sin(w * time)])
        omega_coeffs = np.exp(-gamma / 2. * time) * np.array(
            [(-w * np.sin(w * time) - gamma / 2. * np.cos(w * time)),
             (w * np.cos(w * time) - gamma / 2. * np.sin(w * time))])

    return np.array([theta_coeffs, omega_coeffs])


def calculate_sine_pi(t, b, k, i, g_0_mag, w_d, phi):
    """Calculate the particular integral contributions to theta and omega for a 
    sinusoidal forcing torque.
    :param t: Array or single value of time values.
    :param b: Damping coefficient
    :param k: Elastic coefficient.
    :param i: Moment of inertia.
    :param g_0_mag: Amplitude of driving torque.
    :param w_d: Angular frequency.
    :param phi: Phase in radians.
    :return: Theta and omega arrays at the times in t for the PI solution 
    part."""
    # Check all parameters are of correct format.
    g_0_mag, w_d, phi = h.check_types_lengths(g_0_mag, w_d, phi)

    # Calculate the forcing torque's amplitude.
    a_0 = g_0_mag * (k - i * w_d ** 2) / (
        b ** 2 * w_d ** 2 + (k - i * w_d ** 2) ** 2)
    b_0 = -g_0_mag * b * w_d / (b ** 2 * w_d ** 2 + (k - i * w_d ** 2) ** 2)

    theta_pi = a_0 * np.sin(w_d * t + phi) + b_0 * np.cos(w_d * t + phi)
    omega_pi = a_0 * w_d * np.cos(w_d * t + phi) - b_0 * w_d * np.sin(
        w_d * t + phi)
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
    pis = baked_torque(t_i, b, k, i)
    matrix = calculate_cf_matrix(t_i, b, k, i).squeeze()
    rhs_vector = np.array([[theta_i], [omega_i]]) - pis
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
    cf_constants = solve_for_ics(t_i, y_i[0], y_i[1], b, k, i,
                                 baked_torque).squeeze()
    cf_matrix = calculate_cf_matrix(t, b, k, i)
    results = np.einsum('ijk,j', cf_matrix, cf_constants)
    results += baked_torque(t, b, k, i)
    return np.array([t, results[0, :], results[1, :]]).T


def theory_response(b, k, i, b_prime, k_prime, w_d):
    """Gets the transfer function for a sinusoidal input torque. Note that 
    w_d must be the only array."""
    denominator = (-i * w_d ** 2 + w_d * (b - b_prime) * 1j + (k - k_prime))
    valids = np.where(denominator != 0)[0]
    print(denominator, valids)
    transfer = np.ones(denominator.shape) * np.Inf
    transfer[valids] = 1 / denominator[valids]
    return transfer

# baked = h.baker(calculate_sine_pi, ["", "", "", "", 1, 0.001, 0],
#                pos_to_pass_through=(0, 3))
# results = calc_theory_soln(np.arange(0, 50, 0.01), 0, [0, 0], -1, 5, 1,
#                           baked)
# plt.plot(results[:, 0], results[:, 1])
# plt.show()
#
# trans = theory_response(-1, 5, 1, 0, 0, np.arange(0, 140, 1))
# plt.plot(np.arange(0, 140, 1), np.absolute(trans)**2)
# plt.show()
