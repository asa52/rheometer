"""Code for the simulation of the pendulum under NR."""

import numpy as np
from scipy.integrate import ode

import helpers as h


def f_analytic(t, y, i, g_0, b, b_prime, k, k_prime):
    """Returns the RHS of the ODE y'(t) = f(t, y).
    :param t: Time in seconds.
    :param y: A vector of [theta, omega], where omega = d(theta)/dt.
    :param i: Moment of inertia of pendulum.
    :param g_0: Analytic driving torque function object, eg. g_0(t) = sin(t).
    :param b_prime: Simulated damping coefficient.
    :param k_prime: Simulated elastic coefficient.
    :param b: Actual damping coefficient of system.
    :param k: Actual elastic coefficient of system."""
    return [y[1], (g_0(t) - (b - b_prime) * y[1] - (k - k_prime) * y[0]) / i]


def f_full_torque(t, y, i, b, k, g):
    """RHS of the ODE with only full torque on the RHS. Variables defined as 
    above. Has the same Jacobian as f."""
    return np.array([y[1], (g(t) - b * y[1] - k * y[0]) / i])


def jac(t, y, i, b, k):
    """The Jacobian J[i, j] = df[i]/dy[j] of the above f."""
    return [[0, 1], [-k / i, -b / i]]


def ode_integrator(y0, t0, i, b_prime, k_prime, b, k, g_0_mags, w_ds, phases,
                   t_fin, dt, torque_func):
    r = ode(f_analytic).set_integrator('vode')
    baked_g_0 = h.baker(torque_func, args=['', w_ds, g_0_mags, phases])
    r.set_initial_value(y0, t0).set_f_params(i, baked_g_0, b, b_prime, k,
                                             k_prime)
    results = [[t0, *y0]]
    while r.successful() and r.t < t_fin:
        data_point = [r.t + dt, *np.real(r.integrate(r.t + dt))]
        results.append(data_point)
    exp_results = np.array(results)
    return exp_results


def analytic_torque(t, omega_ds, amplitudes, phases):
    """Return the value of the analytic driving torque at time t.
    :param t: Time in seconds - a single value.
    :param omega_ds: Angular frequency of the sinusoid.
    :param amplitudes: Amplitude of sinusoid.
    :param phases: Phase of sinusoid in radians."""

    amplitudes, omega_ds, phases = h.check_types_lengths(amplitudes, omega_ds,
                                                         phases)
    torque = 0
    for i in range(len(amplitudes)):
        torque += amplitudes[i] * np.sin(omega_ds[i] * t + phases[i])
    # ran = np.random.uniform() * amplitudes[0] * 0.1
    return torque #+ ran


def main():
    ode_integrator([0.1, 0], 0, 1e-7, 0, 0, 1e-6, 1e-6, 1e-7, 137, 0, 5, 3.8e-4,
                   analytic_torque)


def rk4(f):
    """Source: https://rosettacode.org/wiki/Runge-Kutta_method#Python"""
    return lambda t, y, dt: (
        lambda dy1: (
            lambda dy2: (
                lambda dy3: (
                    lambda dy4: (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6
                )(dt * f(t + dt, y + dy3))
            )(dt * f(t + dt / 2, y + dy2 / 2))
        )(dt * f(t + dt / 2, y + dy1 / 2))
    )(dt * f(t, y))