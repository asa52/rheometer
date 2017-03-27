"""Code for the simulation of the pendulum under NR."""

import time
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

import simulations.helpers as h
import simulations.theory_calc as t


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


def exp_vs_theory(y0, t0, i, b_prime, omega_sim, k_prime, theta_sim, b, k,
                  g_0_mags, w_ds, phases, t_fin, initial_dt, create_plot=False):
    """Compare experiment to theory for one set of parameters and return the 
    difference between the two."""

    r = ode(f, jac).set_integrator('dop853')
    baked_g_0 = h.baker(analytic_torque, args=['', w_ds, g_0_mags, phases])
    r.set_initial_value(y0, t0).set_f_params(
        i, baked_g_0, b_prime, omega_sim, k_prime, theta_sim, b,
        k).set_jac_params(i, b, k)

    results = [[t0, *y0]]
    while r.successful() and r.t < t_fin:
        data_point = [r.t+initial_dt, *np.real(r.integrate(r.t+initial_dt))]
        results.append(data_point)
        # TODO recalculate the parameters and set them again!
    exp_results = np.array(results)

    # Theoretical calculation.
    sines_torque = h.baker(
        t.calculate_sine_pi, ["", "", "", "", g_0_mags, w_ds, phases],
        position_to_pass_through=(0, 3))
    theory = t.calc_theory_soln(exp_results[:, 0], t0, y0, b - b_prime,
                                k - k_prime, i, sines_torque)

    # Normalise error by amplitude
    max_theta_diff = np.max(np.abs(exp_results[:, 1] - theory[:, 1]))
    max_omega_diff = np.max(np.abs(exp_results[:, 2] - theory[:, 2]))
    normalised_theta_diff = (exp_results[:, 1] - theory[:, 1])/np.max(
        exp_results[:, 1])
    normalised_omega_diff = (exp_results[:, 2] - theory[:, 2])/np.max(
        exp_results[:, 2])
    max_theta_norm = np.max(np.abs(normalised_theta_diff))
    max_omega_norm = np.max(np.abs(normalised_omega_diff))

    # Plotting - for 4 subplots on 1 figure.
    if create_plot:
        plt.figure(figsize=(7, 10))
        ax1 = plt.subplot(413)
        plt.plot(exp_results[:, 0], normalised_theta_diff, 'k',
                 label=r'$\theta$')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.grid()
        plt.ylabel(r'$(\theta_{sim}-\theta_{an})/|\theta_{max}|$', fontsize=14,
                   fontweight='bold')

        # share x only
        ax2 = plt.subplot(411, sharex=ax1)
        plt.plot(exp_results[:, 0], exp_results[:, 1], 'r-.',
                 label=r'Simulated')
        plt.plot(theory[:, 0], theory[:, 1], 'b:', label=r'Analytic')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2)
        plt.xlim([t0, t_fin])
        plt.ylabel(r'$\theta$/rad', fontsize=14, fontweight='bold')
        plt.grid()

        ax3 = plt.subplot(414, sharex=ax1)
        plt.plot(exp_results[:, 0], normalised_omega_diff, 'k',
                 label=r'$\omega$')
        plt.setp(ax1.get_xticklabels())
        plt.xlabel('t/s', fontsize=14, fontweight='bold')
        plt.ylabel(r'$(\omega_{sim}-\omega_{an})/|\omega_{max}|$',
                   fontsize=14, fontweight='bold')
        plt.grid()

        ax4 = plt.subplot(412, sharex=ax1)
        plt.plot(exp_results[:, 0], exp_results[:, 2], 'r-.',
                 label=r'$\omega_{exp}$')
        plt.plot(exp_results[:, 0], theory[:, 2], 'b:',
                 label=r'$\omega_{theory}$')
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.xlim([t0, t_fin])
        plt.ylabel('$\omega$/rad/s', fontsize=14, fontweight='bold')
        plt.ticklabel_format(useOffset=False)
        plt.grid()
        plt.show()

    return [max_theta_diff, max_omega_diff, max_theta_norm, max_omega_norm]


def main():
    start_time = time.time()
    # Constants for this test - doing with a single sine analytic forcing term
    # only!
    t0 = 0
    b_prime = 0
    k_prime = 0
    theta_sim = 0
    omega_sim = 0
    t_fin = 10
    initial_dt = 0.001

    # Variable parameters
    theta_0s = np.linspace(0, np.pi/4, 3)
    omega_0s = np.linspace(0, np.pi/4, 3)
    i_s = np.array([.00000001, .0000001, .000001])
    bs = np.array([.000000001, .000000005, .00000001, .00000005, .0000001])
    ks = np.array([.0000005, .000001, .000005])
    g_0_mags = np.array([.0000001, .0000005, .000001])
    w_ds = np.array([25])
    phases = np.array([0])

    max_norm_errs = []
    for g_0_mag in g_0_mags:
        for w_d in np.nditer(w_ds):
            for phase in phases:
                for theta_0 in np.nditer(theta_0s):
                    for omega_0 in np.nditer(omega_0s):
                        for i in i_s:
                            for b in bs:
                                for k in ks:
                                    err = exp_vs_theory(
                                        [theta_0, omega_0], t0, i, b_prime,
                                        omega_sim, k_prime, theta_sim, b, k,
                                        [g_0_mag, 0], [w_d, 0], [phase, 0],
                                        t_fin, initial_dt)
                                    max_norm_errs.append(
                                        [t0, t_fin, initial_dt, theta_0,
                                         omega_0, i, b, k, g_0_mag, w_d, phase,
                                         k_prime, theta_sim, b_prime,
                                         omega_sim, *err])
    max_norm_errs = np.array(max_norm_errs)
    np.savetxt('testing-sim-wo-norm-res.txt', max_norm_errs)
    runtime = time.time() - start_time
    print("runtime", runtime)

if __name__ == '__main__':
    main()
