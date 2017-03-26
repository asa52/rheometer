import time
import numpy as np
from scipy.integrate import ode

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


def analytic_torque(t, omega_d, amplitude, phase):
    """Return the value of the analytic driving torque at time t.
    :param t: Time in seconds.
    :param omega_d: Angular frequency of the sinusoid.
    :param amplitude: Amplitude of sinusoid.
    :param phase: Phase of sinusoid in radians."""

    # TODO Add noise.
    return amplitude * np.sin(omega_d * t + phase)


def exp_vs_theory(y0, t0, i, b_prime, omega_sim, k_prime, theta_sim, b, k,
                  g_0_mags, w_ds, phases, t_fin, initial_dt):
    r = ode(f, jac).set_integrator('dop853')
    baked_g_0 = h.baker(analytic_torque, args=['', np.pi / 2, 1, 0])
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
    normalised_theta_diff = (exp_results[:, 1] - theory[:, 1])/max(theory[:, 1])
    normalised_omega_diff = (exp_results[:, 2] - theory[:, 2])/max(theory[:, 2])
    theta_max_discrep = max(np.abs(normalised_theta_diff))
    omega_max_discrep = max(np.abs(normalised_omega_diff))
    return [theta_max_discrep, omega_max_discrep]

    # Plotting - for 4 subplots on 1 figure.
    #ax1 = plt.subplot(223)
    #plt.plot(exp_results[:, 0], normalised_theta_diff, 'k', label=r'$\theta$')
    #plt.setp(ax1.get_xticklabels())
    #plt.xlabel('t/s')
#
    ## share x only
    #ax2 = plt.subplot(221, sharex=ax1)
    #plt.plot(exp_results[:, 0], exp_results[:, 1], 'r-.', label=r'$\theta_{'
    #                                                          r'exp}$')
    #plt.plot(exp_results[:, 0], theory[:, 1], 'b:', label=r'$\theta_{theory}$')
    #plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.legend(loc='best')
    #plt.xlim([t0, t_fin])
#
    #ax3 = plt.subplot(224, sharey=ax1)
    #plt.plot(exp_results[:, 0], normalised_omega_diff, 'k', label=r'$\omega$')
    #plt.setp(ax1.get_xticklabels())
    #plt.setp(ax3.get_yticklabels(), visible=False)
    #plt.xlabel('t/s', fontsize=14, fontweight='bold')
#
    #ax4 = plt.subplot(222, sharex=ax3)
    #plt.plot(exp_results[:, 0], exp_results[:, 2], 'm-.', label=r'$\omega_{'
    #                                                            r'exp}$')
    #plt.plot(exp_results[:, 0], theory[:, 2], 'g:', label=r'$\omega_{theory}$')
    #plt.setp(ax4.get_xticklabels(), visible=False)
    #plt.legend(loc='best')
    #plt.xlim([t0, t_fin])
#
    #plt.ylabel('$\omega$/rad/s', fontsize=14, fontweight='bold')
    #plt.ticklabel_format(useOffset=False)
    #plt.show()


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
    theta_0s = np.linspace(0, np.pi/2, 5)
    omega_0s = np.linspace(0, 1., 5)
    i_s = [.00000001, 0.0000001, 0.000001]
    bs = [.000000001, .000000005, .00000001, .00000005, .0000001]
    ks = [.0000005, .000001, .000005]
    g_0_mags = [.0000001, .0000005, .000001]
    w_ds = np.array([np.pi / 2])
    phases = [0]

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
    np.savetxt('testing-sim-wo-norm-res', max_norm_errs)
    runtime = time.time() - start_time
    print("runtime", runtime)

if __name__ == '__main__':
    main()
