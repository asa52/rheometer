import time
import numpy as np
from matplotlib import pyplot as plt

from simulations import conditional_pend as c
from simulations import helpers as h
from simulations import theory_calc as t


def exp_vs_theory(y0, t0, i, b_prime, omega_sim, k_prime, theta_sim, b, k,
                  g_0_mags, w_ds, phases, t_fin, dt, create_plot=False):
    """Compare experiment to theory for one set of parameters and return the 
    difference between the two. Uses only the analytic torque expression."""
    exp_results = c.ode_integrator(y0, t0, i, b_prime, omega_sim, k_prime,
                                   theta_sim, b, k, g_0_mags, w_ds, phases,
                                   t_fin, dt, c.analytic_torque)

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


def exp_vs_an_parameters():
    start_time = time.time()

    # Define the initial and setup conditions.
    config_dict = h.yaml_read('config.yaml')
    t0 = config_dict['t0']
    t_fin = config_dict['t_fin']
    dt = config_dict['dt']
    b_prime = config_dict['b_prime']
    k_prime = config_dict['k_prime']
    theta_sim = config_dict['theta_sim']
    omega_sim = config_dict['omega_sim']
    theta_0s = config_dict['theta_0s']
    omega_0s = config_dict['omega_0s']
    i_s = config_dict['i_s']
    bs = config_dict['bs']
    ks = config_dict['ks']
    g_0_mags = config_dict['g_0_mags']
    w_ds = config_dict['w_ds']
    phases = config_dict['phases']

    max_norm_errs = []
    for g_0_mag in g_0_mags:
        for w_d in np.nditer(w_ds):
            for phase in phases:
                for theta_0 in np.nditer(theta_0s):
                    for omega_0 in np.nditer(omega_0s):
                        for i in i_s:
                            for b in bs:
                                for k in ks:
                                    # todo change t0, i, etc. to 0th indices.
                                    err = exp_vs_theory(
                                        [theta_0, omega_0], t0, i, b_prime,
                                        omega_sim, k_prime, theta_sim, b, k,
                                        [g_0_mag, 0], [w_d, 0], [phase, 0],
                                        t_fin, dt)
                                    max_norm_errs.append(
                                        [t0, t_fin, dt, theta_0,
                                         omega_0, i, b, k, g_0_mag, w_d, phase,
                                         k_prime, theta_sim, b_prime,
                                         omega_sim, *err])
    max_norm_errs = np.array(max_norm_errs)
    np.savetxt('testing-sim-wo-norm-res.txt', max_norm_errs)
    runtime = time.time() - start_time
    print("runtime", runtime)
    return


if __name__ == '__main__':
    exp_vs_an_parameters()
