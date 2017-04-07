"""High-level functions to perform experiments on the system and collect a 
set of data, which can then be stored in a file. Maybe define a general 
experiment class here?"""

import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from simulations.experiment import conditional_pend as c
from simulations.experiment import helpers as h
from simulations.experiment import theory_calc as t
from simulations.configs_and_testing import testing as test


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
    config_dict = h.yaml_read('MeasuringAccuracy.yaml')
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


# TODO create a Nyquist checking function. Also a function to generate time
# TODO values equally spaced, but closely spaced near the expected peak.


class Experiment:
    """Define experiments here, with input data, an output file to save to, 
    and a sequence of operations that generates a final array of results."""

    def __init__(self, config=None, filename=None, description=None):
        # Set initial conditions. Create output file with the given name.
        # Read config parameters from a YAML file. Create a timer but don't
        # start. Description describes the aim of experiment.
        if filename is None:
            self.filename = str(self.__class__.__name__) + '-'
        if config is None:
            self.config = self.filename[:-1] + '.yaml'
        if description is None:
            self.description = self.__class__.__doc__

        self.string_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        self.filename += self.string_time
        self.timer = test.CodeTimer()
        self.log_text = ''

        # Create appropriate directories and read config_file.
        self.savepath = '../../../Tests/ExperimentClasses/{}/'.format(
            self.__class__.__name__)
        self.logpath = self.savepath + 'logs/'
        self.config_path = '../configs_and_testing/'
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)
        self.config_dict = h.yaml_read(self.config_path + self.config)

    def run(self, quick_save=True):
        """Run the experiment and get a set of results. Save at the end if 
        specified. Start timer and log runtime always, in a separate logging 
        file. If not auto_save, ask for comments before saving. Save the 
        range of parameters run for at the top in summarised form."""

        self.timer.start_timer()
        results = self.main_operation()
        timings = self.timer.see_checkpoints()

        # Save timing code.
        if not quick_save:
            # save results and logs.
            comments = input('Add some comments about this experiment: ')
        else:
            comments = 'None'

        with open(self.logpath + self.filename + '-log.txt', 'w') as log:
            log.write(
                'This is the log file for {}.\n\nExperiment description: {}\n\n'
                'Configuration:\n{}\n\nLogs:\n{}\n\nComments: {}.\n\nTime from '
                'start\tName\tCheckpoint no.\n'.format(
                    self.filename, self.description, self.config_dict,
                    self.log_text, comments))
        with open(self.logpath + self.filename + '-log.txt', 'ab') as log:
            np.savetxt(log, timings, fmt='%s')

        results.to_csv('{}'.format(self.savepath + self.filename + '.csv'))
        print('Results saved to {} and log file {}.'.format(
            self.filename + '.csv', self.filename + '-log.txt'))

    def _log(self, message, log_type='INFO'):
        """Add a log message along with the current time to the log file. Can 
        only be used during a main_operation function."""
        time_now = self.timer.checkpoint(checkp_name=message)[0]
        self.log_text += '{}\t{}\t{}\n'.format(time_now, log_type, message)

    def main_operation(self):
        """Override this with the main sequence of operations to be done for 
        each experiment. Return a pandas data frame with correctly labelled 
        columns. Note that innermost brackets represent a single row."""
        print('The experiment being run, {}, does not have a main_operation() '
              'method, so the base class one is being called.'.format(
            self.__class__.__name__))
        return pd.DataFrame(data=[3, 4, 5], columns=['Test'])


class MeasuringAccuracy(Experiment):
    """Compare the accuracy of the measurement functions to the theoretically 
    expected measurements."""

    def __init__(self, config=None, filename=None, description=None):
        super(MeasuringAccuracy, self).__init__(config, filename, description)

    def main_operation(self):
        # do stuff - needs changing!
        return


m1 = MeasuringAccuracy()
m1.run()


def measurement_accuracy(y0, t0, t, i, b_prime, k_prime, b, k, g_0_mags, w_ds,
                         phases, create_plot=False):
    """Compare the accuracy of the measurement functions and their error when a 
    noiseless, 'perfect' signal is measured and also compared to the 
    theoretically expected graph."""

    # Want to get set of measurements for a good range of k, b, i near the
    # experiment ones, and all 3 types of transients. Get the response curve
    # for each of these cases (over the frequency range 0 - 30 Hz only,
    # as this is the range to be tested). Also see what happens when b -
    # b_prime is negative, and k - k_prime. Check how measurements fare here.
    # Trivially, also try a case for varying the input torque amplitude and
    # phase to check that this does not have a significant effect.

    # Theoretical calculation.
    sines_torque = h.baker(
        t.calculate_sine_pi, ["", "", "", "", g_0_mags, w_ds, phases],
        position_to_pass_through=(0, 3))
    theory = t.calc_theory_soln(t, t0, y0, b - b_prime, k - k_prime, i,
                                sines_torque)

# if __name__ == '__main__':
#    exp_vs_an_parameters()
