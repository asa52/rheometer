"""High-level functions to perform experiments on the system and collect a 
set of data, which can then be stored in a file. Maybe define a general 
experiment class here?"""

import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from simulations.experiment import conditional_pend as c
from simulations.experiment import plotter as p
from simulations.experiment import helpers as h
from simulations.experiment import theory as t
from simulations.experiment import measurement as m
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
        pos_to_pass_through=(0, 3))
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
        self.prms = h.yaml_read(self.config_path + self.config)

    def run(self, tags=False, save=True):
        """Run the experiment and get a set of results. Save at the end if 
        specified. Start timer and log runtime always, in a separate logging 
        file. If not auto_save, ask for comments before saving. Save the 
        range of parameters run for at the top in summarised form. The dict 
        of dataframes from main_operation is saved in a separate file each."""

        self.timer.start_timer()
        results = self.main_operation()
        timings = self.timer.see_checkpoints()

        # Save timing code.
        if tags:
            # save results and logs.
            comments = input('Add some comments about this experiment: ')
        else:
            comments = 'None'

        with open(self.logpath + self.filename + '-log.txt', 'w') as log:
            log.write(
                'This is the log file for {}.\n\nExperiment description: {}\n\n'
                'Configuration:\n{}\n\nLogs:\n{}\n\nComments: {}.\n\nTime from '
                'start\tName\tCheckpoint no.\n'.format(
                    self.filename, self.description, self.prms,
                    self.log_text, comments))
        with open(self.logpath + self.filename + '-log.txt', 'ab') as log:
            np.savetxt(log, timings, fmt='%s')
        print('Log file saved to {}.'.format(self.filename + '-log.txt'))

        if save:
            for dataframe in results:
                results[dataframe].to_csv('{}'.format(self.savepath +
                                                      self.filename +
                                                      '-' + dataframe + '.csv'))
                print('Results saved to {}.'.format(self.filename + '.csv'))

    def main_operation(self):
        """Override this with the main sequence of operations to be done for 
        each experiment. Return a pandas data frame with correctly labelled 
        columns. Note that innermost brackets represent a single row."""
        print('The experiment being run, {}, does not have a main_operation() '
              'method defined, so the base class one (ie. from Experiment) is '
              'being called.'.format(self.__class__.__name__))
        return {'test': pd.DataFrame(data=[3, 4, 5], columns=['Test'])}

    def _log(self, message, log_type='INFO'):
        """Add a log message along with the current time to the log file. Can 
        only be used during a main_operation function."""
        time_now = self.timer.checkpoint(checkp_name=message)[0]
        self.log_text += '{}\t{}\t{}\n'.format(time_now, log_type, message)


class MeasuringAccuracy(Experiment):
    """Compare the accuracy of the measurement functions to the theoretically 
    expected measurements."""

    def __init__(self, config=None, filename=None, description=None):
        super(MeasuringAccuracy, self).__init__(config, filename, description)

    def _single_operation(self, times, b, k, i, b_prime, k_prime, w_d, g_0_mag,
                          phase, n_frq_peak, t0, y0):
        """One run for one value of b, k, i, b_prime, k_prime, 
        torque_amplitude, torque_phase, torque_frequency and noise. t is a time 
        array."""

        times = h.check_nyquist(times, w_d, b, b_prime, k, k_prime, i)

        torque = g_0_mag * np.sin(w_d * times + phase)
        pi_contr = h.baker(t.calculate_sine_pi, [
            "", "", "", "", g_0_mag, w_d, phase], pos_to_pass_through=(0, 3))
        theory = t.calc_theory_soln(times, t0, y0, b - b_prime, k - k_prime, i,
                                    pi_contr)
        self._log('after nyquist check')
        if b - b_prime > 0:
            # Will only reach steady state if this is the case, otherwise no
            # point making a response curve. Measure one point.
            ss_times = m.identify_ss(times, theory[:, 1])
            self._log('before fft')
            frq, fft_theta = m.calc_fft(
                times[(times >= ss_times[0]) * (times <= ss_times[1])],
                theory[:, 1][(times >= ss_times[0]) * (times <= ss_times[1])])
            # for low frequencies, the length of time of the signal must also
            # be sufficiently wrong for the peak position to be measured
            # properly.
            self._log('before mmts')
            # Half-amplitude of peak used to calculate bandwidth.
            freq = m.calc_freqs(np.absolute(fft_theta), frq, n_peaks=n_frq_peak)
            amp = m.calc_one_amplitude(theory[:, 1][(times >= ss_times[0]) *
                                                    (times <= ss_times[1])])
            phase = m.calc_phase(theory[:, 1], torque)
            self._log('after mmts')
            return True, theory, np.array([freq, amp, phase])
        else:
            return False, theory

    def main_operation(self):
        # feed in fixed b - b', k - k', torque, i, noise level and type,
        # but a range of driving frequencies to test.

        b_s = self.prms['b_s']
        k_s = self.prms['k_s']
        i_s = self.prms['i_s']
        b_primes = self.prms['b\'s']
        k_primes = self.prms['k\'s']
        w_ds = np.arange(self.prms['w_d_start'], self.prms['w_d_final'],
                         self.prms['w_d_step'])
        g_0_mags = self.prms['g0s']
        phases = self.prms['phis']
        t0 = self.prms['t0'].squeeze()
        y0 = np.array([self.prms['theta_0'], self.prms[
            'omega_0']]).squeeze()
        times = np.arange(t0, self.prms['tfin'], self.prms['dt'])
        self._log('after config setup')

        # Get the theoretical position of the peak and generate a finely
        # spaced set of frequency points around it to measure with greater
        # resolution here.
        w2, gamma = h.find_w2_gamma(b_s - b_primes, k_s - k_primes, i_s)

        # 5 * bandwidth on either side
        gamma = np.absolute(gamma)
        width = 5 * gamma if gamma != 0 else self.prms['w_d_step']
        if w2 <= 0:
            w_res = np.sqrt(-w2)
        else:
            w_res = 0
        w_range = np.linspace(w_res - width, w_res + width, 20)
        single_run = h.baker(self._single_operation,
                             [times, '', '', '', '', '', '', '', '', '', t0,
                              y0], pos_to_pass_through=(1, 9))

        ws = [w_ds, w_range]
        self._log('after baker function')
        for j in range(len(ws)):
            all_times = h.all_combs(single_run, b_s, k_s, i_s, b_primes,
                                    k_primes, ws[j], g_0_mags, phases, 1)
            self._log('after all combs run')
            # final argument 1/2 for 1 frequency peak expected.
            fft_data = []
            real_data = []
            for i in range(len(all_times)):
                real_data.append(all_times[i][-1][1])
                if all_times[i][-1][0]:
                    fft_data.append(all_times[i][-1][-1])
            if j == 0:
                fft_mmts = np.array(fft_data)
                # Get the theoretical response curves. Note that w_d is the only
                # array.
                fft_theory = np.array(h.all_combs(t.theory_response, b_s, k_s,
                                                  i_s, b_primes, k_primes,
                                                  ws[j]))
            else:
                fft_mmts = np.append(fft_mmts, np.array(fft_data), axis=0)
                fft_mmts = fft_mmts[fft_mmts[:, 0, 0].argsort()]
                fft_theory = np.append(fft_theory, np.array(h.all_combs(
                    t.theory_response, b_s, k_s, i_s, b_primes, k_primes,
                    ws[j])), axis=0)
                fft_theory = fft_theory[fft_theory[:, -2].argsort()]
            self._log('after fft setup')
        ang_freqs = np.array([fft_mmts[:, 0, 0],
                              fft_mmts[:, 0, 1]]).T * 2 * np.pi
        amps = np.array([fft_mmts[:, 1, 0], fft_mmts[:, 1, 1]]).T / g_0_mags
        phases = np.array([fft_mmts[:, 2, 0], fft_mmts[:, 2, 1]]).T

        theory_n_mmt = \
            [
                [
                    [
                        [fft_theory[:, -2], np.absolute(fft_theory[:, -1])],
                        [ang_freqs, amps]
                    ],
                    [
                        [fft_theory[:, -2], np.angle(fft_theory[:, -1])],
                        [ang_freqs, phases]
                    ]
                ]
            ]
        self._log('before plot')
        p.two_by_n_plotter(theory_n_mmt, x_axes_labels=['$\omega$/rad/s'],
                           y_top_labels=[r'$\left|\frac{\theta(\omega)}{G('
                                         r'\omega)}\right|$'],
                           y_bottom_labels=[r'arg$(\frac{\theta(\omega)}{G('
                                            r'\omega)})$'])
        self._log('after plot')
        #fft_theory = pd.DataFrame(fft_theory, columns=['b', 'k', 'I', 'b\'',
        #                                               'k\'', 'w_d',
        #                                               'transfer'])
        #resp_curve2 = pd.DataFrame(np.array(h.all_combs(
        #    t.theory_response, b_s, k_s, i_s, b_primes, k_primes, w_range)),
        #     columns=['b', 'k', 'I', 'b\'', 'k\'', 'w_d', 'transfer'])
        #resp_curve2 = resp_curve2.append(resp_curve2).sort_values('w_d')
        #return super(MeasuringAccuracy, self).main_operation()


# plot time domain displacement and velocity using theory and check it
# agrees with analytic forms once in each of 3 types of transients
# and b - b', k - k' <, =, > 0.

# For each value of b - b', k - k' and other parameters varied,
# plot the expected response curve against the actual, which is
# obtained by using the measurement functions and varying the driving
# frequency.


# Nyquist sampling error! Be aware of this! Both freq and amplitude can
# be wrongly measured as a result! Digitisation error also. Also set the
# first_is_peak parameter if first signal in the amplitude can be the
# peak (true usually only for noiseless signals).
# If another frequency present, perhaps filter the signal first?


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
        pos_to_pass_through=(0, 3))
    theory = t.calc_theory_soln(t, t0, y0, b - b_prime, k - k_prime, i,
                                sines_torque)

# if __name__ == '__main__':
#    exp_vs_an_parameters()
