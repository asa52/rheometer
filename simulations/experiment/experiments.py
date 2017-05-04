"""High-level functions to perform experiments on the system and collect a 
set of data, which can then be stored in a file."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import ode

import conditional_pend as c
import fourier_decomp as fourier
import helpers as h
import measurement as m
import plotter as p
import read_saved as r
import theory as t
import noise_delays as n
# import c_talker as talk


class Experiment:
    """Define experiments here, with input data, an output file to save to, 
    and a sequence of operations that generates a final array of results."""

    def __init__(self, config=None, filename=None, description=None):
        # Set initial conditions. Create output file with the given name.
        # Read config parameters from a YAML file or feed in dictionary
        # directly. Create a timer but don't start. Description describes the
        # aim of experiment.
        if filename is None:
            self.filename_root = str(self.__class__.__name__) + '-'
        if config is None:
            self.config = self.filename_root[:-1] + '.yaml'
        if description is None:
            self.description = self.__class__.__doc__

        self._update_filename()
        self.timer = h.CodeTimer()
        self.log_text = ''

        # Create appropriate directories and read config_file.
        self.savepath = '../../../Tests/ExperimentClasses/{}/'.format(
            self.__class__.__name__)
        self.logpath = self.savepath + 'logs/'
        self.plotpath = self.savepath + 'plots/'
        self.config_path = '../configs/'
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)
        if not os.path.exists(self.plotpath):
            os.makedirs(self.plotpath)
        if type(config) is not dict:
            self.prms = h.yaml_read(self.config_path + self.config)
        else:
            self.prms = config

    def _update_filename(self):
        """Get an updated filename with the most recent time, allowing 
        multiple files to be saved by the class."""
        self.string_time = h.time_for_name()
        self.filename = self.filename_root + self.string_time

    def run(self, tags=False, savedata=True, plot=True):
        """Run the experiment and get a set of results. Save at the end if 
        specified. Start timer and log runtime always, in a separate logging 
        file. Save the range of parameters run for at the top in summarised 
        form. The dict of data frames from main_operation is saved in a separate 
        file each."""

        self.timer.start_timer()
        results = self.main_operation(plot=plot)
        timings = self.timer.see_checkpoints()

        # Save timing code.
        if tags:
            # save results and logs.
            comments = input('Add some comments about this experiment: ')
        else:
            comments = 'None'

        if savedata:
            with open(self.logpath + self.filename + '-log.txt', 'w') as log:
                log.write(
                    'This is the log file for {}.\n\nExperiment description: {}'
                    '\n\nConfiguration:\n{}\n\nLogs:\n{}\n\nComments: {}.\n\n'
                    'Time from start\tName\tCheckpoint no.\n'.format(
                        self.filename, self.description, self.prms,
                        self.log_text, comments))
            with open(self.logpath + self.filename + '-log.txt', 'ab') as log:
                np.savetxt(log, timings, fmt='%s')
            print('Log file saved to {}.'.format(self.filename + '-log.txt'))
            for dataframe in results:
                results[dataframe].to_csv('{}'.format(self.savepath +
                                                      self.filename +
                                                      '-' + dataframe + '.csv'))
                print('Results saved to {}.'.format(self.filename + '.csv'))

        return results

    def main_operation(self, plot=True):
        """Override this with the main sequence of operations to be done for 
        each experiment. Return a pandas data frame with correctly labelled 
        columns. Note that innermost brackets represent a single row."""
        print('The experiment being run, {}, does not have a main_operation() '
              'method defined, so the base class one (ie. from Experiment) is '
              'being called.'.format(self.__class__.__name__))
        return {'test': pd.DataFrame(data=[3, 4, 5], columns=['Test'])}

    def _log(self, message, log_type='INFO', printit=False):
        """Add a log message along with the current time to the log file. Can 
        only be used during a main_operation function."""
        time_now = self.timer.checkpoint(checkp_name=message)[0]
        self.log_text += '{}\t{}\t{}\n'.format(time_now, log_type, message)
        if printit:
            print(message)

    def __del__(self):
        """Deletes the timer object when the experiment is being deleted."""
        del self.timer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


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

        times = m.check_nyquist(times, w_d, b, b_prime, k, k_prime, i)
        decimal_periods = np.abs(5.231 * 2*np.pi/w_d)

        time_index = 0
        num = 1
        # The index for 5.231 periods of time being elapsed is used to
        # calculate the number per segment to use for the autocorrelation.
        while time_index < 10:
            time_index = np.abs(times - num * decimal_periods).argmin()
            num += 1

        torque = g_0_mag * np.sin(w_d * times + phase)
        pi_contr = h.baker(t.calculate_sine_pi, [
            "", "", "", "", g_0_mag, w_d, phase], pos_to_pass_through=(0, 3))
        theory = t.calc_theory_soln(times, t0, y0, b - b_prime, k - k_prime, i,
                                    pi_contr)
        w_res = t.w_res_gamma(b - b_prime, k - k_prime, i)[0]
        if b - b_prime >= 0:
            if b - b_prime == 0 and np.isreal(w_res):
                # filter out the transient frequency.
                theory[:, 1] = m.remove_one_frequency(times, theory[:, 1],
                                                      w_res)
                theory[:, 2] = m.remove_one_frequency(times, theory[:, 2],
                                                      w_res)

            # Will only reach steady state if this is the case, otherwise no
            # point making a response curve. Measure one point. b - b' = 0
            # has two steady state frequencies, the transient and PI.
            ss_times = m.identify_ss(times, theory[:, 1],
                                     n_per_segment=time_index)

            if ss_times is not False:
                self._log('before fft')
                frq, fft_theta = m.calc_fft(
                    times[(times >= ss_times[0]) * (times <= ss_times[1])],
                    theory[:, 1][(times >= ss_times[0]) * (times <=
                                                           ss_times[1])])
                # For low frequencies, the length of time of the signal must
                # also be sufficiently wrong for the peak position to be
                # measured properly.
                freq = m.calc_freqs(np.absolute(fft_theta), frq,
                                    n_peaks=n_frq_peak)
                amp = m.calc_one_amplitude(theory[:, 1][(times >= ss_times[0]) *
                                                        (times <= ss_times[1])])
                phase = m.calc_phase(theory[:, 1], torque)
                return True, theory, np.array([freq, amp, phase])
            else:
                return False, theory
        else:
            return False, theory

    def main_operation(self, plot=False):
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

        # Get the theoretical position of the peak and generate a finely
        # spaced set of frequency points around it to measure with greater
        # resolution here.
        w2_res = (k_s - k_primes) / i_s - (b_s - b_primes) ** 2 / (2 * i_s **
                                                                   2)
        gamma = (b_s - b_primes)/i_s

        # 5 * bandwidth on either side
        gamma = np.absolute(gamma)
        width = 5 * gamma if gamma != 0 else self.prms['w_d_step']
        if w2_res >= 0:
            w_res = np.sqrt(w2_res)
        else:
            w_res = 0
        w_range = np.linspace(w_res - width if w_res - width > 2 else 2,
                              w_res + width, 100)
        single_run = h.baker(self._single_operation,
                             [times, '', '', '', '', '', '', '', '', '', t0,
                              y0], pos_to_pass_through=(1, 9))

        ws = [w_ds, w_range]
        for j in range(len(ws)):
            all_times = h.all_combs(single_run, b_s, k_s, i_s, b_primes,
                                    k_primes, ws[j], g_0_mags, phases, 1)

            # final argument 1/2 for 1 frequency peak expected.
            fft_data = []
            real_data = []
            for i in range(len(all_times)):
                real_data.append(all_times[i][-1][1])
                if all_times[i][-1][0]:
                    fft_data.append(all_times[i][-1][-1])
            if j == 0:
                fft_mmts = np.array(fft_data)
            else:
                fft_mmts = np.append(fft_mmts, np.array(fft_data), axis=0)
                fft_mmts = fft_mmts[fft_mmts[:, 0, 0].argsort()]

        ang_freqs = np.array([fft_mmts[:, 0, 0],
                              fft_mmts[:, 0, 1]]).T * 2 * np.pi
        amps = np.array([fft_mmts[:, 1, 0], fft_mmts[:, 1, 1]]).T / g_0_mags
        phases = np.array([fft_mmts[:, 2, 0], fft_mmts[:, 2, 1]]).T

        # For a single set of data, get the transfer function once only. This
        # allows error to be calculated as specifically the experimental
        # ang_freqs are used.
        fft_theory = np.array(h.all_combs(t.theory_response, b_s, k_s, i_s,
                                          b_primes, k_primes, ang_freqs[:, 0]))

        theory_amps = np.absolute(fft_theory[:, -1])
        theory_phases = np.angle(fft_theory[:, -1])
        amp_err, phase_err = m.calc_norm_errs(
            [amps[:, 0], theory_amps], [phases[:, 0], theory_phases])[1]

        theory_n_mmt = \
            [[[[ang_freqs, theory_amps], [ang_freqs, amps]],
              [[ang_freqs, amp_err / np.max(amps[:, 0])]]],
             [[[ang_freqs, theory_phases], [ang_freqs, phases]],
              [[ang_freqs, phase_err / np.max(np.absolute(phases[:, 0]))]]]]

        p.two_by_n_plotter(
            theory_n_mmt, self.filename, self.prms, savepath=self.plotpath,
            show=False, x_axes_labels=['$\omega$/rad/s', '$\omega$/rad/s'],
            y_top_labels=[r'$\left|R(\omega)\right|$/rad/(Nm)',
                          r'arg[$R(\omega)$]/rad'],
            y_bottom_labels=[r'Normalised error in $\left|R(\omega)\right|$',
                             r'Normalised error in arg[$R(\omega)$]'])

    def run(self, tags=None, savedata=False, plot=False):
        # Don't save data.
        super(MeasuringAccuracy, self).run(tags=tags, savedata=savedata,
                                           plot=plot)


class TheoryVsFourier(Experiment):
    """Experiment to compare the analytical theory solution to the digitised 
    sinusoidal solution."""

    def __init__(self):
        super(TheoryVsFourier, self).__init__()
        self.divider = self.prms['max_step_divider']

    def main_operation(self, plot=None):
        # Define the initial and setup conditions.
        t0 = self.prms['t0']
        t_fin = self.prms['tfin']
        w_d = self.prms['w_d']
        dt = 2 * np.pi / (w_d * self.prms['sampling_divider'])
        b_prime = self.prms['b\'']
        k_prime = self.prms['k\'']
        theta_0 = self.prms['theta_0']
        omega_0 = self.prms['omega_0']
        i_ = self.prms['i']
        b_ = self.prms['b']
        k_ = self.prms['k']
        g_0_mag = self.prms['g_0_mag']
        phase = self.prms['phi']

        max_errs = self._one_run(theta_0, omega_0, t0, i_, b_prime, k_prime, b_,
                                 k_, g_0_mag, w_d, phase, t_fin, dt)
        max_errs = pd.DataFrame(data=max_errs, columns=[
            'theta_0', 'omega_0', 't0', 'i', 'b\'', 'k\'', 'b', 'k', 'g_0_mag',
            'w_d', 'phase', 't_fin', 'dt', 'max_theta_diff', 'max_omega_diff',
            'max_theta_norm', 'max_omega_norm'])
        return {'max_errs': max_errs}

    def _one_run(self, theta_0, omega_0, t0, i, b_prime, k_prime, b, k,
                 g_0_mag, w_d, phase, t_fin, dt, create_plot=True):
        """Compare experiment to theory for one set of parameters and return the 
        difference between the two. Uses only the analytic torque expression."""
        y0 = [theta_0, omega_0]
        times = np.arange(t0, t_fin, dt / self.divider)

        # Standard sine calculation.
        # Calculate fourier series first 1000 terms.
        ind_mag_phi = fourier.get_fourier_series(1000, w_d, dt, g_0_mag)
        print(ind_mag_phi)
        f_torque = fourier.get_torque(times, ind_mag_phi[:, 0],
                                      ind_mag_phi[:, 1], ind_mag_phi[:, 2], w_d)

        # Calculate theoretical vs actual torque and plot.
        plt.plot(times, f_torque, label='Digitised')
        plt.plot(times, g_0_mag * np.sin(w_d * times + phase), label='Analogue')
        plt.tick_params(direction='out')
        plt.grid(True)
        plt.xlabel('t/s', fontweight='bold', fontsize=13)
        plt.ylabel('$G_{tot}$/Nm', fontweight='bold', fontsize=13)
        plt.show()

        # Calculate theoretical results.
        torque_sine = h.baker(t.calculate_sine_pi,
                              ["", "", "", "", g_0_mag, w_d, phase],
                              pos_to_pass_through=(0, 3))
        theory = t.calc_theory_soln(times, t0, y0, b - b_prime, k - k_prime, i, 
                                    torque_sine)

        # fourier theoretical results.
        torque_fourier = h.baker(t.calculate_sine_pi,
                                 ["", "", "", "", ind_mag_phi[:, 1],
                                  w_d * ind_mag_phi[:, 0], ind_mag_phi[:, 2]],
                                 pos_to_pass_through=(0, 3))
        theory_fourier = t.calc_theory_soln(times, t0, y0, b - b_prime, 
                                            k - k_prime, i, torque_fourier)

        # Normalise error by amplitude
        max_theta_diff = np.max(np.abs(theory_fourier[:, 1] - theory[:, 1]))
        max_omega_diff = np.max(np.abs(theory_fourier[:, 2] - theory[:, 2]))
        norm_theta_diff = (theory_fourier[:, 1] - theory[:, 1]) / np.max(
            theory_fourier[:, 1])
        norm_omega_diff = (theory_fourier[:, 2] - theory[:, 2]) / np.max(
            theory_fourier[:, 2])
        max_theta_norm = np.max(np.abs(norm_theta_diff))
        max_omega_norm = np.max(np.abs(norm_omega_diff))

        # Plotting - for 4 subplots on 1 figure.
        if create_plot:
            self._update_filename()
            plotting_data = \
                [
                    [
                        [[theory_fourier[:, 0], theory_fourier[:, 1],
                          r'Digitised'],
                         [theory[:, 0], theory[:, 1], r'Analogue']],
                        [[theory_fourier[:, 0], norm_theta_diff]],
                    ],
                    [
                        [[theory_fourier[:, 0], theory_fourier[:, 2],
                          r'Digitised'],
                         [theory[:, 0], theory[:, 2], r'Analogue']],
                        [[theory_fourier[:, 0], norm_omega_diff]],
                    ]
                ]
            params = {'theta_0': theta_0, 'omega_0': omega_0, 't0': t0, 'I': i,
                      'b\'': b_prime, 'k\'': k_prime, 'b': b, 'k': k,
                      'g_0_mag': g_0_mag, 'w_d': w_d, 'phi': phase,
                      't_fin': t_fin, 'dt': dt}
            p.two_by_n_plotter(
                plotting_data, self.filename, params, savepath=self.plotpath,
                show=True, x_axes_labels=['t/s', 't/s'],
                tag='with-uniform-noise-torque-over-100',
                y_top_labels=[r'$\theta$/rad', r'$\dot{\theta}$/rad/s'],
                y_bottom_labels=[r'$(\theta_{sim}-\theta_{an})/|\theta_{max}|$',
                                 r'$(\dot{\theta}_{sim}-\dot{\theta}_{an})/'
                                 r'|\dot{\theta}_{max}|$'],
                legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, 1),
                        'ncol': 2})

        return [max_theta_diff, max_omega_diff, max_theta_norm, max_omega_norm]


class NRRegimes(Experiment):
    """Tests the theoretical against the ODE integrator function in the time 
    domain for one set of control parameters, allowing for variable noise and 
    delays to be introduced at any stage."""

    def __init__(self):
        super(NRRegimes, self).__init__()

        # set initial parameters
        i = self.prms['i']
        b = self.prms['b']
        k = self.prms['k']
        y0 = [self.prms['theta_0'], self.prms['omega_0']]
        t0 = self.prms['t0']
        t_fin = self.prms['tfin']

        # Get the driving frequency and hence the sampling rate.
        self.w_d = self.prms['w_d']
        self.dt = 2 * np.pi / (self.w_d * 120)
        # 120 as 120 points per cycle on C code

        self.display_dt = self.dt / 10  # The time on which to display
        # calculations
        # Convert a torque from the config file to a digitised value.
        self.g0_volt = self._torque_to_volt(self.prms['g_0_mag'])
        # convert centre point of voltage range to a voltage.
        # self.prms['torque_to_current_err'] TODO where to use this?

        # k' and b' conversion
        self.k_pr_volt = self._torque_to_volt(
            self.prms['k\''], subtr_midpoint=False) * self.prms['rad_to_0.1um']
        self.b_pr_volt = self._torque_to_volt(
            self.prms['b\''], subtr_midpoint=False) / self.prms['rad_to_0.1um']\
            * self.dt   # todo check this

        assert len(self.k_pr_volt) == len(self.b_pr_volt) == len(self.g0_volt) \
            == 1, "Only one set of parameters can be run at once!"

        # Set the parameters in the Arduino script.
        talk.set_k_b_primes(int(self.k_pr_volt[0]), int(self.b_pr_volt[0]))
        talk.set_amp(int(self.g0_volt[0]))
        # set initial conditions
        #talk.set_mu(int(theta in rad) / self.prms['rad_to_0.1um'])
        # set dmudt also at time 0

        print("Init parameters: dt: {}, display_dt: {}, b: {}, b': {}, k: {}, "
              "k': {}, I: {}, y0: {}, t0: {}, tfin: {}, g0: {}, w_d: {}".format(
            self.dt, self.display_dt, b, self.prms['b\''], k, self.prms[
                'k\''], i, y0, t0, t_fin, self.prms['g_0_mag'], self.prms[
                'w_d']))
        print("Parameters from the C code: k': {}, b': {}, g0: {}".format(
            talk.get_k_prime(), talk.get_b_prime(), talk.get_amp()))

        # set initial parameters.
        self.digitised_torque = 0
        self.digitised_theta_sim = 0
        self.digitised_omega_sim = 0
        self.total_torque = 0
        self.theta_sim = 0
        self.omega_sim = 0
        self.ccode_time = 0
        self.torques = self._get_readings(0)

    def _torque_to_volt(self, torque, subtr_midpoint=True):
        #return ((torque / self.prms['torque_to_current']) - self.prms[
        #    'dac_to_current_intercept']) / self.prms['dac_to_current_gradient']
        if subtr_midpoint:
            return 1122334456 * torque - 7.12345679
        else:
            return 1122334456 * torque

    def _volt_to_torque(self, volt):
        #return ((volt) * self.prms['dac_to_current_gradient'] +
        #        self.prms['dac_to_current_intercept']) * \
        #       self.prms['torque_to_current']
        return ((volt - 2047) + 7.12345679) / 1122334456.

    def _update_torque(self, input_theta):
        """Get the updated torque from the C script given the angular 
        displacement in radians."""

        # convert to microns and then find closest match to a value from the
        # proximity sensor. todo why is the calibration curve for the sensor
        # todo increasing in signal as the displacement increases?
        input_theta /= self.prms['rad_to_0.1um']
        idx = (np.abs(self.prms['sensor_to_disp'] - input_theta)).argmin()\
            + 1558

        # note that idx is the proximity sensor reading, which should be fed
        # into the C code.
        self.digitised_torque = talk.get_torque(int(idx))
        # also get the digitised theta and d(theta)/dt values.
        self.digitised_theta_sim = talk.get_mu()
        self.digitised_omega_sim = talk.get_dmudt()

        self.total_torque = self._volt_to_torque(self.digitised_torque)
        self.theta_sim = self.digitised_theta_sim * self.prms['rad_to_0.1um']
        self.omega_sim = self.digitised_omega_sim * self.prms['rad_to_0.1um']\
            / self.dt
        # todo check the above formulae.

        self.ccode_time = talk.get_point_in_cycle()
        print("Torque now updated to {}".format(self.digitised_torque))

    def _get_recent_torque(self, current_time):
        """Return the last calculated torque."""

        current_reading = self._get_readings(current_time)
        print("Current torques reading", current_reading)
        self.torques = np.vstack((self.torques, current_reading))

        return self.total_torque

    def _get_readings(self, current_time):
        current_reading = np.array(
            [self.ccode_time, current_time, self.digitised_torque,
             self.total_torque, self.digitised_theta_sim, self.theta_sim,
             self.digitised_omega_sim, self.omega_sim])
        for i in range(len(current_reading)):
            try:
                assert len(current_reading[i]) == 1
                current_reading[i] = current_reading[i][0]
            except TypeError:
                pass
        return np.array(current_reading)

    def main_operation(self):
        """Run the ODE integrator for the system in question."""
        # Set parameters.
        i = self.prms['i']
        b = self.prms['b']
        k = self.prms['k']
        y0 = np.array([self.prms['theta_0'], self.prms['omega_0']]).squeeze()
        t0 = self.prms['t0']
        t_fin = self.prms['tfin']

        r = ode(c.f_full_torque)#.set_integrator('dop853')
        self._update_torque(y0[0])
        r.set_initial_value(y0, t0).set_f_params(
            i, b, k, self._get_recent_torque).set_jac_params(i, b, k)

        results = [[*t0, *y0]]

        while r.successful() and r.t < t_fin:
            y = np.real(r.integrate(r.t + self.display_dt))
            data_point = [*(r.t + self.display_dt), *y]
            results.append(data_point)
            print("Time-theta-omega", data_point)
            # Recalculate the reset the torque every dt seconds.

            # get the last set of consecutive points where the digitised
            # torque (-6th column) has the same value as the current one
            # every cycle. If the corresponding times have a range greater
            # than or equal to dt, re-measure the torque.
            matching_indices = h.find_consec_indices(self.torques[:, -6])
            if self.torques[-1, 1] - min(self.torques[matching_indices,
                                                      1]) >= self.dt:
                self._update_torque(y[0])
                print("triggered")
                r.set_initial_value(r.y, r.t).set_f_params(
                    i, b, k, self._get_recent_torque)

        results = np.array(results).squeeze()
        sines_torque = h.baker(t.calculate_sine_pi,
                               ["", "", "", "", self.prms['g_0_mag'],
                                self.prms['w_d'], np.array([0])],
                               pos_to_pass_through=(0, 3))

        theory = t.calc_theory_soln(
            np.linspace(0,2,1000), t0[0], y0, (b - self.prms['b\''])[0],
            (k - self.prms['k\''])[0], i[0], sines_torque)
        print("Init parameters: dt: {}, display_dt: {}, b: {}, b': {}, k: {}, "
              "k': {}, I: {}, y0: {}, t0: {}, tfin: {}, g0: {}, w_d: {}".format(
            self.dt, self.display_dt, b, self.prms['b\''], k, self.prms[
                'k\''], i, y0, t0, t_fin, self.prms['g_0_mag'], self.prms[
                'w_d']))
        print("Parameters from the C code: k': {}, b': {}, g0: {}".format(
            talk.get_k_prime(), talk.get_b_prime(), talk.get_amp()))

        plt.plot(theory[:, 0], theory[:, 1])
        plt.plot(results[:, 0], results[:, 1])
        plt.show()
        #exp_results = pd.DataFrame(np.array(results).squeeze(),
        #                           columns=['t', 'theta', 'omega'])
        #print("Integration complete. Summary of results: Time-theta-omega and "
        #      "torques-summary")
        #print(exp_results)
        # print(np.array(self.torques))
        #print(pd.DataFrame(np.array(self.torques), columns=[
        #    't', 'total-torque', 'theta-sim', 'omega-sim']))
        #return {'displacements': exp_results}


class NRRegimesPython(Experiment):
    """Tests the theoretical against the ODE integrator function in the time 
    domain for one set of control parameters, allowing for variable noise and 
    delays to be introduced at any stage. Uses only Python."""

    def __init__(self, config=None):
        """Create the variables to perform the ODE numerical integration with 
        feedback."""
        super(NRRegimesPython, self).__init__(config=config)
        # set initial parameters
        self.i = self.prms['i']
        self.b = self.prms['b']
        self.k = self.prms['k']
        self.k_prime = self.prms['k\'']
        self.b_prime = self.prms['b\'']
        self.y0 = np.array([self.prms['theta_0'],
                            self.prms['omega_0']]).squeeze()
        self.t0 = h.convert_to_array(self.prms['t0'])
        self.t_fin = self.prms['tfin']
        self.g_0 = self.prms['g_0_mag']
        self.divider = self.prms['max_step_divider']

        # Get the driving frequency and hence the sampling rate.
        self.w_d = self.prms['w_d']
        self.dt = 2 * np.pi / (self.w_d * 120)
        self.phi = self.prms['phi']

        # Create an array of the sine values for the analytic torque.
        self.torque_sine = self.g_0 * np.sin(self.w_d * np.arange(
            self.t0, 2 * np.pi / self.w_d, self.dt) + self.phi)

        # set initial parameters. For 1st run, set them equal to the actual
        # initial conditions to avoid later errors in calculation.
        self.total_torque = self.torque_sine[0]     # Only analytic torque here.
        self.theta_sim = self.y0[0]
        self.omega_sim = self.y0[1]
        self.torques = np.array([[*self.t0, self.total_torque, self.theta_sim,
                                 self.omega_sim]])

    def _update_torque(self, input_theta, i):
        """Get the updated torque given the angular displacement in radians."""
        last_theta_sim = self.torques[-1, 2]

        # theta_sim value in torques array
        self.theta_sim = input_theta  # todo add noise and delays here
        self.omega_sim = (self.theta_sim - last_theta_sim) / self.dt
        self.total_torque = self.torque_sine[i % len(self.torque_sine)] + \
            self.k_prime * self.theta_sim + self.b_prime * self.omega_sim
        return self.total_torque

    def _get_recent_torque(self, current_time):
        """Return the last calculated torque."""
        current_reading = self._get_readings(current_time)
        self.torques = np.vstack((self.torques, current_reading))
        return self.total_torque

    def _get_readings(self, current_time):
        """Get a single value of torque, theta_sim, and omega_sim, given the 
        current time."""
        current_reading = np.array(
            [current_time, self.total_torque, self.theta_sim, self.omega_sim])
        for i in range(len(current_reading)):
            try:
                assert len(current_reading[i]) == 1
                current_reading[i] = current_reading[i][0]
            except TypeError:
                pass
        return np.array(current_reading)

    def main_operation(self, plot=True):
        """Run the ODE integrator for the system in question and save the 
        plots."""
        rd = ode(c.f_full_torque).set_integrator(
            'vode', max_step=self.dt / self.divider)

        torque_index = 0
        self._update_torque(self.y0[0], torque_index)
        rd.set_initial_value(self.y0, self.t0).set_f_params(
            self.i, self.b, self.k, self._get_recent_torque).set_jac_params(
            self.i, self.b, self.k)

        results = [[*self.t0, *self.y0]]
        sim_result_compare = []
        while rd.successful() and rd.t < self.t_fin:
            t_now = rd.t + self.dt
            y = np.real(rd.integrate(t_now))
            data_point = [*t_now, *y]
            results.append(data_point)

            # Get the last set of consecutive points where the torque has the
            # same value as the current one every cycle. If the corresponding
            # times have a range greater than or equal to dt, re-measure the
            # torque.
            match_indices = h.find_consec_indices(self.torques[:, 1])
            if self.torques[-1, 0] - min(self.torques[match_indices, 0]) \
                    >= self.dt:
                torque_index += 1
                self._update_torque(y[0], torque_index)
                self._get_recent_torque(t_now)
                sim_result_compare.append([*data_point, self.torques[-1, -2],
                                           self.torques[-1, -1]])
                rd.set_initial_value(y, t_now).set_f_params(
                    self.i, self.b, self.k, self._get_recent_torque)

        results = np.array(results).squeeze()
        sim_result_compare = np.array(sim_result_compare).squeeze()

        if plot:
            # Calculate theoretical results.
            sines_torque = h.baker(t.calculate_sine_pi,
                                   ["", "", "", "", self.g_0, self.w_d,
                                    self.phi], pos_to_pass_through=(0, 3))
            theory = t.calc_theory_soln(
                results[:, 0], self.t0, self.y0, self.b - self.b_prime,
                self.k - self.k_prime, self.i, sines_torque)
            print(
                "Init parameters: dt: {}, b: {}, b': {}, k: {}, k': {}, I: {}, "
                "y0: {}, t0: {}, tfin: {}, g0: {}, w_d: {}".format(
                    self.dt, self.b, self.b_prime, self.k, self.k_prime, self.i,
                    self.y0, self.t0, self.t_fin, self.g_0, self.w_d))

            # Find absolute errors and plot.
            iterator_diffs = m.calc_norm_errs(
                [results[:, 1], theory[:, 1]], [results[:, 2], theory[:, 2]])[1]
            simulated_diffs = m.calc_norm_errs(
                [sim_result_compare[:, 3], sim_result_compare[:, 1]],
                [sim_result_compare[:, 4], sim_result_compare[:, 2]])[1]

            real_space_data = \
                [[[[theory[:, 0], theory[:, 1]], [results[:, 0], results[:, 1]],
                   [self.torques[:, 0], self.torques[:, 2]]],
                  [[results[:, 0], iterator_diffs[0]],
                   [sim_result_compare[:, 0], simulated_diffs[0]]]],
                 [[[theory[:, 0], theory[:, 2]], [results[:, 0], results[:, 2]],
                   [self.torques[:, 0], self.torques[:, 3]]],
                  [[results[:, 0], iterator_diffs[1]],
                   [sim_result_compare[:, 0], simulated_diffs[1]]]]]

            p.two_by_n_plotter(
                real_space_data, self.filename, self.prms,
                savepath=self.plotpath, show=True, x_axes_labels=['t/s', 't/s'],
                y_top_labels=[r'$\theta$/rad', r'$\dot{\theta}$/rad/s'],
                y_bottom_labels=[r'$\Delta\theta$/rad',
                                 r'$\Delta\dot{\theta}$/rad/s'])

        exp_results = pd.DataFrame(results, columns=['t', 'theta', 'omega'])
        torques = pd.DataFrame(self.torques, columns=[
            't', 'total torque', 'theta_sim', 'omega_sim'])
        return {'displacements': exp_results, 'measured-vals': torques}


class FixedStepIntegrator(Experiment):
    """Tests the theoretical against the manually written RK4 fixed step ODE 
    integrator function in the time domain for one set of control parameters, 
    allowing for variable noise and delays to be introduced at any stage."""

    def __init__(self, config=None):
        """Create the variables to perform the ODE numerical integration with 
        feedback."""
        super(FixedStepIntegrator, self).__init__(config=config)

        # set initial parameters
        self.i = self.prms['i']
        self.b = self.prms['b']
        self.k = self.prms['k']
        self.k_prime = self.prms['k\'']
        self.b_prime = self.prms['b\'']
        self.y0 = np.array([self.prms['theta_0'],
                            self.prms['omega_0']]).squeeze()
        self.t0 = h.convert_to_array(self.prms['t0'])
        self.t_fin = self.prms['tfin']
        self.g_0 = self.prms['g_0_mag']
        self.divider = self.prms['max_step_divider']
        self.period_divider = self.prms['sampling_divider']
        self.delay = self.prms['delay']

        # Get the driving frequency and hence the sampling rate.
        self.w_d = self.prms['w_d']
        self.dt = 2 * np.pi / (self.w_d * self.period_divider)
        self.phi = self.prms['phi']

        # Create an array of the sine values for the analytic torque.
        self.torque_sine = self.g_0 * np.sin(self.w_d * np.arange(
            self.t0, 2 * np.pi / self.w_d + self.t0, self.dt) + self.phi)

        # set initial parameters. For 1st run, set them equal to the actual
        # initial conditions to avoid later errors in calculation.
        self.total_torque = self.torque_sine[0]  # Only analytic torque here.
        self.theta_sim = self.y0[0]
        self.omega_sim = self.y0[1]
        self.torques = np.array([[*self.t0, self.total_torque, self.theta_sim,
                                  self.omega_sim]])

    def _update_torque(self, input_theta, i):
        """Get the updated torque given the angular displacement in radians."""
        last_theta_sim = self.torques[-1, 2]

        # theta_sim value in torques array
        self.theta_sim = input_theta  # todo add noise and delays here
        self.omega_sim = (self.theta_sim - last_theta_sim) / self.dt
        self.total_torque = self.torque_sine[i] + \
            self.k_prime * self.theta_sim + self.b_prime * self.omega_sim

    def _get_recent_torque(self, current_time):
        """Return the last calculated torque."""
        current_reading = self._get_readings(current_time)
        self.torques = np.vstack((self.torques, current_reading))
        return self.total_torque

    def _get_readings(self, current_time):
        """Get a single value of torque, theta_sim, and omega_sim, given the 
        current time."""
        current_reading = np.array(
            [current_time, self.torque_plus_noise(), self.theta_sim,
             self.omega_sim])
        for i in range(len(current_reading)):
            try:
                assert len(current_reading[i]) == 1
                current_reading[i] = current_reading[i][0]
            except TypeError:
                pass
        return np.array(current_reading)

    def torque_plus_noise(self, noise=0):
        """The noise parameter is either 0 or a baked noise function to 
        generate random noise on each measurement."""
        return self.total_torque + noise

    def main_operation(self, plot=True):
        """Run the ODE integrator for the system in question and save the 
        plots."""
        torque_index = 0
        self._update_torque(self.y0[0], torque_index)

        f_baked = h.baker(c.f_full_torque,
                          ["", "", self.i, self.b, self.k,
                           self._get_recent_torque], pos_to_pass_through=(0, 1))
        # Set initial conditions
        dy = c.rk4(f_baked)
        times, y, stepsize = *self.t0, self.y0, self.dt[0] / self.divider[0]

        results = []
        update_counter = 0
        while times <= self.t_fin:
            results.append([times, *y, self.total_torque, self.theta_sim,
                            self.omega_sim])
            times, y = times + stepsize, y + dy(times, y, stepsize)
            update_counter += 1
            # For a fixed step size, we can just repeat the torque
            # recalculation every self.divider steps, so there is no
            # overshooting.
            if update_counter == self.divider:
                update_counter = 0
                torque_index += 1
                if torque_index == self.period_divider:
                    torque_index = 0
                self._update_torque(y[0], torque_index)

        results = np.array(results).squeeze()
        if plot:
            times = m.check_nyquist(results[:, 0], self.w_d, self.b,
                                    self.b_prime, self.k, self.k_prime, self.i)
            decimal_periods = np.abs(5.231 * 2 * np.pi / self.w_d)
            time_index = 0
            num = 1
            # The index for 5.231 periods of time being elapsed is used to
            # calculate the number per segment to use for the auto-correlation.
            while time_index < 10:
                time_index = np.abs(times - num * decimal_periods).argmin()
                num += 1

            if self.b - self.b_prime > 0:
                # Will only reach steady state if this is the case, otherwise no
                # point making a response curve. Measure one point. b - b' = 0
                # has two steady state frequencies, the transient and PI.
                ss_times = m.enter_ss_times(results[:, 0], results[:, 1])

                if ss_times is not False:
                    frq, fft_theta = m.calc_fft(
                        results[:, 0][(results[:, 0] >= ss_times[0]) *
                                      (results[:, 0] <= ss_times[1])],
                        results[:, 1][(results[:, 0] >= ss_times[0]) *
                                      (results[:, 0] <= ss_times[1])])
                    # for low frequencies, the length of time of the signal must
                    # also be sufficiently wrong for the peak position to be
                    # measured properly.

                    # Half-amplitude of peak used to calculate bandwidth.
                    freq = m.calc_freqs(np.absolute(fft_theta), frq, n_peaks=1)
                else:
                    raise Exception('Panic 2.')

            # Calculate fourier series first 1000 terms.
            ind_mag_phi = fourier.get_fourier_series(1000, self.w_d, self.dt,
                                                     self.g_0)
            f_torque = fourier.get_torque(self.torques[:, 0], ind_mag_phi[:, 0], 
                                          ind_mag_phi[:, 1], ind_mag_phi[:, 2], 
                                          self.w_d)

            # Calculate theoretical vs actual torque and plot.
            #plt.plot(self.torques[:, 0], f_torque, label='Fourier-calculated')
            #plt.plot(self.torques[:, 0], self.torques[:, 1], label='Measured')
            #plt.tick_params(direction='out')
            #plt.grid(True)
            #plt.xlabel('t/s', fontweight='bold', fontsize=13)
            #plt.ylabel('$G_{tot}$/Nm', fontweight='bold', fontsize=13)
            #plt.show()

            # Calculate theoretical results.
            torque_sine = h.baker(t.calculate_sine_pi,
                                  ["", "", "", "", ind_mag_phi[:, 1],
                                   self.w_d * ind_mag_phi[:, 0],
                                   ind_mag_phi[:, 2]],
                                  pos_to_pass_through=(0, 3))
            theory = t.calc_theory_soln(
                results[:, 0], self.t0, self.y0, self.b - self.b_prime,
                self.k - self.k_prime, self.i, torque_sine)

            # fourier theoretical results.
            torque_fourier = h.baker(t.calculate_sine_pi,
                                     ["", "", "", "", ind_mag_phi[:, 1],
                                      self.w_d * ind_mag_phi[:, 0],
                                      ind_mag_phi[:, 2]],
                                     pos_to_pass_through=(0, 3))
            theory_fourier = t.calc_theory_soln(
                results[:, 0], self.t0, self.y0, self.b - self.b_prime,
                self.k - self.k_prime, self.i, torque_fourier)
            print(
                "Init parameters: dt: {}, b: {}, b': {}, k: {}, k': {}, I: {}, "
                "y0: {}, t0: {}, tfin: {}, g0: {}, w_d: {}".format(
                    self.dt, self.b, self.b_prime, self.k, self.k_prime,
                    self.i, self.y0, self.t0, self.t_fin, self.g_0, self.w_d))

            # Find absolute errors and plot.
            iterator_diffs = m.calc_norm_errs(
                [results[:, 1], theory[:, 1]], [results[:, 2], theory[:, 2]])[1]
            simu_diffs = m.calc_norm_errs([results[:, 4], results[:, 1]],
                                          [results[:, 5], results[:, 2]])[1]
            real_space_data = \
                [[[[theory[:, 0], theory[:, 1]],
                   [theory_fourier[:, 0], theory_fourier[:, 1]],
                   [results[:, 0], results[:, 1]],
                   [results[:, 0], results[:, 4]]],
                  [[results[:, 0], iterator_diffs[0]],
                   [results[:, 0], simu_diffs[0]]]],
                 [[[theory[:, 0], theory[:, 2]],
                   [theory_fourier[:, 0], theory_fourier[:, 2]],
                   [results[:, 0], results[:, 2]],
                   [results[:, 0], results[:, 5]]],
                  [[results[:, 0], iterator_diffs[1]],
                   [results[:, 0], simu_diffs[1]]]]]

            p.two_by_n_plotter(
                real_space_data, self.filename, self.prms, tag='fourier-added',
                savepath=self.plotpath, show=True, x_axes_labels=['t/s', 't/s'],
                y_top_labels=[r'$\theta$/rad', r'$\dot{\theta}$/rad/s'],
                y_bottom_labels=[r'$\Delta\theta$/rad',
                                 r'$\Delta\dot{\theta}$/rad/s'])

        exp_results = pd.DataFrame(
            results, columns=['t', 'theta', 'omega', 'total-torque',
                              'theta-sim', 'omega-sim'])
        return {'all-mmts': exp_results}


class ReadAllData(Experiment):
    """Read all data from a certain dataset and produce the real and fourier 
    space plots."""

    def __init__(self, config=None):
        super(ReadAllData, self).__init__(config=config)

    def main_operation(self, plot=True):
        """Read all data from a specific folder and produce real space and 
        fourier space plots. Use for only one set of b, k, b', k', i, 
        etc. parameters at a time."""
        directory = self.prms['directory'][0]
        all_same = self.prms['all_same'][0]
        file_list = h.find_files(directory)
        filename_roots = r.check_matches(file_list, num_one='all-mmts',
                                         num_two=None)

        all_data = r.read_all_data(directory, filename_roots,
                                   disps_ext='all-mmts', torque_ext='all-mmts')
        sorted_by_params = r.sort_data(all_data, all_same=all_same)

        b = sorted_by_params[0][0]['b']
        k = sorted_by_params[0][0]['k']
        b_prime = sorted_by_params[0][0]['b\'']
        k_prime = sorted_by_params[0][0]['k\'']
        i = sorted_by_params[0][0]['i']

        # Baked theoretical response function to require wd input only,
        # which depends on the range measured.
        need_wd = h.baker(t.theory_response, args=[b, k, i, b_prime, k_prime,
                                                   ''], pos_to_pass_through=5)
        fft_data = r.match_torques(sorted_by_params, plot_real=self.prms[
            'plot_real'][0], savepath=directory + 'plots/')
        print(fft_data)
        if plot:
            r.prepare_to_plot(fft_data, need_wd, savepath=directory + 'plots/')
        
    def run(self, tags=False, savedata=False, plot=True):
        super(ReadAllData, self).run(tags=tags, savedata=savedata, plot=plot)
