"""Read data from each CSV file, find the appropriate log file, and group data 
into sets of the same parameters but varying driving frequency. Once this is 
done, calculate the response curve for each set."""

import re
from itertools import groupby
from operator import itemgetter
from time import time

import numpy as np
import pandas as pd

import helpers as h
import measurement as m
import plotter as p


def check_matches(files_list, num_one=r'displacements',
                  num_two='measured-vals.csv'):
    """Given a list of file names as strings, finds the file names containing 
    the string num_one at the end, and checks that equivalent files containing 
    num_two exist in the list. Returns the list of the filename roots (the 
    start of the file name) where this condition is met."""
    fname_roots = []
    len_num_one = len(num_one) + 4  # add 4 for csv extension
    for each_file in files_list:
        if re.search(re.compile(num_one), each_file):
            if num_two is None:
                fname_roots.append(each_file[:-len_num_one])
            elif each_file[:-len_num_one] + num_two in files_list:
                fname_roots.append(each_file[:-len_num_one])
    return fname_roots


def prepare_to_plot(grouped_mmts, theory_resp, savepath=None, show=True):
    """For this particular arrangement of measurements, reformat the grouped 
    data to be plotted.
    :param grouped_mmts: The measurements of frequency, amplitude and phase 
    returned by measure_all_groups.
    :param theory_resp: Theory's response function baked to require just the 
    driving angular frequency.
    :param savepath: The path to save the graphs to.
    :param show: Whether to show each graph or not."""
    mmts = [[[], []]]

    for parameter_set in grouped_mmts:
        to_plot = np.array(parameter_set[-1])
        # Convert to angular frequencies.
        simulated_ang_freqs = to_plot[:, 0, 0, :] * 2 * np.pi
        # Sort by frequencies.
        sort_args = simulated_ang_freqs[:, 0].argsort()
        simulated_ang_freqs = simulated_ang_freqs[sort_args].squeeze()
        simulated_amps = to_plot[:, 0, 1, :][sort_args].squeeze()
        simulated_phase = to_plot[:, 0, 2, :][sort_args].squeeze()

        #measured_freqs = to_plot[:, 1, 0, :]
        #measured_amps = to_plot[:, 1, 1, :]
        #measured_phase = to_plot[:, 1, 2, :]

        mmts[0][0][0].append([simulated_ang_freqs, simulated_amps,
                              r'$k={}, k\'={}, b={}, b\'={}$'.format(
                                  parameter_set[2], parameter_set[3],
                                  parameter_set[0], parameter_set[1]), ''])
        mmts[0][0][1].append([simulated_ang_freqs, simulated_phase,
                              r'$k={}, k\'={}, b={}, b\'={}$'.format(
                                  parameter_set[2], parameter_set[3],
                                  parameter_set[0], parameter_set[1]), ''])

    # Generate evenly spaced angular frequencies.
    even_spaced_wd = np.linspace(10, 140, 5000)
    mmts[0][0][0].append([even_spaced_wd, np.absolute(theory_resp(
        even_spaced_wd)), r'Theoretical'])
    mmts[0][0][1].append([even_spaced_wd, np.angle(theory_resp(even_spaced_wd)),
                         r'Theoretical'])
    p.two_by_n_plotter(
        mmts, '', {'i': grouped_mmts[0][4], 'g_0_mag': grouped_mmts[0][5],
                   'phi': grouped_mmts[0][6], 't0': grouped_mmts[0][7],
                   'tfin': grouped_mmts[0][8], 'theta_0': grouped_mmts[0][9],
                   'omega_0': grouped_mmts[0][10]},
        savepath=savepath, show=show, x_axes_labels=['$\omega$/rad/s'],
        tag='response-curve-{}'.format(h.time_for_name()),
        y_top_labels=[r'$\left|R(\omega)\right|$/rad/(Nm)'],
        y_bottom_labels=[r'$\phi(R(\omega))$/rad'])


def read_all_data(path, fname_roots, disps_ext=r'displacements',
                  torque_ext=r'measured-vals'):
    """Read all data for a list of filename roots, as well as the parameters 
    used for them. Return a list of dictionaries containing the parameters, 
    as well as the time-theta data under the 'disps' key, and the torque data 
    in the 'torques' key.
    :param path: String for the directory to look in.
    :param fname_roots: A list of filename roots.
    :param disps_ext: Displacements file extension.
    :param torque_ext: Torques file extension. Set to same as above if 
    reading from the same file."""

    outputs = []
    for root in fname_roots:
        param_dict = _get_config(path, root)
        if disps_ext == torque_ext:
            df = pd.read_csv('{}{}{}.csv'.format(path, root, disps_ext),
                             index_col=0)
            param_dict['disps'] = df[['t', 'theta', 'omega']]
            param_dict['torques'] = df[['t', 'total-torque', 'theta-sim',
                                        'omega-sim']]
        else:
            param_dict['disps'] = pd.read_csv('{}{}{}.csv'.format(
                path, root, disps_ext), index_col=0)
            param_dict['torques'] = pd.read_csv('{}{}{}.csv'.format(
                path, root, torque_ext), index_col=0)
        outputs.append(param_dict)
    return outputs


def sort_data(all_datasets, all_same=False,
              sort_by=('b', 'phi', "b'", 't0', 'k', 'omega_0', 'tfin', 'i',
                       "k'", 'g_0_mag', 'theta_0')):
    """Sort all data by grouping according to parameter values. Return a list of 
    lists of dictionaries. If all_same is True, just returns the entire 
    thing as if all parameters belong to one set."""
    if all_same:
        return [all_datasets]
    else:
        grouper = itemgetter(*sort_by)
        sets = []
        for group in groupby(sorted(all_datasets, key=grouper), key=grouper):
            same_parameters = []
            for element in group[1]:
                same_parameters.append(element)
            sets.append(same_parameters)
        return sets


def match_torques(grouped_sets, plot_real=False, savepath=None):
    """Match the times to the torques and displacements and return an array 
    of those values for each of the parameter combinations.
    :param grouped_sets: All grouped data.
    :param plot_real: Plot the real space data and save with logs.
    :param savepath: Save path to send to the n_plotter function."""

    fft_mmts = []
    for group in grouped_sets:
        # a list of dictionaries with the same parameter values except for w_d.
        b = group[0]['b']
        b_prime = group[0]['b\'']
        k = group[0]['k']
        k_prime = group[0]['k\'']
        i = group[0]['i']
        g_0_mag = group[0]['g_0_mag']
        phi = group[0]['phi']
        t0 = group[0]['t0'].squeeze()
        tfin = group[0]['tfin']
        w_d = group[0]['w_d']
        theta_0 = group[0]['theta_0']
        omega_0 = group[0]['omega_0']
        one_group = [b, b_prime, k, k_prime, i, g_0_mag, phi, t0, tfin, theta_0,
                     omega_0]
        one_dataset = []
        for dataset in group:
            # one dictionary with real space data and torque values.
            disps = dataset['disps']
            torques = dataset['torques']
            analytic_torque = torques['total-torque'] - k_prime * torques[
                'theta-sim'] - b_prime * torques['omega-sim']
            analytic, theta_sim, omega_sim = [], [], []

            for t in disps['t']:
                # match up the real and torque data times.
                idx = (np.abs(torques['t'] - t)).argmin()
                analytic.append(analytic_torque[idx])
                theta_sim.append(torques['theta-sim'][idx])
                omega_sim.append(torques['omega-sim'][idx])
            output = np.vstack((disps.as_matrix().T, analytic, theta_sim,
                                omega_sim)).T

            if plot_real:
                expected_torque = g_0_mag * np.sin(w_d * torques['t'] + phi)

                # Get the number of segments to use per correlation
                # calculation - equal to one period of the expected torque.
                period = 2 * np.pi / w_d
                match_one_period = np.abs(torques['t'] - period)
                index_of_period = match_one_period.argmin()
                if match_one_period[index_of_period] < 0:
                    index_of_period += 1

                real_space = \
                    [
                        [
                            [
                                [output[:, 0], output[:, 1]]
                            ],
                            [
                                [torques['t'], expected_torque],
                                [torques['t'], analytic_torque]
                            ]
                        ]
                    ]

                p.two_by_n_plotter(
                    real_space, 't-torque',
                    {'b': b, 'b\'': b_prime, 'k': k, 'k\'': k_prime, 'i': i,
                     'g_0_mag': g_0_mag, 'phi': phi, 't0': t0, 'tfin': tfin,
                     'theta_0': theta_0, 'omega_0': omega_0}, show=True,
                    savepath=savepath, x_axes_labels=['t/s'],
                    tag='{}'.format(time()), y_top_labels=[r'$\theta$/rad'],
                    y_bottom_labels=[r'$G_{s}(t)$/Nm'])

            # Times have now been matched and we are ready to obtain frequency,
            # amplitude and phase values from the output data.
            # Do once for simulated values and compare to measured values.
            measure_for = [[output[:, 1], output[:, 2]]]#, only plot actual ones
                           #[output[:, 4], output[:, 5]]]
            mmts = []
            for measure in measure_for:
                # Normalise the theta by the analytic torque amplitude to get
                # the response function.
                mmts.append(m.one_mmt_set(
                    output[:, 0], measure[0] / g_0_mag, output[:, 3], b,
                    b_prime, k, k_prime, i))
            one_dataset.append(mmts)
            one_group.append(one_dataset)
        fft_mmts.append(one_group)
    return fft_mmts


def _get_config(path, filename_root):
    """Given a filename root and a path, looks for log files corresponding to 
    each file name in the path/logs/ folder. Extracts the parameters used, 
    as recorded in the log."""
    log_files = h.find_files(path + 'logs/', check_type='.txt')
    root = filename_root
    look_for = root + 'log.txt'
    assert look_for in log_files
    with open('{}logs/{}'.format(path, look_for), 'r') as f:
        log_contents = f.read()
        config_string = re.findall('(?<={)[^}]*(?=})', log_contents)
        assert len(config_string) == 1
        config_string = config_string[0].replace('array', 'np.array')
        configs = config_string.split(',')
        config_dict = {}
        for key_string in configs:
            if 'dtype' not in key_string and 'noise_type' not in key_string:
                key = eval(key_string.split(':')[0])
                value = eval(key_string.split(':')[1])
                config_dict[key] = value
            else:
                pass
        # NOTE this will split the string into individual keys as long as the
        # numpy arrays only have 1 element each!
    return config_dict
