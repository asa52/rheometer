"""Functions to generate noise and delays to introduce imperfections into the 
pendulum."""

import numpy as np
import scipy.constants as const

import helpers as h


def johnson_nyquist(temp, resistance, mmt_time):
    """Johnson-Nyquist thermal noise, returned in units of torque."""
    current_noise = np.sqrt(4 * const.k * temp / (mmt_time * resistance))
    approx_torque_ratio = h.order_of_mag(measure_once[0][0])
    return approx_torque_ratio * np.random.normal(0, current_noise)


def shot(current, mmt_time):
    """Shot noise in terms of torque."""
    current_noise_var = 2 * const.e * current / mmt_time
    approx_torque_ratio = h.order_of_mag(measure_once[0][0])
    return approx_torque_ratio * np.random.poisson(current_noise_var)


def single_freq(mag, time, freq=50):
    """Single frequency noise of a specified current magnitude, returned in 
    terms of torque."""
    ang_freq = freq * 2 * np.pi
    approx_torque_ratio = h.order_of_mag(measure_once[0][0])
    return approx_torque_ratio * mag * np.sin(ang_freq * time)


def one_over_f(mag, dc_mag):
    """1/f noise in terms of torque. Magnitude of noise decreases with 
    increasing frequency (of the current)."""
    # TODO check this form - is it Gaussian, etc?
    offset = (mag / dc_mag) ** 2
    approx_torque_ratio = h.order_of_mag(measure_once[0][0])
    freq = np.linspace(0.01, 1000, 1000)
    current_noise = mag / np.sqrt(freq) + offset    # goes as 1/f for power.
    return approx_torque_ratio * np.sum(np.random.normal(
        loc=0, scale=current_noise))


def get_measured_vals():
    """Get the values measured for the rheometer previously, when experiment
    done in water."""
    andreas_mmts = h.yaml_read('../configs/measured_vals.yaml')
    torque_current_ratio = np.array([
        andreas_mmts['torque_to_current'],
        andreas_mmts['torque_to_current_err']]).squeeze() * 10**6
    theta_current_ratio = np.array([
        andreas_mmts['rad_to_current'],
        andreas_mmts['rad_to_current_err']]).squeeze() * 10**6
    system_k = np.array([
        andreas_mmts['k_for_system'],
        andreas_mmts['k_for_system_err']]).squeeze()
    system_i = np.array([
        andreas_mmts['i_for_system'],
        andreas_mmts['i_for_system_err']]).squeeze()
    return torque_current_ratio, theta_current_ratio, system_k, system_i


def get_old_theta(current_time, delay, disp_array):
    """Return the value of theta_sim in the torque array at the time just 
    before (or exactly equal to) current_time - delay, as well as the value 
    just before that.
    :param current_time: The current value of time.
    :param delay: The delay in seconds of the measuring instrument.
    :param disp_array: The array/list in the format:
    [[time-vals-column], [theta-column], ...]."""
    disp_array = h.convert_to_array(disp_array)
    old_time = current_time - delay
    old_enough = disp_array[np.where(disp_array[:, 0] <= old_time)]
    closest = (np.abs(old_enough[:, 0] - old_time)).argmin()
    return old_enough[closest, 1]


measure_once = get_measured_vals()
