"""Attempt the same conditions with this handmade RK4 solver to compare the 
scipy.integrate.ode vode integrator to. This has a fixed step size so just 
integrate after a certain number of loops with no if statement."""

import matplotlib.pyplot as plt
import numpy as np

import conditional_pend as c
import helpers as h
import theory as t


def RK4(f):
    return lambda t, y, dt: (
        lambda dy1: (
            lambda dy2: (
                lambda dy3: (
                    lambda dy4: (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6
                )(dt * f(t + dt, y + dy3))
            )(dt * f(t + dt / 2, y + dy2 / 2))
        )(dt * f(t + dt / 2, y + dy1 / 2))
    )(dt * f(t, y))

times = np.arange(0, 5, 0.0001277)
sines_torque = h.baker(
    t.calculate_sine_pi, ["", "", "", "", [1.e-7], [41], [0]],
    pos_to_pass_through=(0, 3))
theory = t.calc_theory_soln(times, [0], np.array([0., 0.]), 5.e-7, 1.e-3,
                            5.5e-7, sines_torque)
baked_torque = h.baker(c.analytic_torque, ["", [41.], [1.e-7], [0]],
                       pos_to_pass_through=0)
f_baked = h.baker(c.f_full_torque, ["", "", 5.5e-7, 5.e-7, 1.e-3,
                                    baked_torque], pos_to_pass_through=(0, 1))

dy = RK4(f_baked)

t, y, dt = 0., np.array([0, 0]), 0.0001277
results = []
while t <= 5.:
    if abs(round(t) - t) < 1e-5:
        results.append([t, *y])
    t, y = t + dt, y + dy(t, y, dt)
results = np.array(results)

plt.plot(results[:, 0], results[:, 1], ':')
plt.plot(theory[:, 0], theory[:, 1], ':')
plt.show()