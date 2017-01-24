"""Solves the ODE for a forced, damped harmonic oscillator with an arbitrary
forcing function."""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# The second order differential equation for the angle theta of a
# torsional pendulum acted on by elastic tension with friction can be written:
# TODO change this to include all the geometry of the pendulum
# theta''(t) + b*theta'(t) + c*theta(t) = F(t)

# where b and c are positive constants, and a prime (') denotes a derivative.
# To solve this equation with odeint, we must first convert it to a system of
# first order equations. By defining the angular velocity omega(t) = theta'(t),
# we obtain the system:
# theta'(t) = omega(t)
# omega'(t) = F(t) - b*omega(t) - c*theta(t)

# SOURCE: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/
# scipy.integrate.odeint.html#scipy.integrate.odeint


def forced_pend(y, t, b, c, f_func):
    """Forced torsional pendulum oscillator.
    :param y: List of [theta, theta'].
    :param t: Array of times to solve with.
    :param b: Coefficient of theta' on LHS of ODE.
    :param c: Same for theta.
    :param f_func: Function object for the forcing function at time t.
    :return: dy/dt."""
    theta, omega = y
    dydt = [omega, f_func(t) - b * omega - c * theta]
    return dydt


def _bc_consts(beta, w_dr):
    C = -8*beta*w_dr/((25-w_dr**2)**2+64*w_dr**2)
    D = beta*(25-w_dr**2)/((25-w_dr**2)**2+64*w_dr**2)
    A = 1-C
    B = (4*A - D*w_dr)/3
    return A, B, C, D


def theta_theory(b, c, w_dr, beta, t):
    """Theoretical values of theta."""
    A, B, C, D = _bc_consts(beta, w_dr)
    return (A*np.cos(3*t)+B*np.sin(3*t))*np.exp(-4*t) + C*np.cos(w_dr*t) + \
           D*np.sin(w_dr*t)


def omega_theory(b, c, w_dr, beta, t):
    """Theoretical values of omega."""
    A, B, C, D = _bc_consts(beta, w_dr)
    return np.exp(-4*t)*((3*B-4*A)*np.cos(3*t)-(3*A+4*B)*np.sin(3*t)) - \
           C*w_dr*np.sin(w_dr*t) + D*w_dr*np.cos(w_dr*t)


def main():
    b = 8
    c = 25
    w_dr = np.pi/2
    beta = 5

    f_func = lambda x: beta * np.sin(w_dr*x)
    y0 = [1., 0.0]
    t = np.linspace(0, 10, 101)

    sol = odeint(forced_pend, y0, t, args=(b, c, f_func))

    ax1 = plt.subplot(212)
    plt.plot(t, sol[:, 0] - theta_theory(b, c, w_dr, beta, t), 'b',
             label=r'$\theta$')
    plt.plot(t, sol[:, 1] - omega_theory(b, c, w_dr, beta, t), 'g',
             label=r'$\omega$')
    plt.setp(ax1.get_xticklabels())
    plt.xlabel('t')

    # share x only
    ax2 = plt.subplot(211, sharex=ax1)
    plt.plot(t, sol[:, 0], 'b', label=r'$\theta$')
    plt.plot(t, sol[:, 1], 'g', label=r'$\omega$')
    plt.plot(t, theta_theory(b, c, w_dr, beta, t), 'b.', label=r'$\theta_'
                                                               r'{theory}$')
    plt.plot(t, omega_theory(b, c, w_dr, beta, t), 'g.', label=r'$\omega_{'
                                                               r'theory}$')
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
