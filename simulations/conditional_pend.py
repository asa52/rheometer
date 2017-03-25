import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from simulations.theory_calc import theta_theory, omega_theory


# Initial conditions
y0, t0 = [0., 10.], 0


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
    return [y[1], (g_0(t) + b_prime * omega_sim + k_prime * theta_sim -
                       b * y[1] - k * y[0]) / i]


def jac(t, y, i, b, k):
    """The Jacobian J[i, j] = df[i]/dy[j] of the above f."""
    return [[0, 1], [-k/i, -b/i]]


def driving_torque(t, omega_d=np.pi/2, amplitude=1, phase=0):
    """Return the value of the driving torque at time t.
    :param t: Time in seconds.
    :param omega_d: Angular frequency of the sinusoid.
    :param amplitude: Amplitude of sinusoid.
    :param phase: Phase of sinusoid in radians."""

    # TODO Add noise

    return amplitude * np.sin(omega_d * t + phase)


def baker(fun, args=None, kwargs=None, position_to_pass_through=(0, 0)):
    """Returns an object given by the function 'fun' with its arguments,
    known as a curried function or closure. These objects can be passed into
    other functions to be evaluated.

    :param fun: The function object without any arguments specified.
    :param args: A list of the positional arguments. Put any placeholder in
    the index that will not be baked into the function.
    :param kwargs: A list of keyword arguments.
    :param position_to_pass_through: A tuple specifying the index of
    positional arguments for the function 'fun' that will be skipped in
    baking. For example, (1,3) will skip positional arguments 1 through to
    3, so that the baked arguments in function 'fun' will be:
        fun(baked, unbaked, unbaked, unbaked, baked...).
    If a single position is to be skipped, enter an integer for this
    argument. For example, entering 1 will result in:
        fun(baked, unbaked, baked...).
    NOTE: Ensure the result can fill in the exact number of missing
    positional arguments!
    :return: The object containing the function with its arguments."""

    # Defaults.
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if type(position_to_pass_through) is int:
        position_to_pass_through = (position_to_pass_through,
                                    position_to_pass_through)
    elif type(position_to_pass_through) is not tuple:
        raise TypeError('The variable \'position_to_pass_through\' must be a '
                        'tuple or int.')

    def wrapped(*result):
        """Parameter position_to_pass_through specifies the index of the
        parameter 'result' in sequence of positional arguments for 'fun'."""
        return fun(*(args[:position_to_pass_through[0]] + list(result) + args[(
            position_to_pass_through[1]+1):]), **kwargs)

    return wrapped


r = ode(f, jac).set_integrator('dop853')
baked_g_0 = baker(driving_torque, args=[''], kwargs={'omega_d': np.pi/2})
r.set_initial_value(y0, t0).set_f_params(1, baked_g_0, 0, 0, 0, 0,
                                         0, 2).set_jac_params(1, 0, 2)
t1 = 1000
dt = 0.01

results = [[t0, *y0]]
while r.successful() and r.t < t1:
    data_point = [r.t+dt, *np.real(r.integrate(r.t+dt))]
    results.append(data_point)

results = np.array(results)
print(results)
plt.plot(results[:, 0], results[:, 1])
plt.plot(results[:, 0], results[:, 2])

plt.show()


def main():
    b = 8
    c = 25
    w_dr = np.pi/2
    beta = 5

    f_func = lambda x: beta * np.sin(w_dr*x)
    y0 = [1., 0.0]
    t = np.linspace(0, 10, 101)

    sol = \SOLUTION HERE

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