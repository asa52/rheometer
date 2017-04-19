from scipy.integrate import ode


def f(t, ys, a):
    return a

p = [1]
r = ode(f).set_initial_value(0, 0).set_f_params(*p)

s = 0
while r.successful() and s < 5:
    r.integrate(r.t+1)
    print(r.t, r.y, p)

    p[0] += 1
    r = ode(f).set_initial_value(r.y,r.t).set_f_params(*p) ### Option 1
    # r = r.set_f_params(*p) ### Option 2

    s += 1
