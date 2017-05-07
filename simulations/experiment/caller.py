"""Front-end file to call different experiments from. Separate from others so 
only this file needs to be changed when different experiments are run."""

import experiments as exp
import helpers as h
import theory as t
import numpy as np


def main():
    """This is the function run when this file is called."""

    configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
    k_eff = 1.e-4
    b_eff = 5.e-7
    configs['k'] = np.array([k_eff])
    w_res, gamma = t.w_res_gamma(b_eff, k_eff, i)
    configs['w_d'] = w_res
    print(w_res, gamma)
    #period = 2 * np.pi / w_res
    #for delay in np.array([0.19]):
    #    for wd in np.linspace(10, 60, 100):
     #       print('Delay of 0.19 over many frequencies - 3 peaks expected.')
      #      configs['delay'] = np.array([delay])
      #      configs['w_d'] = np.array([wd])
      #      rk_test = exp.FixedStepIntegrator(config=configs)
      #      rk_test.run(plot=False)

    # theory_vs_fourier = TheoryVsFourier()
    # theory_vs_fourier.run()

    # TODO note down the config for the Python experiment, including the type of
    # TODO integrator used and the behaviour of speed for high frequencies - the
    # frequency does not match the actual frequency (slightly lower - check).
    # Largely speaking this does not matter, but does explain why the errors are
    # so large.


if __name__ == '__main__':
    main()
