"""Front-end file to call different experiments from. Separate from others so 
only this file needs to be changed when different experiments are run."""

import numpy as np
import helpers as h
import experiments as exp


def main():
    """This is the function run when this file is called."""
    read_all = exp.ReadAllData()
    read_all.run()

    #configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
    #k_eff = 5.e-4
    #b_eff = 5.e-7
    #i = configs['i']
    #w_res = np.sqrt(k_eff/i - b_eff**2/(2*i**2))
    #gamma = b_eff/i
    #for k in np.array([0.5 * k_eff, k_eff, 1.5 * k_eff]):
    #    configs['k'] = np.array([k])
    #    configs['k\''] = np.array([k - k_eff])
    #    for b in np.array([0.5 * b_eff, b_eff, 1.5 * b_eff]):
    #        configs['b'] = np.array([b])
    #        configs['b\''] = np.array([b - b_eff])
    #        for wd in np.linspace(w_res - gamma, w_res + gamma, 10):
    #            configs['w_d'] = wd
    #            rk_test = exp.FixedStepIntegrator(config=configs)
    #            rk_test.run(plot=False)
    #        for wd in np.arange(10, 150, 10):
    #            configs['w_d'] = wd
    #            rk_test = exp.FixedStepIntegrator(config=configs)
    #            rk_test.run(plot=False)
    # theory_vs_fourier = TheoryVsFourier()
    # theory_vs_fourier.run()

    # TODO note down the config for the Python experiment, including the type of
    # TODO integrator used and the behaviour of speed for high frequencies - the
    # frequency does not match the actual frequency (slightly lower - check).
    # Largely speaking this does not matter, but does explain why the errors are
    # so large.


if __name__ == '__main__':
    main()
