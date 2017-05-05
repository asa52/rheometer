"""Front-end file to call different experiments from. Separate from others so 
only this file needs to be changed when different experiments are run."""

import experiments as exp


def main():
    """This is the function run when this file is called."""
    #configs = h.yaml_read('../configs/ReadAllData.yaml')
    #configs['directory'] = np.array([r'C:/Users/Abhishek/OneDrive -
    ## University '
    #                                 r'Of Cambridge/Project/Tests/'
    #                                 r'ExperimentClasses/FixedStepIntegrator/'
    #                                 r'k-keff-plot-120-low-freq/'])
    #read_all = exp.ReadAllData(config=configs)
    #read_all.run(plot=False)

    #configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
    #k_eff = 5.e-4
    #b_eff = 5.e-7
    #i = configs['i']
    #configs['b'] = np.array([b_eff])
    #w_res, gamma = t.w_res_gamma(b_eff, k_eff, i)
    #configs['w_d'] = w_res
    #print(w_res, gamma)
    #for k_pr in np.linspace(-5 * k_eff, 5 * k_eff, 11):
    #    print("11 values for the k'/k_eff plot. high frequency. dt=T/{
    # }".format(
    #        configs['sampling_divider']))
    #    configs['k'] = np.array([k_pr + k_eff])
    #    configs['k\''] = np.array([k_pr])
    #    # configs['b'] = np.array([b_eff])
    #    # configs['b\''] = np.array([b - b_eff])
    rk_test = exp.FixedStepIntegrator()
    rk_test.run(tags='test-w-delay-0')

    # theory_vs_fourier = TheoryVsFourier()
    # theory_vs_fourier.run()

    # TODO note down the config for the Python experiment, including the type of
    # TODO integrator used and the behaviour of speed for high frequencies - the
    # frequency does not match the actual frequency (slightly lower - check).
    # Largely speaking this does not matter, but does explain why the errors are
    # so large.


if __name__ == '__main__':
    main()
