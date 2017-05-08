"""Front-end file to call different experiments from. Separate from others so 
only this file needs to be changed when different experiments are run."""

import matplotlib
matplotlib.use('Agg')
import experiments as exp
import helpers as h
import theory as t
import numpy as np
import measurement as m
import sys


def main(make_then_read=True):
    """This is the function run when this file is called."""
    if make_then_read:
        print(sys.argv)
        num_args = len(sys.argv[1:])
        if num_args < 5:
            print('Incorrect number of input arguments. 5 expected. Switching '
                  'ALL parameters to default values.')
            dt_fraction = 120.
            k_eff = 1.e-4
            b_pr = 0.
            b_eff = 5.e-7
            delay = 0.

        else:
            dt_fraction = float(sys.argv[1])
            k_eff = float(sys.argv[2])
            b_pr = float(sys.argv[3])
            b_eff = float(sys.argv[4])
            delay = float(sys.argv[5])

        configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
        i = configs['i']
        configs['k'] = np.array([k_eff])
        configs['k\''] = np.array([0.])     # todo remove these
        configs['b\''] = np.array([b_pr])
        configs['b'] = np.array([b_eff - b_pr])
        configs['sampling_divider'] = np.array([dt_fraction])
        configs['delay'] = np.array([delay])
        w_res, gamma = t.w_res_gamma(b_eff, k_eff, i)
        configs['w_d'] = w_res
        print(w_res, gamma)
        k_range = np.linspace(-5. * k_eff/10., 1.5 * k_eff, 11)
        for k_pr in k_range:
            print("Running for parameters dt={},k_pr={},keff={},bpr={},beff={},"
                  "delay={}".format(dt_fraction, k_pr, k_eff, b_pr, b_eff,
                                    delay))
            configs['k\''] = np.array([k_pr])
            configs['k'] = np.array([k_pr + k_eff])
            rk_test = exp.FixedStepIntegrator(config=configs, norm_struct=False)
            results = rk_test.run(plot=True, savedata=False)['all-mmts']
            mmts = m.one_mmt_set(results['t'], results['theta'],
                                 results['sine-torque'], configs['b'],
                                 configs['b\''], configs['k'], configs['k\''],
                                 configs['i'])
            transfer = t.theory_response(configs['b'], configs['k'],
                                         configs['i'], configs['b\''],
                                         configs['k\''], w_res)
            print(mmts)
            mmts[0, :] = mmts[0, :] * 2 * np.pi
            mmts[1, :] = mmts[1, :] / np.abs(transfer) * 10**7 # NOTE this is
            #  because g0mag is 10^-7
            mmts[2, :] = np.abs(mmts[2, :] / np.angle(transfer))
            print(mmts)
            with open('dt={},keff={},bpr={},beff={},delay={}.txt'.format(
                    dt_fraction, k_eff, b_pr, b_eff, delay), 'a') as f:
                write_string = '{} {} {} {} {} {}'.format(
                    mmts[0, 0], mmts[0, 1], mmts[1, 0], mmts[1, 1], mmts[2, 0],
                    mmts[2, 1])
                f.write(write_string)
    else:
        #configs = h.yaml_read('../configs/ReadAllData.yaml')
        #reader = exp.ReadAllData(config=configs, norm_struct=True)
        #reader.run(plot=False)

        # Find the maximum normalised error for no NR
        configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
        k_range = np.array([1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4, 3e-4,
                            5e-4, 8e-4, 1e-3, 2e-3])
        b_range = np.array([1e-6, 2e-6, 5e-6, 1e-8, 2e-8, 3e-8, 5e-8, 8e-8,
                            5e-10])
        i = configs['i']
        for divider in [50., 120., 500.]:
            for b in b_range:
                for k in k_range:
                    configs['k'] = np.array([k])
                    configs['b'] = np.array([b])
                    configs['sampling_divider'] = np.array([divider])
                    w_res, gamma = t.w_res_gamma(b, k, i)
                    configs['w_d'] = w_res
                    configs['tfin'] = np.array([10.]) if 10. > 4. / gamma else \
                        np.array([4 / gamma])
                    print(w_res, gamma)
                    print('k b divider', k, b, divider)

                    rk_test = exp.FixedStepIntegratorError(config=configs,
                                                           norm_struct=False)
                    results = np.array(rk_test.run(plot=False, savedata=False)[
                                           'all-mmts'].values)
                    print(results)
                    with open('nonNRtests.txt', 'ab') as f:
                        np.savetxt(f, results, delimiter=',', newline='\r\n')

    # theory_vs_fourier = TheoryVsFourier()
    # theory_vs_fourier.run()

    # TODO note down the config for the Python experiment, including the type of
    # TODO integrator used and the behaviour of speed for high frequencies - the
    # frequency does not match the actual frequency (slightly lower - check).
    # Largely speaking this does not matter, but does explain why the errors are
    # so large.


if __name__ == '__main__':
    main(make_then_read=False)
