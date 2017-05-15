"""Front-end file to call different experiments from. Separate from others so 
only this file needs to be changed when different experiments are run. Main 
location for scripting different tests."""

import numpy as np

import experiments as exp
import helpers as h
import measurement as m
import theory as t


def nr_test():
    # Find the maximum normalised error for no NR
    configs = h.yaml_read('../configs/FixedStepIntegrator.yaml')
    keff_range = np.array([1e-5])
    beff_range = np.array([1e-8])
    i = configs['i']
    for divider in [120.]:
        for beff in beff_range:
            for keff in keff_range:
                kpr_range = np.array([-5e-6])
                bpr_range = np.array([0.])
                for kpr in kpr_range:
                    for bpr in bpr_range:
                        configs['k\''] = np.array([kpr])
                        configs['k'] = np.array([keff + kpr])
                        configs['b\''] = np.array([bpr])
                        configs['b'] = np.array([beff + bpr])
                        configs['sampling_divider'] = np.array([divider])
                        try:
                            w_res, gamma = t.w_res_gamma(beff, keff, i)
                            configs['delay'] = 2*np.pi/(2 * w_res * divider)
                            delay = configs['delay']
                            if w_res < 9:
                                raise ValueError
                            configs['w_d'] = w_res
                            configs['tfin'] = np.array(
                                [10.]) if 10. > 4. / gamma else \
                                np.array([4 / gamma])
                            print(w_res, gamma)
                            print('keff k\' beff b\' divider', keff, kpr, beff,
                                  bpr, divider)

                            rk_test = exp.FixedStepIntegrator(
                                config=configs, norm_struct=False)
                            results = rk_test.run(
                                plot=False, savedata=False)['all-mmts']
                            mmts = m.one_mmt_set(
                                results['t'], results['theta'],
                                results['sine-torque'], configs['b'],
                                configs['b\''], configs['k'],
                                configs['k\''], configs['i'])
                            transfer = t.theory_response(
                                configs['b'], configs['k'], configs['i'],
                                configs['b\''], configs['k\''], w_res)
                            mmts[0, :] = mmts[0, :] * 2 * np.pi
                            mmts[1, :] = mmts[1, :] / np.absolute(transfer) * \
                                         10 ** 7
                            # NOTE this is because g0mag is 10^-7
                            mmts[2, :] = np.abs(mmts[2, :] / np.angle(transfer))
                            print(mmts)
                            with open('NR-no-delay-var-b-different-k-2.txt',
                                      'a') as f:
                                write_string = '{} {} {} {} {} {} {} {} {} {}' \
                                               ' {} {}'.format(
                                    divider, kpr, keff, bpr, beff, delay,
                                    mmts[0, 0], mmts[0, 1], mmts[1, 0],
                                    mmts[1, 1], mmts[2, 0], mmts[2, 1])
                                f.write(write_string + '\r\n')
                        except ValueError:
                            pass
                            # Ignore values where there is no resonant frequency
                            # due to overdamping, or if they are too small.


if __name__ == '__main__':
    nr_test()
