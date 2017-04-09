"""Generic plotting code, as Bash on Windows cannot make displayable plots."""

import numpy as np
import matplotlib.pyplot as plt

import simulations.experiment.helpers as h


def two_by_n_plotter(datasets, x_axes_labels=None, y_top_labels=None,
                     y_bottom_labels=None):
    """Plots a 2 x N series of subplots in a single figure. The format for 
    datasets is:
        [
            [
                [[x1_top1, y1_top1],       [x2_top1, y2_top1],       ...],
                [[x1_bottom1, y1_bottom1], [x2_bottom1, y2_bottom1], ...],
            ], 
            ...
        ].
    x1, y1 is one data series, plotted in either the top or bottom plot of 
    set n of N. Error bars may be allowed by making x or y a 2-column array, 
    with the second column specifying the errors. The x axes of the top and 
    bottom plot are shared in each pair.
    The axes labels are lists with the appropriate string labels. Axes are 
    labelled from left to right in the figure."""

    # create subplots figure
    fig, ax = plt.subplots(ncols=len(datasets), nrows=2, sharex='col')
    x_axis, y_axis_top, y_axis_bottom = False, False, False
    if x_axes_labels is not None:
        x_axis = True
        assert len(datasets) == len(x_axes_labels)
    if y_top_labels is not None:
        y_axis_top = True
        assert len(datasets) == len(y_top_labels)
    if y_bottom_labels is not None:
        y_axis_bottom = True
        assert len(datasets) == len(y_bottom_labels)

    # Perform length of array and format checks.
    for k in range(len(datasets)):
        for j in range(len(datasets[k])):
            for one_series in datasets[k][j]:
                axes = h.check_types_lengths(*one_series)
                for i in range(len(axes)):
                    if not (axes[i].ndim == 2 and axes[i].shape[-1] == 2):
                        # Data has no error bars - create error bars of zero.
                        axes[i] = np.vstack((axes[i],
                                             np.zeros(axes[i].size))).T
                if len(datasets) == 1:
                    axis = ax[j]
                else:
                    axis = ax[j, k]
                axis.errorbar(axes[0][:, 0], axes[1][:, 0],
                              xerr=axes[0][:, 1], yerr=axes[1][:, 1], fmt=':')
                #axis.set_xlim(min(axes[0][:, 0]), max(axes[0][:, 0]))
                #axis.set_ylim(min(axes[1][:, 0]), max(axes[1][:, 0]))
                axis.tick_params(direction='out', labelsize=12)
                axis.grid(True)
                if x_axis:
                    if x_axes_labels[k] is not None and j == 1:
                        axis.set_xlabel(x_axes_labels[k], fontsize=14,
                                        fontweight='bold', )
                if j == 0:
                    if y_axis_top:
                        if y_top_labels[k] is not None:
                            axis.set_ylabel(y_top_labels[k], fontsize=14,
                                            fontweight='bold', labelpad=-5)
                elif j == 1:
                    if y_axis_bottom:
                        if y_bottom_labels[k] is not None:
                            axis.set_ylabel(y_bottom_labels[k], fontsize=14,
                                            fontweight='bold', labelpad=-5)
                axis.ticklabel_format(useOffset=False)
    plt.show()
    #plt.figure(figsize=(7, 10))
    #ax1 = plt.subplot(413)
    #plt.plot(exp_results[:, 0], normalised_theta_diff, 'k',
    #         label=r'$\theta$')
    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.grid()
    #plt.ylabel(r'$(\theta_{sim}-\theta_{an})/|\theta_{max}|$', fontsize=14,
    #           fontweight='bold')#

    # share x only
    #ax2 = plt.subplot(411, sharex=ax1)
    #plt.plot(exp_results[:, 0], exp_results[:, 1], 'r-.',
    #         label=r'Simulated')
    #plt.plot(theory[:, 0], theory[:, 1], 'b:', label=r'Analytic')
    #plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2)
    #plt.xlim([t0, t_fin])
    #plt.ylabel(r'$\theta$/rad', fontsize=14, fontweight='bold')
    #plt.grid()

    #ax3 = plt.subplot(414, sharex=ax1)
    #plt.plot(exp_results[:, 0], normalised_omega_diff, 'k',
    #         label=r'$\omega$')
    #plt.setp(ax1.get_xticklabels())
    #plt.xlabel('t/s', fontsize=14, fontweight='bold')
    #plt.ylabel(r'$(\omega_{sim}-\omega_{an})/|\omega_{max}|$',
     #          fontsize=14, fontweight='bold')
    #plt.grid()

    #ax4 = plt.subplot(412, sharex=ax1)
    #plt.plot(exp_results[:, 0], exp_results[:, 2], 'r-.',
    #         label=r'$\omega_{exp}$')
    #plt.plot(exp_results[:, 0], theory[:, 2], 'b:',
    #         label=r'$\omega_{theory}$')
    #plt.setp(ax4.get_xticklabels(), visible=False)
    #plt.xlim([t0, t_fin])
    #plt.ylabel('$\omega$/rad/s', fontsize=14, fontweight='bold')
    #plt.ticklabel_format(useOffset=False)
    #plt.grid()
    #plt.show()

if __name__ == '__main__':
    t = np.linspace(0.001, 10, 10000)
    thing = [
                [
                    [
                        [t, np.sin(t)],
                        [t, np.cos(t)]
                    ],
                    [
                        [t, np.exp(-t)],
                        [t, np.tanh(t)]
                    ],
                ],
        [
            [
                [t, np.sin(t)],
                [t, np.cos(t)]
            ],
            [
                [t, np.log(t)],
                [t, np.log10(t)]
            ],
        ],
        [
            [
                [t, np.sin(t)],
                [t, np.cos(t)]
            ],
            [
                [t, np.log(t)],
                [t, np.log10(t)]
            ],
        ],
        [
            [
                [t, np.sin(t)],
                [t, np.cos(t)]
            ],
            [
                [t, np.log(t)],
                [t, np.log10(t)]
            ],
        ],
    ]
    two_by_n_plotter(thing, x_axes_labels=['t', 't', 't', 't'],
                     y_top_labels=[1, 2, 3, 4], y_bottom_labels=[1, 2, 3, 4])
