"""Plotting functions for the main types of plots I need."""

import matplotlib.pyplot as plt
import numpy as np

import helpers as h


def two_by_n_plotter(datasets, start, params_dict, savepath=None, show=False,
                     tag=None, x_axes_labels=None, y_top_labels=None,
                     y_bottom_labels=None, **kwargs):
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
    bottom plot are shared in each pair. A third entry in [x1, y1, label] is 
    a string specifying the label of the data series, if one is desired for a 
    legend. A fourth entry after that can be used for the fmt (markerstyle) for 
    that series.
    The axes labels are lists with the appropriate string labels. Axes are 
    labelled from left to right in the figure.
    params_dict specifies the parameters to be written to the text file 
    accompanying the plot, to record the control variables.
    kwargs is a list of custom commands to be passed to the plot, such as 
    legend formats, just before plotting. They are called as methods of the 
    figure object. The 'legend' parameter takes a dictionary of keyword 
    arguments as its value; these are passed into the fig.legend method, 
    for example. """

    # create subplots figure
    fig, ax = plt.subplots(ncols=len(datasets), nrows=2, sharex='col',
                           figsize=(21, 10.5))
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
                axes = h.check_types_lengths(*one_series[0:2])
                if len(one_series) >= 3 and type(one_series[2]) is str:
                    # Data label used for the third entry.
                    label = one_series[2]
                else:
                    label = None

                if len(one_series) == 4 and type(one_series[3]) is str:
                    # fmt used for the 4th entry.
                    fmt = one_series[3]
                else:
                    fmt = ':'

                for i in range(len(axes)):
                    if not (axes[i].ndim == 2 and axes[i].shape[-1] == 2):
                        # Data has no error bars - create error bars of zero.
                        axes[i] = np.vstack((axes[i],
                                             np.zeros(axes[i].size))).T
                if len(datasets) == 1:
                    axis = ax[j]
                else:
                    axis = ax[j, k]
                axis.errorbar(axes[0][:, 0], axes[1][:, 0], xerr=axes[0][:, 1],
                              yerr=axes[1][:, 1], fmt=fmt, label=label,
                              markersize=2)
                axis.tick_params(direction='out')
                axis.grid(True)
                if x_axis:
                    if x_axes_labels[k] is not None and j == 1:
                        axis.set_xlabel(x_axes_labels[k], fontweight='bold',
                                        fontsize=13)
                if j == 0:
                    if y_axis_top:
                        if y_top_labels[k] is not None:
                            axis.set_ylabel(y_top_labels[k], fontweight='bold',
                                            fontsize=13)
                elif j == 1:
                    if y_axis_bottom:
                        if y_bottom_labels[k] is not None:
                            axis.set_ylabel(y_bottom_labels[k],
                                            fontweight='bold', fontsize=13)
                axis.ticklabel_format(axis='both', useOffset=False)
                axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                if 'legend' in kwargs and label is not None:
                    legend_params = kwargs['legend']
                    axis.legend(**legend_params)

    plot_name, descrip_name = _create_names(start, tag, filetype='png')
    if savepath is not None:
        # Save only if save path is not none.
        fig.savefig(savepath + plot_name, dpi=300)
        with open(savepath + descrip_name, 'w') as f:
            for key in params_dict:
                if type(params_dict[key]) is np.ndarray:
                    f.write('{}: {}\r\n'.format(key, np.asscalar(params_dict[
                                                                    key])))
                else:
                    f.write('{}: {}\r\n'.format(key, params_dict[key]))
    if show:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
    plt.close(fig)
    return


def _create_names(start, tag=None, filetype='pdf'):
    """Create the plot file name according to the format: 
    start-tags-format. All arguments are strings apart from kwargs 
    which is a dictionary of control parameters."""
    plot_name = start + '-'
    if tag is not None:
        plot_name += '-' + tag
    descrip_name = plot_name + '.txt'
    plot_name += '.' + filetype
    return plot_name, descrip_name
