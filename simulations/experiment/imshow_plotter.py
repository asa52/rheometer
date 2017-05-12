import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1)
grid1 = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.6, share_all=True,
                  label_mode="L", cbar_location="right", cbar_mode="single",
                  cbar_size="7%", cbar_pad="7%", aspect=True)

amp_50 = [[[1.614614948, 1.613213622, 1.609882919, 1.604729819],
           [1.24148894, 1.241296857, 1.240991127, 1.240566319],
           [1.150274635, 1.150203522, 1.150103559, 1.149974955],
           [np.nan, np.nan, np.nan, np.nan]],
          [[1.042329922, 1.042323877, 1.042318423, 1.042311686],
           [1.07370, 1.07368, 1.07366, 1.07364],
           [1.0455, 1.0173, 1.0039, 1.0286],
           [np.nan, np.nan, np.nan, np.nan]],
          [[0.635, 0.635, 0.634, 0.630],
           [0.756, 0.757, 0.755, 0.749],
           [0.934, 0.937, 0.934, 0.923],
           [1.220, 1.228, 1.222, 1.200]]]

amp_120_200 = [[[0.806529227, 0.808024379, 0.808753377, 0.808711473],
                [0.881086873, 0.883070314, 0.884056657, 0.88404104],
                [0.970726933, 0.973420931, 0.974789211, 0.974816208],
                [1.080469457, 1.084226269, 1.086170946, 1.086277147]],
               [[0.874153397, 0.875465423, 0.876428887, 0.877043217],
                [0.925006639, 0.926567847, 0.927718364, 0.928455211],
                [0.982095143, 0.98397105, 0.985357322, 0.986248415],
                [1.04662492, 1.048899887, 1.050587951, 1.051677791]]]

phase_120_200 = [[[1.003773145, 1.020726863, 1.037341507, 1.053956151],
                  [0.991566468, 1.005920616, 1.025247855, 1.044688119],
                  [0.981281212, 1.002077773, 1.023326434, 1.044575094],
                  [0.975403923, 0.998347955, 1.021518037, 1.044122995]],
                 [[0.998224371, 1.007108026, 1.015652609, 1.027520087],
                  [0.989611974, 1.001682893, 1.013957255, 1.026095989],
                  [0.986424708, 0.999241584, 1.012329716, 1.02480752],
                  [0.983034, 0.996257761, 1.010091849, 1.023383423]]]

phases_50 = [[[0.972561164, 0.983944433, 0.994012301, 1.002208255],
              [0.972004649, 0.987157044, 1.002258847, 1.012807342],
              [0.967375453, 1.002739474, 1.032664822, 1.066156927],
              [np.nan, np.nan, np.nan, np.nan]],
             [[0.972173388, 0.972676844, 0.973683755, 0.979442033],
              [0.973, 0.974, 0.980, 0.982],
              [0.973, 0.986, 0.998, 1.011],
              [np.nan, np.nan, np.nan, np.nan]],
             [[0.99, 1.03, 1.06, 1.10],
              [0.98, 1.03, 1.07, 1.11],
              [0.96, 1.01, 1.06, 1.11],
              [0.94, 1.01, 1.07, 1.12]]]

norm = colors.Normalize(vmin=0.95, vmax=1.05)

for n in range(2):
    im = grid1[n].imshow(phase_120_200[n],
                         extent=[-5.00e-01-3.33e-1, 1.50e+00+3.33e-1,
                                 -5.00e-01-3.33e-1, 1.50e+00+3.33e-1],
                         cmap='gnuplot', norm=norm)
    grid1[n].xaxis.set_ticklabels([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].xaxis.set_ticks([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].yaxis.set_ticklabels([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].yaxis.set_ticks([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].xaxis.set_tick_params(labelsize=13)
    grid1[n].yaxis.set_tick_params(labelsize=13)
    grid1[n].xaxis.label.set_visible(True)
    grid1[n].yaxis.label.set_visible(True)

#grid1[1].annotate(
#    '', xytext=(0.16667, -0.5), xycoords='data', xy=(0.16667, 1.5),
#    arrowprops=dict(arrowstyle="-|>", color='k'))

#grid1[2].yaxis.set_label_position('left')
#grid1[2].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}}$", fontsize=20)

#grid1[2].xaxis.set_label_position('bottom')
#grid1[2].xaxis.set_label_text(r"$\frac{b'}{b_{eff}}}$", fontsize=20)

grid1[1].xaxis.set_label_position('bottom')
grid1[1].xaxis.set_label_text(r"$\frac{b'}{b_{eff}}}$", fontsize=20)

grid1[1].yaxis.set_label_position('left')
grid1[1].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}$", fontsize=20)

grid1[0].yaxis.set_label_position('left')
grid1[0].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}$", fontsize=20)

#grid1[0].set_title(r'$k_{eff}=1\times10^{-3}$, $b_{eff}=2\times10^{-7}$',
#                   fontsize=13)
#grid1[1].set_title(r'$k_{eff}=1\times10^{-5}$, $b_{eff}=5\times10^{-9}$',
#                   fontsize=13)
#grid1[2].set_title(r'$k_{eff}=1\times10^{-5}$, $b_{eff}=2\times10^{-7}$',
#                   fontsize=13)

grid1[0].set_title(r'$\Delta t=\frac{T}{120}$', fontsize=13)
grid1[1].set_title(r'$\Delta t=\frac{T}{200}$', fontsize=13)

cb1 = grid1.cbar_axes[0].colorbar(im)

cb1.set_label_text(r'$\frac{\phi[R(\omega_{res})]}{\phi[R_{A}(\omega_{res})]}$',
                   rotation=90, fontsize=16)
cb1.ax.tick_params(labelsize=13)
plt.show()
