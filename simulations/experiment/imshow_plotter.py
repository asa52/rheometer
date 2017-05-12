import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
import numpy as np

fig = plt.figure(1)

grid1 = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.6, share_all=True,
                  label_mode="L", cbar_location="right", cbar_mode="single",
                  cbar_size="7%", cbar_pad="7%", aspect=True)

amp_50 = [[[0.614614948, 0.613213622, 0.609882919, 0.604729819],
           [0.24148894, 0.241296857, 0.240991127, 0.240566319],
           [0.150274635, 0.150203522, 0.150103559, 0.149974955],
           [np.nan, np.nan, np.nan, np.nan]],
          [[0.042329922, 0.042323877, 0.042318423, 0.042311686],
           [0.07370, 0.07368, 0.07366, 0.07364],
           [0.0455, 0.0173, 0.0039, 0.0286],
           [np.nan, np.nan, np.nan, np.nan]],
          [[0.635, 0.635, 0.634, 0.630],
           [0.756, 0.757, 0.755, 0.749],
           [0.934, 0.937, 0.934, 0.923],
           [1.220, 1.228, 1.222, 1.200]]]

phases = [[[0.972561164, 0.983944433, 0.994012301, 1.002208255],
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

norm = colors.Normalize(vmin=0.92, vmax=1.12)

for n in range(3):
    im = grid1[n].imshow(phases[n],
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

grid1[2].yaxis.set_label_position('left')
grid1[2].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}}$", fontsize=20)

grid1[2].xaxis.set_label_position('bottom')
grid1[2].xaxis.set_label_text(r"$\frac{b'}{b_{eff}}}$", fontsize=20)

grid1[1].yaxis.set_label_position('left')
grid1[1].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}$", fontsize=20)

grid1[0].yaxis.set_label_position('left')
grid1[0].yaxis.set_label_text(r"$\frac{k'}{k_{eff}}$", fontsize=20)

grid1[0].set_title(r'$k_{eff}=1\times10^{-3}$, $b_{eff}=2\times10^{-7}$',
                   fontsize=13)
grid1[1].set_title(r'$k_{eff}=1\times10^{-5}$, $b_{eff}=5\times10^{-9}$',
                   fontsize=13)
grid1[2].set_title(r'$k_{eff}=1\times10^{-5}$, $b_{eff}=2\times10^{-7}$',
                   fontsize=13)
cb1 = grid1.cbar_axes[0].colorbar(im)

cb1.set_label_text('Fractional error in $\phi[R(\omega_{res})]$', rotation=90,
                   fontsize=16)
cb1.ax.tick_params(labelsize=13)
plt.show()
