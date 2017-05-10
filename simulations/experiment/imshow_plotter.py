
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
import numpy as np

fig = plt.figure(1)

grid1 = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5, share_all=True,
                  label_mode="L", cbar_location="right", cbar_mode="single",
                  cbar_size="7%", cbar_pad="7%", aspect=True)

for n in range(4):
    im = grid1[n].imshow([[1.220, 1.228, 1.222, 1.200],
                          [0.934, np.nan, 0.934, 0.923],
                          [0.756, 0.757, 0.755, 0.749],
                          [0.635, 0.635, 0.634, 0.630]],
                         extent=[-5.00e-01, 1.50e+00, -5.00e-01, 1.50e+00])
    grid1[n].xaxis.set_ticklabels([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].xaxis.set_ticks([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].yaxis.set_ticklabels([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].yaxis.set_ticks([-0.5, 0., 0.5, 1.0, 1.5])
    grid1[n].xaxis.set_tick_params(labelsize=13)
    grid1[n].yaxis.set_tick_params(labelsize=13)
    grid1[n].xaxis.label.set_visible(True)
    grid1[n].yaxis.label.set_visible(True)

grid1[2].xaxis.set_label_position('bottom')
grid1[2].xaxis.set_label_text('foo', fontsize=16)


grid1[3].xaxis.set_label_position('bottom')
grid1[3].xaxis.set_label_text('bar', fontsize=16)


grid1[2].yaxis.set_label_position('left')
grid1[2].yaxis.set_label_text('foo', fontsize=16)

grid1[0].yaxis.set_label_position('left')
grid1[0].yaxis.set_label_text('bar', fontsize=16)

cb1 = grid1.cbar_axes[0].colorbar(im, ticks=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

cb1.set_label_text('Fractional error in $|R(\omega_{res})|$', rotation=90,
                   fontsize=16)
cb1.ax.tick_params(labelsize=13)
fig.text(0.5, 0.05, r"$\mathbf{\frac{b'}{b_{eff}}}$", ha='center', fontsize=20)
fig.text(0.235, 0.55, r"$\mathbf{\frac{k'}{k_{eff}}}$", va='center',
         rotation='vertical', fontsize=20)

plt.show()
