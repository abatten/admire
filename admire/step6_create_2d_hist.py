import numpy as np
import matplotlib.pyplot as plt
from fruitbat import methods, table, cosmologies
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyx

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rc('axes', labelsize=20) 
plt.rc('axes', labelsize=20) 
data = np.load('2darray_logged.npz')

fig, ax = plt.subplots(1, 1, constrained_layout=True)





extents = [0, 3, 5, -1]
im = ax.imshow(data["arr_0"].T, aspect=3/5, cmap="magma", interpolation='none', extent=extents)
ax.set_ylim(1, 4)
ax.set_xlim(0, 3)
ax.set_xlabel(r"Redshift")
ax.set_ylabel(r"$\rm{log\left(DM\right)}\ \left[pc\ cm^{-3}\right]$")

dm_vals = 10**np.linspace(1, 4, 1000)

method_list = methods.available_methods()

colours = ["#66c2a5", "#e7298a", "#8da0cb"]
label = [r"$\rm{Ioka\ 2003}$", r"$\rm{Inoue\ 2004}$", r"$\rm{Zhang\ 2018}$"]
lstyle = ["-", "--", "-."]

for j, method in enumerate(method_list):
    z_vals = np.zeros(len(dm_vals))
    cosmology = 'Planck18'

    table_name = "".join(["_".join([method, cosmology]), ".npz"])
    lookup_table = table.load(table_name)

    for i, dm in enumerate(dm_vals):
        z_vals[i] = table.get_z_from_table(dm, lookup_table)
    ax.plot(z_vals, np.log10(dm_vals), colours[j], linestyle=lstyle[j], label=label[j], linewidth=2)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
legend = plt.legend(frameon=False, fontsize=20)
plt.setp(legend.get_texts(), color='w')

#plt.colorbar(im, cax=cax)

plt.savefig("postimage_magma_lines_colorbar.png", dpi=800)
