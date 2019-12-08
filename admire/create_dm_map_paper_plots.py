import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import e13tools
from pyx import plot_tools
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sbn

colormap = "gothic_r"

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)

#files = sorted(glob("/home/abatten/oz071/abatten/EAGLE/ADMIRE_L0100N1504/maps/T4EOS/comoving*.hdf5"))

files = sorted(glob("/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/maps/T4EOS/DM/*.hdf5"))
#files = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/zinterp/T4EOS/output/sum_dm_maps_012.hdf5"
#filename = "dispersion_measure_L0100N1504_T4EOS_z3.0.hdf5"

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from numpy.random import rand

fig = plt.figure(1, figsize=(18, 6))

grid1 = ImageGrid(fig=fig, 
                rect=111,
                nrows_ncols = (1, 3),
                axes_pad = 0.2,
                share_all=True,
                label_mode = "L",
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad="2%",
                aspect = True)

z_idx_list = [0, 5, 10]

for idx, filename in enumerate(files):
    with h5py.File(filename, "r") as f:
       
         #print(z_idx)
         extent = (0, 50, 0, 50)
         data = f["DM"][0:16000, 0:16000]
         #im = sbn.heatmap(data , cmap=colormap, vmax=500, ax=grid1[idx], cbar_ax=grid1[-1], square=True)

         im = plot_tools.plot_2d_array(data, extents=extent, cmap=colormap, vmax=500, passed_ax=grid1[idx])
#
#        # Set axis labels
         grid1[idx].set_xlabel(r"$\rm{x\ [cMpc]}$")
         #grid1[idx].tick_params(axis='both', which='major', pad=10)
grid1[0].set_ylabel(r"$\rm{y\ [cMpc]}$")
grid1.axes_all

cb1 = grid1.cbar_axes[0].colorbar(im, extend="max")
cb1.set_label_text(r"$\rm{DM\ \left[ pc\ cm^{-3} \right]}$")
plt.subplots_adjust(left=0.03, bottom=0.15, right=0.95, top=0.97)
plt.savefig("Testing.png", dpi=150)











#fig = plt.figure(figsize=(18, 7.5))
#
#
#gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=(9, 9, 9, 1))
#
## first graph
#ax1 = plt.subplot(gs[0, 0])
#ax2 = plt.subplot(gs[0, 1])
#ax3 = plt.subplot(gs[0, 2])
#cax = plt.subplot(gs[0, 3])
#
#
#axes = [ax1, ax2, ax3]
#
#
#
##grid = plt.GridSpec(1, 3, hspace=0.2, wspace=0.2)
#
#vmax_list = [100, 300, 600]
#
##grid = ImageGrid(fig, 111, 
##                nrows_ncols=(1, 3),
##                axes_pad=0.35,
##                share_all=True,
##                direction="column",
##                cbar_location="top",
##                cbar_mode="each",
##                cbar_size="8%",
##                cbar_pad=0.35)
#
#
#axes[0].set_ylabel(r"$\rm{y\ [cMpc]}$")
##
#cbar = plt.colorbar(im, cax=cax, extend="max") 
#
#
##divider = make_axes_locatable(axes)
##cax = divider.append_axes("right", size="4%", pad=0.04)
##cbar = fig.colorbar(im, ax=axes, extend="max")
##cbar.set_label(r"$\rm{DM\ \left[ pc\ cm^{-3} \right]}$")
##plt.tight_layout()
#plt.savefig("Testing.png", dpi=250)
#
#
#
#
#
#
#
## Set up figure and image grid
##        # Make the colour bar
##        divider = make_axes_locatable(ax)
##        cax = divider.append_axes("right", size="5%", pad=0.05)
##        cbar = plt.colorbar(im, cax=cax, extend="max")
##        cbar.set_label(r"$\rm{DM \ \left[pc \ cm^{-3} \right]}$")
##        #im.set_clim(vmin=0, vmax=300)
##
##        plt.savefig(f"summed_dm_large_comoving_{colormap}_1{i:02d}.png", dpi=300)
##        plt.savefig(f"summed_dm_large_comoving_{colormap}_1{i:02d}.pdf", dpi=300)
##        plt.savefig(f"summed_dm_large_comoving_{colormap}_1{i:02d}.eps", dpi=300)
##        #plt.savefig(f"rainforest_dm_cmap_large_comoving_{i:02d}.pdf", dpi=800)
##
