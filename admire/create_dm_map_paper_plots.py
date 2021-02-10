import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import e13tools
import cmasher as cmr
from pyx import plot_tools
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sbn
import astropy.units as apu
import matplotlib.patheffects as PathEffects

colormap = cmr.freeze_r

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('font', size=20)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc('ytick', labelsize=20)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=24)
plt.rc('axes', labelsize=24)
plt.rc("xtick", top=True)
plt.rc("ytick", right=True)

#files = sorted(glob("/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/maps/T4EOS/DM/coldens*.hdf5"))

#files = sorted(glob("/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS/interpolated_dm_map_idx_corrected*.hdf5"))
files = sorted(glob("/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/maps/T4EOS/DM/*.hdf5"))

#files = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/zinterp/T4EOS/output/sum_dm_maps_012.hdf5"
#filename = "dispersion_measure_L0100N1504_T4EOS_z3.0.hdf5"

print(files)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from numpy.random import rand

#fig = plt.figure(1, figsize=(18, 6))

#grid1 = ImageGrid(fig=fig,
#                rect=111,
#                nrows_ncols = (2, 3),
#                axes_pad = 0.2,
#                share_all=True,
#                label_mode = "L",
#                cbar_location = "right",
#                cbar_mode="single",
#                cbar_size="7%",
#                cbar_pad="2%",
#                aspect = True)

#z_idx_list = [0, 5, 10]
#z_idx_list
print("plotting")

fig, ax = plt.subplots(nrows=2, ncols=3, constrained_layout=True, sharex=True, sharey=True, figsize=(18,12))
w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)

#files = [files[0], files[34], files[65]]
#files = [files[0], files[34], files[65]]

files = [files[16], files[7], files[0]]

for idx, filename in enumerate(files):
    with h5py.File(filename, "r") as f:

         print(idx, f["Header"].attrs["Redshift"])
         extent = (0, 50, 0, 50)
         data = f["map"][0:16000, 0:16000] * apu.cm**-2

         data = data.to("pc cm^-3").value
         #im = sbn.heatmap(data , cmap=colormap, vmax=500, ax=grid1[idx], cbar_ax=grid1[-1], square=True)

#         im = plot_tools.plot_2d_array(data, extents=extent, cmap=colormap, passed_ax=grid1[idx], vmax=600)
#         im.axes.text(0.15, 0.9, r'$z = ${:.2f}'.format(f["HEADER"].attrs["Redshift"]), horizontalalignment='center', verticalalignment='center', transform=im.axes.transAxes, fontsize=16)

         z = f["Header"].attrs["Redshift"]
         if z < 0.001:
             z = 0.0

         print("row1")
         im1 = plot_tools.plot_2d_array(data, extents=extent, cmap=colormap, passed_ax=ax[0][idx], vmax=400)
         txt1 = im1.axes.text(0.15, 0.9, r'$z = $ {:.2f}'.format(z), horizontalalignment='center', verticalalignment='center', transform=im1.axes.transAxes, fontsize=24)
         txt1.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

         print("row2")
         im2 = plot_tools.plot_2d_array(data/(1+z), extents=extent, cmap=colormap, passed_ax=ax[1][idx], vmax=400)
         txt2 = im2.axes.text(0.15, 0.9, r'$z = $ {:.2f}'.format(z), horizontalalignment='center', verticalalignment='center', transform=im2.axes.transAxes, fontsize=24)
         txt2.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
#        # Set axis labels
         #grid1[idx].set_xlabel(r"$\rm{x\ [cMpc]}$")
         #grid1[idx].tick_params(axis='both', which='major', pad=10)
#grid1[0].set_ylabel(r"$\rm{y\ [cMpc]}$")
#grid1.axes_all


cbar1 = plt.colorbar(im1, ax=ax[0][2], extend="max")
cbar2 = plt.colorbar(im2, ax=ax[1][2], extend="max")

cbar1.set_label(r"$\mathrm{DM\ \left[ pc\ cm^{-3} \right]}$", fontsize=24)
cbar2.set_label(r"$\mathrm{DM}/(1+z)\ \mathrm{\left[ pc\ cm^{-3} \right]}$", fontsize=24)


ax[0][0].set_ylabel("$\mathrm{y\ [cMpc]}$")
ax[1][0].set_ylabel("$\mathrm{y\ [cMpc]}$")

ax[1][0].set_xlabel("$\mathrm{x\ [cMpc]}$")
ax[1][1].set_xlabel("$\mathrm{x\ [cMpc]}$")
ax[1][2].set_xlabel("$\mathrm{x\ [cMpc]}$")
#cb1 = grid1.cbar_axes[0].colorbar(im)
#cb1.set_label_text(r"$\rm{DM\ \left[ pc\ cm^{-3} \right]}$")
#plt.subplots_adjust(left=0.03, bottom=0.15, right=0.95, top=0.97)
print("saving")
plt.savefig("DM_6_panel_plot_50cMpc_New.png", dpi=250)











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
