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


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc('ytick', labelsize=12)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('axes', grid=True)
plt.rc("xtick", top=True)
plt.rc("ytick", right=True)
plt.rc("grid", linewidth=0.8)
plt.rc("grid", linestyle=":")
plt.rc("grid", alpha=0.8)



SIMPREFIX = "Ref"
SIMNAME = "L0050N0752"
SNAPNUM = 28
NUMPIX = 16000
HALOSONLY = False

PIXWIDTH = 16000
MPCWIDTH = 50

OUTPUTSUBDIR = "analysis_plots/shuffled"

file = f"coldens_electrons_{SIMNAME}_{SNAPNUM}_test3.4_PtAb_C2Sm_{NUMPIX}pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.hdf5"


# Redirect path when only using halo data
if HALOSONLY:
    HALOS = "/HALOS"
else:
    HALOS = ""

path = f"/fred/oz071/abatten/ADMIRE_ANALYSIS{HALOS}/ADMIRE_{SIMPREFIX}{SIMNAME}/all_snapshot_data/maps/T4EOS/DM/{file}"

print(path)

with h5py.File(path, "r") as f:

    print(f["Header"].attrs["Redshift"])
    extent = (0, MPCWIDTH, 0, MPCWIDTH)

    print("converting")
    data = f["map"][:, :] * apu.cm**-2
    data = data.to("pc cm^-3").value

    z = f["Header"].attrs["Redshift"]
    if z < 0.001:
        z = 0.0


    print("plotting")

    ##### MAKE PLOT #####
    fig, ax = plt.subplots(1, 1, figsize=(8,6))    
    im1 = plot_tools.plot_2d_array(data, extents=extent, cmap=colormap, passed_ax=ax, vmax=400)
    txt1 = im1.axes.text(0.15, 0.9, r'$z = $ {:.2f}'.format(z), horizontalalignment='center', verticalalignment='center', transform=im1.axes.transAxes, fontsize=14)
    txt1.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    ##### COLOR BAR #####
    cbar = plt.colorbar(im1, ax=ax, extend="max")
    cbar.set_label(r"$\mathrm{DM\ \left[ pc\ cm^{-3} \right]}$", fontsize=14)

    ax.set_xlabel("$\mathrm{x\ [cMpc]}$")
    ax.set_ylabel("$\mathrm{y\ [cMpc]}$")

    print("saving")


    output_filename = f"./{OUTPUTSUBDIR}/DM_Map_{SIMPREFIX}{SIMNAME}_z{z:.2f}_pix{PIXWIDTH}.png"

    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    