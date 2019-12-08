import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fruitbat import methods, table, cosmologies
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import e13tools
from fruitbat import methods, table, cosmologies
from astropy.io import ascii
import os
import h5py

from glob import glob

from pyx import plot_tools

from matplotlib.ticker import MaxNLocator

##############################################################
plot_dm_z_histogram = True
plot_ioka = True
plot_localised_frbs = True
plot_cumulative_mean = True
plot_cumulative_median = True
plot_cumulative_mode = True
plot_pol_data = True
add_dolag_data = True
plot_mcquinn_data = True
add_illustris_data = True


##############################################################


##############################################################
# LOCALISED FRB'S DATA

frb_obs_list = [
    # Bannister Localised FRB
    {"DM": np.array([361.42]),
     ""
     "DM_MW": np.array([71.5]), 
     "z": np.array([0.4214]),
     "Label": "Bannister+(2019)"
    },
    
    # Ravi Localised FRB
    {"DM": np.array([760.8]),
     "DM_MW": np.array([87]), 
     "z": np.array([0.66]),
     "Label": "Ravi+(2019)"
    }    
]
##############################################################











# Set the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=20)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=20)
plt.rc('axes', labelsize=20)

path = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/linear_redshift_interp/output/T4EOS"
#path = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS/"
files = sorted(glob(path + "/admire_output_2D_array_logged*"))




#files = sorted(glob(path + "/admire_output_DM_z_*"))


#filename = "admire_output_2D_array_logged_unnormed_000.npz"

#for idx, fname in enumerate(files):
#    print(f"{idx+1}/25")
#    with h5py.File(fname, "r") as ds:
#        if idx == 0:
#            dm_data = ds["DMz_hist"][:]
#            bin_centres = ds["Bin_Centres"][:]
#            data_redshifts = ds["Redshifts"][:]
#            bins = ds["Bin_Edges"][:]
#
#        else:
#            dm_data += ds["DMz_hist"][:]

#print(dm_data)
print(files)
for i, f in enumerate(files):
    print(f"{i+1}/25")
    if i == 0:
        data = np.load(os.path.join(path, f))["arr_0"]
    else:
        data += np.load(os.path.join(path, f))["arr_1"]



# Load the 2D data files
#data = np.load(os.path.join(path, filename))
#data = np.load('2darray_logged.npz')
#data = np.load('2darray.npz')  #linear

fig, ax = plt.subplots(nrows=1, ncols=1)

# Each row is a different redshift so we want the transpose of the data
dm_data = data.T

# Normalise
dm_data = dm_data / np.sum(dm_data, axis=0)

# Mask any data point that has a value of less than 0.001.
dm_data = np.ma.masked_where(dm_data < 1e-3, dm_data)

# Import the rainforest colour map
#prism.import_cmaps("/Users/abatten/PhD")
cmap = plt.cm.rainforest_r

# Set all the masked pixels to white
cmap.set_bad(color='white')

#extents = [0, 3, 10000, 0.1]  # linear
#extents = [0, 3, 5, -1]
#extents2 = [0, 2.99, -1, 5]



#ymax = 5000   # linear
#ymin = 0.1    # linear
ymax = 4000
ymin = 0
xmax = 3.0
xmin = 0.0

aspect = (ymax - ymin) / (xmax-xmin)


bins = 10**np.linspace(-1, 5, 1000)

data_redshifts = np.array(
    [0.        , 0.02271576, 0.04567908, 0.06890121, 0.09239378,
     0.1161688 , 0.1402387 , 0.16461634, 0.18931503, 0.21434853,
     0.23973108, 0.26547743, 0.29160285, 0.31812316, 0.34505475,
     0.37241459, 0.40022031, 0.42849013, 0.45724301, 0.48649858,
     0.51627725, 0.54660017, 0.57748936, 0.60896766, 0.64105881,
     0.67378753, 0.7071795 , 0.74126146, 0.77606125, 0.81160784,
     0.84793147, 0.88506357, 0.92303701, 0.96188603, 1.00164638,
     1.04235539, 1.08405204, 1.12677711, 1.1705732 , 1.21548489,
     1.26155887, 1.30884399, 1.35739143, 1.40725491, 1.45849065,
     1.51115776, 1.56531826, 1.62103729, 1.6783833 , 1.73742834,
     1.79824817, 1.86092257, 1.92553561, 1.99217588, 2.06093685,
     2.13191717, 2.20522104, 2.2809586 , 2.35924626, 2.44020723,
     2.52397206, 2.61067901, 2.70047481, 2.79351511, 2.88996519, 
     2.99000085])

#im = plot_tools.plot_2d_array(dm_data, extents=extents2, cmap=cmap, passed_ax=ax)

#im = plot_tools.plot_2d_array(10**dm_data, xvals=data_redshifts, yvals=bins, cmap=cmap, passed_ax=ax)
#im = ax.imshow(dm_data, aspect=aspect, cmap=cmap, interpolation='none', extent=extents)


add_mean_median_mode_data = False
mean_median_mode_path = "mean_median_mode_interpolated_redshift.txt"
mean_median_mode_summed_path = "mean_median_mode_interpolated_redshift_summed.txt"


if add_mean_median_mode_data:
    data = ascii.read(mean_median_mode_path)
    data_summed = ascii.read(mean_median_mode_summed_path)

    mean_sum = np.zeros(65)
    mode_sum = np.zeros(65)
    median_sum = np.zeros(65)

    for j, _ in enumerate(data):
        median_sum[j] = np.sum(data[:j]["Median"])
        mean_sum[j] = np.sum(data[:j]["Mean"])
        mode_sum[j] = np.sum(data[:j]["Mode"])


    ax.plot(data[:]["Redshift"], np.log10(mean_sum), linestyle=":", linewidth=2, label="$\mathrm{Cumulative\ Mean}$", color="blueviolet")
    ax.plot(data[:]["Redshift"], np.log10(median_sum), linestyle=":", linewidth=2, label="$\mathrm{Cumulative\ Median}$", color="maroon")
    ax.plot(data[:]["Redshift"], np.log10(mode_sum), linestyle=":", linewidth=2, label="$\mathrm{Cumulative\ Mode}$", color="black")
    ax.plot(data_summed[:]["Redshift"], np.log10(data_summed[:]["Mean"]), linestyle="-", linewidth=2, label="$\mathrm{Summed\ Mean}$", color="magenta")
    ax.plot(data_summed[:]["Redshift"], np.log10(data_summed[:]["Median"]), linestyle="--", linewidth=2, label="$\mathrm{Summed\ Median}$", color="cyan")
    ax.plot(data_summed[:]["Redshift"], np.log10(data_summed[:]["Mode"]), linestyle="-.", linewidth=2, label="$\mathrm{Summed\ Mode}$", color="orange")


ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
ax.set_xlabel(r"Redshift")
#ax.set_ylabel(r"$\rm{log_{10}\ DM}\ \left[pc\ cm^{-3}\right]$")
ax.set_ylabel(r"$\rm{DM}\ \left[pc\ cm^{-3}\right]$")
#ax.set_ylabel(r"$\rm{DM}\ \left[pc\ cm^{-3}\right]$")   # linear
dm_vals = 10**np.linspace(1, 3.8, 1000)

method_list = methods.available_methods()

#colours = ['#66c2a5','#fc8d62','#8da0cb']
#colours = ["#66c2a5", "#e7298a", "#8da0cb"]
line_colours = ['#d95f02','#377eb8', '#e7298a']
label = [r"$\rm{Ioka\ (2003)}$", r"$\rm{Inoue\ (2004)}$", r"$\rm{Zhang\ (2018)}$"]
lstyle = ["-", "-", "-"]

# Add the Bannister FRB
for frb in frb_obs_list:
    redshift = frb["z"]
    dm = frb["DM"] - frb["DM_MW"]
    #plt.scatter(redshift, np.log10(dm), marker='o', label=frb["Label"])


if plot_ioka:
    for j, method in enumerate(method_list):
        cosmology = 'EAGLE'
    
        table_name = "".join(["_".join([method, cosmology]), ".npz"])
        lookup_table = table.load(table_name)
    
        # Calculate the z_vals from the analytical relations
        z_vals = np.zeros(len(dm_vals))
        for i, dm in enumerate(dm_vals):
            z_vals[i] = table.get_z_from_table(dm, lookup_table)
    
        # Overplot the analytical relations
#        ax.plot(z_vals, np.log10(dm_vals), line_colours[j],
#            linestyle=lstyle[j], label=label[j], linewidth=2, alpha=0.6)
        ax.plot(z_vals, dm_vals, line_colours[j],
            linestyle=lstyle[j], label=label[j], linewidth=2, alpha=0.6)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        print("HERE")
        print(z_vals)
        print(dm_vals)
        legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
        #plt.setp(legend.get_texts(), color='k')
        plt.xlim(0, 3)
        plt.tight_layout()
        plt.savefig(f"EAGLE_{j}.png", dpi=150)


dolag_data_path = "/fred/oz071/abatten/Dolag_DMz/dolag.a10.out"
dolag_err_path = "/fred/oz071/abatten/Dolag_DMz/"
if add_dolag_data:
    x, y = np.genfromtxt(dolag_data_path, unpack=True,usecols=[0,1])
    x_vals, y_errs = np.genfromtxt(dolag_err_path + "dolag.b10.out",unpack=True,usecols=[0,1])
    y_low = y - y_errs[0:8]
    y_high = y_errs[8:] - y
    
    y_errs = np.array([(yl, yh) for yl,yh in zip(y_low, y_high)]).T
    print(y_errs)


    ax.errorbar(x, y, yerr=y_errs, color="black", marker="^", ls="none", label="Dolag+(2015)") 
    #ax.plot(x, y, color="black", linestyle="--", label="Dolag+(2015)") 


if plot_pol_data:
    pol_reds_free, pol_dms_free = np.genfromtxt("Pol2019.data", unpack=True)
    ax.plot(pol_reds_free, pol_dms_free, color="red", linestyle="--", label="Pol et al. (2019)")

if plot_mcquinn_data:
    mcquinn_reds_free, mcquinn_dms_free = np.genfromtxt("McQuinn2014.data", unpack=True)
    ax.plot(mcquinn_reds_free, mcquinn_dms_free, color="darkorange", linestyle="--", label="McQuinn (2014)")


if add_illustris_data:
    ill_reds_igm, ill_dms_igm = np.genfromtxt("Jaroszynski19_IGM.data", unpack=True)
    #ax.plot(ill_reds_igm, np.log10(ill_dms_igm), color="purple", linestyle=":", label="Jaroszynski (2019) IGM")
    #ax.plot(ill_reds_igm, np.log10(ill_dms_igm), color="purple", linestyle=":", label="Jaroszynski (2019) IGM")
    ill_reds_free, ill_dms_free = np.genfromtxt("Jaroszynski19_All_Free_Electrons.data", unpack=True)
    #ax.plot(ill_reds_free, np.log10(ill_dms_free), color="purple", label="Jaroszynski (2019) All Free")
    print("Redshift", ill_reds_free)
    print("DMs", ill_dms_free)
    ax.plot(ill_reds_free, ill_dms_free, color="purple", linestyle="--", label="Jaroszynski (2019)")


# Set up the colour bar
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
legend = ax.legend(frameon=False, fontsize=8, loc="lower right")4
#plt.setp(legend.get_texts(), color='k')
plt.xlim(0, 3)
#cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{PDF}$")
#cbar.ax.tick_params(axis='y', direction='out')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
#plt.savefig("RefL0025N0376_DM_z_relation_bannister_Dolag_Jaro_LINES.png", dpi=300)
plt.savefig("EAGLE_DMz_relation_model_inoue_mcquinn_dolag_Jaro_Pol.png", dpi=150)
#plt.savefig("postimage_linear_rainforest_lines_colorbar_low_alpha.png", dpi=800)  # linear
