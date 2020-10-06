import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
from matplotlib.legend_handler import HandlerPolyCollection
from matplotlib.collections import LineCollection
import matplotlib.ticker as mtick


import cmasher
import e13tools
from glob import glob

from astropy import cosmology
import astropy.units as apu

from pyx import plot_tools
from pyx import print_tools

from model import DMzModel


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=18)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc('ytick', labelsize=18)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=20)
plt.rc('axes', labelsize=20)
plt.rc("xtick", top=True)
plt.rc("ytick", right=True)

class HandlerColorPolyCollection(HandlerPolyCollection):
    def create_artists(self, legend, artist, xdescent, ydescent,
                        width, height, fontsize, trans):

        cmap = artist.cmap
        x = np.linspace(0, width, cmap.N)
        y = np.zeros(cmap.N) + height / 2 - ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, transform=trans)
        lc.set_array(x)
        lc.set_linewidth(5)
        return([lc])


def make_lookback_axis3(ax, cosmo, max_redshift):
    ax2 = ax.twiny()

    minor_tick_labels = np.arange(0, 13, 2)
    major_tick_labels = np.arange(1, 13, 2)

    major_tick_loc = np.zeros(len(major_tick_labels))
    minor_tick_loc = np.zeros(len(minor_tick_labels))

    for idx, labels in enumerate(zip(major_tick_labels, minor_tick_labels)):
        major, minor = labels

        if major < 0.0001:
            major_tick_loc[idx] = 0
        else:
            major_tick_loc[idx] = cosmology.z_at_value(cosmo.lookback_time, apu.Gyr * major) / max_redshift

        if minor < 0.0001:
            minor_tick_loc[idx] = 0
        else:
            minor_tick_loc[idx] = cosmology.z_at_value(cosmo.lookback_time, apu.Gyr * minor) / max_redshift



    ax2.set_xticks(major_tick_loc)
    ax2.set_xticks(minor_tick_loc, minor=True)

    major_tick_labels = [f"$\mathrm{x:.0f}$" for x in major_tick_labels]
    ax2.set_xticklabels(major_tick_labels, fontsize=14)

    return ax2


def plot_dmz_relation(models, plot_output_name="dmz_relation",
                      plot_output_path=None, plot_output_format=".png",
                      logy=False, z_min=0.0, z_max=3.0, dm_min=0, dm_max=4000,
                      axis=None, verbose=True):
    """
    Plots the DM-z Relation


    Parameters
    ----------
    models:

    plot_output_name: str, optional
        The name of the output plot. Default: "dmz_relation".

    plot_output_path: str, optional

    plot_output_format: str, optional

    z_min: int or float, optional
        The minimum Default: 0.0

    z_max: int or float, optional
        The maximum redshift (max x limit) for the

    """

    # Create an figure instance, but use a passed axis if given.
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

    # Plot the 2D histogram as an image in the backgound
    for model in models:
        if model.category == '2D-hydrodynamic' and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)

            # Create a 2D image with different x-y bin sizes.
            im = plot_tools.plot_2d_array(
                np.log10(model.Hist),
                xvals=model.z_bins,
                yvals=model.DM_bins,
                cmap=model.color,
                label="This Work",
                vmax=-2,
                vmin=-6,
                passed_ax=ax[0]
            )

            # Set up the colour bar
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{log_{10}(PDF)}$", extend='min')
            cbar = plt.colorbar(im, ax=ax[0], label="$\mathrm{log_{10}}~\mathrm{PDF}(z)$", extend='min', pad=0.005,     )
            cbar.ax.tick_params(axis='y', which='major', direction='out')
            cbar.ax.tick_params(axis='y', which='minor', length=0)

        elif model.category[:4] == "mean" and model.plot_toggle:
            ax[0].plot(model.z_vals, model.DM_vals,
                    color=model.color,
                    linestyle=model.linestyle,
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    label=model.label,
                    path_effects=[
                        pe.Stroke(linewidth=model.linewidth+1,
                                    foreground='k',
                                    alpha=model.alpha),
                        pe.Normal()]
                    )
            ax[1].plot(model.z_vals, model.DM_vals,
                    color=model.color,
                    linestyle=model.linestyle,
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    label=model.label,
                    )


        elif model.category[:3] == "std" and model.plot_toggle:
            ax[0].plot(model.z_vals, model.sigma3_upp,
                    color=model.color,
                    linestyle=model.linestyle,
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    label="$3~\sigma$"
                    )

            ax[0].plot(model.z_vals, model.sigma3_low,
                    color=model.color,
                    linestyle=model.linestyle,
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    )

            ax[0].plot(model.z_vals, model.sigma2_upp,
                    color=model.color,
                    linestyle="--",
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    label="$2~\sigma$"
                    )

            ax[0].plot(model.z_vals, model.sigma2_low,
                    color=model.color,
                    linestyle="--",
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    )

            ax[0].plot(model.z_vals, model.sigma1_upp,
                    color=model.color,
                    linestyle="-.",
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    label="$1~\sigma$"
                    )

            ax[0].plot(model.z_vals, model.sigma1_low,
                    color=model.color,
                    linestyle="-.",
                    linewidth=model.linewidth,
                    alpha=model.alpha,
                    )




        elif model.category[:2] == "1D" and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)

            # If the model has a marker, use an errorbar plot.
            if model.marker is not None:
                pass

            # If the model has a linestyle, use a line plot.
            elif model.linestyle is not None:
                ax[1].plot(model.z_vals, model.DM_vals,
                        color=model.color,
                        linestyle=model.linestyle,
                        linewidth=model.linewidth,
                        alpha=model.alpha,
                        label=model.label,
                        #path_effects=[
                        #    pe.Stroke(linewidth=model.linewidth+1,
                        #              foreground='k',
                        #              alpha=model.alpha),
                        #    pe.Normal()]
                       )

    ax[0].set_xlim(z_min, z_max)
    ax[0].set_ylim(dm_min, dm_max)
    ax[1].set_xlim(z_min, z_max)
    ax[1].set_ylim(dm_min, dm_max)
    # This forces the x-tick labels to be integers for consistancy.
    # This fixes a problem I was having where it changed from ints to floats
    # seemingly randomly for different number of models.
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_xlabel(r"$\rm{Redshift}$")
    ax[1].set_xlabel(r"$\rm{Redshift}$")
    ax[0].set_ylabel(r"$\rm{DM_{cosmic}\ \left[pc\ cm^{-3}\right] }$")
    ax[1].set_ylabel(r"$\rm{DM_{cosmic}\ \left[pc\ cm^{-3}\right] }$")

    handles, labels = ax[0].get_legend_handles_labels()
    #handles = [im, *reversed(handles)]
    #labels = [im.get_label(), *reversed(labels)]
    handles = [*reversed(handles[1:])]
    labels = [*reversed(labels[1:])]
    ax[0].legend(handles, labels, frameon=False, fontsize=16, loc="upper left", handlelength=2.5)
              #handler_map={
              #     im: HandlerColorPolyCollection()})

    handles, labels = ax[1].get_legend_handles_labels()
    handles = [*reversed(handles)]
    labels = [*reversed(labels)]
    #handles = [im, handles[0], *reversed(handles[1:])]
    #labels = [im.get_label(), labels[0], *reversed(labels[1:])]
    ax[1].legend(handles, labels, frameon=False, fontsize=16, loc="upper left", handlelength=2.5)



    p13 = cosmology.Planck13
    ax2 = make_lookback_axis3(ax[0], p13, z_max)
    ax2.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")

    ax2_2 = make_lookback_axis3(ax[1], p13, z_max)
    ax2_2.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")

    plt.tight_layout()
    output_file_name = os.path.join(plot_output_path, f"{plot_output_name}{plot_output_format}")
    plt.savefig(output_file_name, dpi=200)



if __name__ == "__main__":
    print_tools.print_header("DM-z Relation")

    output_file_name = "analysis_plots/shuffled/FBsigmaL0050N0752_Mean"
    #colours = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
    #colours = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
    #colours = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462']
    #colours = list(map(mpl.colors.to_hex, cmasher.heat_r(np.linspace(0.15, 0.70, 6))))
    colours = np.array(list(map(mpl.colors.to_hex, cmasher.heat(np.linspace(0.40, 0.90, 3)))))
    colours = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.20, 0.80, 7)))))[[0, 2, 4, 3, 5, 1, 6]]
    print(colours)


    colours = ["#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020", "#D39B85", "#817066"]
    #colours = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9']
    #colours = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc']
    #colours = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    #colours = ['#1b9e77','#d95f02','#666666','#e7298a','#66a61e','#e6ab02']
    #colours = ['b','g', 'r', 'c','m','y']
    #colours = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
    #colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']
    #colours = ['#332288','#117733','#88CCEE','#DDCC77','#AA4499']
    # MODEL DICT TEMPLATE
    #busted3000 = {
    #    "dir_name"     :
    #    "file_name"    :
    #    "label"        :
    #    "file_format"  :
    #    "category"     :
    #    "dm_scale"     :
    #    "color"        :
    #    "linestyle"    :
    #    "linewidth"    :
    #    "marker"       :
    #    "plot_toggle"  :
    #}

    # IOKA (2003)
    ioka2003 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "ioka2003_EAGLE_DM_min0_max10000_model.txt" ,
        "label"        : "Ioka (2003)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : colours[0],
        "linestyle"    : "-",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }


    # INOUE (2004)
    inoue2004 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "inoue2004_EAGLE_DM_min0_max10000_model.txt",
        "label"        : "Inoue (2004)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : colours[1],
        "linestyle"    : "-",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # ZHANG (2018)
    zhang2018 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "zhang2018_EAGLE_DM_min0_max5000_model.txt",
        "label"        : "Zhang (2018)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : colours[2],
        "linestyle"    : "-",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # McQuinn (2014)
    mcquinn2014 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "mcquinn2014_model.txt",
        "label"        : "McQuinn (2014)",
        "file_format"  : "txt",
        "category"     : "1D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : colours[5],
        "linestyle"    : "--",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # Dolag et al. (2015)
    dolag2015 = {
        "dir_name"     : "/fred/oz071/abatten/Dolag_DMz",
        "file_name"    : "dolag.a10.out",
        "label"        : "Dolag et al. (2015)",
        "file_format"  : "txt",
        "category"     : "1D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : colours[4],
        "linestyle"    : "--",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # Jaroszynski (2019)
    jaroszynski2019 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "jaroszynski2019_all_free_electrons_model.txt",
        "label"        : "Jaroszynski (2019)",
        "file_format"  : "txt",
        "category"     : "1D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : colours[3],
        "linestyle"    : "--",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # Pol et al. (2019)
    pol2019 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "pol2019_model_equation.txt",
        "label"        : "Pol et al. (2019)",
        "file_format"  : "txt",
        "category"     : "1D-semi-analytic",
        "dm_scale"     : "linear",
        "color"        : colours[6],
        "linestyle"    : ":",
        "linewidth"    : 2.5,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # Macquart et al. (2020)
    macquart2020 = {
        "dir_name"     : "/home/abatten/ADMIRE/admire/DM_redshift_models",
        "file_name"    : "macquart_2020_dm_cosmo_model.txt",
        "label"        : "Macquart et al. (2020)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : "green",
        "linestyle"    : "-",
        "linewidth"    : 1,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }


    batten2020_mean = {
        #"dir_name"     : "/home/abatten/ADMIRE/admire",
        "dir_name"     : "/home/abatten/ADMIRE/admire/analysis_outputs/shuffled",
        "file_name"    : "ANALYSIS_RefL0100N1504_mean_var_std_from_pdf.txt",
        "label"        : "$\mathrm{This\ Work}\ \left\langle\mathrm{DM_{cosmic}}\\right\\rangle$",
        "file_format"  : "txt",
        "category"     : "mean-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : "black",
        "linestyle"    : "-",
        "linewidth"    : 4,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    batten2020_conf_int = {
        #"dir_name"     : "/home/abatten/ADMIRE/admire",
        "dir_name"     : "/home/abatten/ADMIRE/admire/analysis_outputs/shuffled",
        "file_name"    : "ANALYSIS_RefL0100N1504_confidence_intervals.txt",
        "label"        : "$\left\langle\mathrm{DM}\\right\\rangle$",
        "file_format"  : "txt",
        "category"     : "std-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : "black",
        "linestyle"    : ":",
        "linewidth"    : 2,
        "alpha"        : 1,
        "marker"       : None,
        "plot_toggle"  : True,
    }


    batten2020 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/shuffled_output",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",
        "label"        : "Batten (2020) RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.ocean_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    batten2020_other = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/output/T4EOS",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DMnormed_bin_width_and_idx_corrected.hdf5",
        "label"        : "Batten (2020) RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RandGaussL0100 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/output/T4EOS",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0100",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0025",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RandGauss L0100",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    NoAGNL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "NoAGNL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    AGNdT9L0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_AGNdT9L0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "AGNdT9L0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    FBZL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_FBZL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "FBZL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    FBsigmaL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_FBsigmaL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "FBsigmaL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.ocean_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    FBconstL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_FBconstL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "FBconstL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.ocean_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }



    model_dicts = [

    #    macquart2020,


        pol2019,
        jaroszynski2019,
        dolag2015,
        mcquinn2014,
        zhang2018,
        inoue2004,
        ioka2003,
        batten2020_conf_int,
        batten2020_mean,

        #batten2020,
    #NoAGNL0050N0752,
    #AGNdT9L0050N0752,
    #FBZL0050N0752,
    FBsigmaL0050N0752,
    #FBconstL0050N0752,
    #    batten2020_other,
    #RandGaussL0100,
    ]

#######################################################################
#######################################################################
#                         DO NOT TOUCH
#######################################################################
#######################################################################

    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)

    plot_dmz_relation(all_models, output_file_name, "")

    print_tools.print_footer()


