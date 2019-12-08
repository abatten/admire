import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmasher
import e13tools
from glob import glob


from pyx import plot_tools
from pyx import print_tools

from model import DMzModel




#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=20)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=20)
plt.rc('axes', labelsize=20)


def plot_dmz_relation(models, plot_output_name=None, plot_output_path=None, 
                      plot_output_format=".png", logy=False, 
                      z_min=0, z_max=3.0, dm_min=0, dm_max=4000,
                      axis=None, verbose=True):
    """
    Plots the DM-z Relation
    """

    # Create an figure instance, but use a passed axis if given.
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)


    # Plot the 2D histogram as an image in the backgound
    for model in models:
        if model.category == '2D-hydrodynamic' and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)
            
            # Create a 2D image with different x-y bin sizes.
            im = plot_tools.plot_2d_array(
                model.Hist, 
                xvals=model.z_bins,
                yvals=model.DM_bins,
                cmap=model.color,
                passed_ax=ax
            )

            # Set up the colour bar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{PDF}$")
            cbar.ax.tick_params(axis='y', direction='out')


        else:
            pass

    # Plot the 1D models over the top.
    for model in models:
        if model.category[:2] == "1D" and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)

#in ["1D-analytic", "1D-semi-analytic", "1D-hydrodynamic"]:

            # If the model has a marker, use an errorbar plot.
            if model.marker is not None:
                pass

            # If the model has a linestyle, use a line plot.
            elif model.linestyle is not None:
                ax.plot(model.z_vals, model.DM_vals, color=model.color, 
                        linestyle=model.linestyle, linewidth=model.linewidth,
                        alpha=0.6, label=model.label)


    ax.set_xlim(z_min, z_max)
    ax.set_ylim(dm_min, dm_max)

    # This forces the x-tick labels to be integers for consistancy.
    # This fixes a problem I was having where it changed from ints to floats 
    # seemingly randomly for different number of models. 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(r"$\rm{Redshift}$")
    ax.set_ylabel(r"$\rm{DM\ \left[pc\ cm^{-3}\right] }$")
    ax.legend(frameon=False, fontsize=10, loc="lower right" )


    plt.tight_layout()
    output_file_name = os.path.join(plot_output_path, f"{plot_output_name}{plot_output_format}")
    plt.savefig(output_file_name, dpi=200)



if __name__ == "__main__":
    print_tools.print_header("DM-z Relation")

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
        "dir_name"     : "/home/abatten/admire/admire/DM_redshift_models",
        "file_name"    : "ioka2003_EAGLE_DM_min0_max10000_model.txt" ,
        "label"        : "Ioka (2003)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : "r",
        "linestyle"    : "-",
        "linewidth"    : 2.0,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    
    # INOUE (2004)
    inoue2004 = {
        "dir_name"     : "/home/abatten/admire/admire/DM_redshift_models",
        "file_name"    : "inoue2004_EAGLE_DM_min0_max10000_model.txt",
        "label"        : "Inoue (2004)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : "b",
        "linestyle"    : "-",
        "linewidth"    : 2.0,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    # ZHANG (2018)
    zhang2018 = {
        "dir_name"     : "/home/abatten/admire/admire/DM_redshift_models",
        "file_name"    : "zhang2018_EAGLE_DM_min0_max5000_model.txt",
        "label"        : "Zhang (2018)",
        "file_format"  : "txt",
        "category"     : "1D-analytic",
        "dm_scale"     : "linear",
        "color"        : "g",
        "linestyle"    : "-",
        "linewidth"    : 2.0,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    # McQuinn (2014)
    mcquinn2014 = {
        "dir_name"     : "/home/abatten/admire/admire/DM_redshift_models",
        "file_name"    : "mcquinn2014_model.txt",
        "label"        : "McQuinn (2014)",
        "file_format"  : "txt",
        "category"     : "1D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : "purple",
        "linestyle"    : "--",
        "linewidth"    : 2.0,
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
        "color"        : "blue",
        "linestyle"    : "--",
        "linewidth"    : 2.0,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    # Pol et al. (2019)
    pol2019 = {
        "dir_name"     : "/home/abatten/admire/admire/DM_redshift_models",
        "file_name"    : "pol2019_model.txt",
        "label"        : "Pol et al. (2019)",
        "file_format"  : "txt",
        "category"     : "1D-semi-analytic",
        "dm_scale"     : "linear",
        "color"        : "orange",
        "linestyle"    : "-.",
        "linewidth"    : 2.0,
        "marker"       : None,
        "plot_toggle"  : False,
    }

    batten2020 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RecalL0025N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed.hdf5",
        "label"        : "Batten (2020) RecalL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.rainforest_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    model_dicts = [
        ioka2003, 
        inoue2004, 
        zhang2018, 
        mcquinn2014, 
        dolag2015, 
        pol2019, 
        batten2020,
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
    
    plot_dmz_relation(all_models, "dmz_relation_full_RecalL0025N0752", "")

    print_tools.print_footer()

