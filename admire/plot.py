import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 18

def _colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def dm_hist(data, bins, density=True, passed_ax=None, fn=None, **kwargs):
    if passed_ax:
        ax = passed_ax
    else:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

    ax.hist(data, bins=bins, density=density, **kwargs)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(r"$\rm{DM\ [pc\ cm^{-3}]}$")
    ax.set_ylabel(r"$\rm{PDF}$")

    if passed_ax:
        return ax
    else:
        if fn:
            plt.tight_layout()
            plt.savefig(fn)
            plt.close()
        else:
            plt.tight_layout()
            plt.savefig("DM_Histgram.png")
            plt.close()



def create_dm_pdf(z, data, bins, params, fit=False):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        ax = dm_hist(data, bins, passed_ax=ax)

        if fit:
            f, err = fit.lognormal(data, log, verb, boot=params["PerformBootstrap"])
            ax.plot(bins, stats.lognormal.pdf(bins, f[0], f[1], f[2]), "r--")
            ax.text(0.7, 0.80, r"\rm{shape} = {0:.3f}".format(f[0]), tramsform=ax.transAxes)
            ax.text(0.7, 0.75, r"\rm{loc} = {0:.3f}".format(f[1]), tramsform=ax.transAxes)
            ax.text(0.7, 0.70, r"\rm{scale} = {0:.3f}".format(f[2]), tramsform=ax.transAxes)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 1.5)
        ax.set_xlabel(r"$\rm{DM\ [pc\ cm^{-3}]}$")
        ax.set_ylabel(r"$\rm{PDF}$")

        plt.tight_layout()
        fn = "{0}/{1}_z_{2:.5f}.png".format(params["PlotDir"],
                                            params["DMPdfFileName"], z)
        plt.savefig(fn)
        plt.close()
        return fn



def dmz_2dhist(zvals, dmvals, bins, density=True, passed_ax=None, fn=None, **kwargs):

    if passed_ax:
        ax = passed_ax
    else:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)


    ax.hist2d(zvals, dmvals, bins=bins)
    ax.set_xlabel(r"$\rm{Redshift}$")
    ax.set_ylabel(r"$\rm{DM\ [pc\ cm^{-3}]}$")

    if passed_ax:
        return ax
    else:
        plt.tight_layout()
        plt.savefig("TESTING_MASTER_PLOT_DM.png")
        plt.close()




def coldens_map(data, z, params, passed_ax=None):

    if passed_ax:
        ax = passed_ax
    else:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

    Cmap = params["ColDensCmap"]
    Vmax = params["ColDensVmax"]
    Vmin = params["ColDensVmin"]
    half_width = 0.5 *params["DistSpacing"]
    extent = [-half_width, half_width, -half_width, half_width]

    im = ax.imshow(data, cmap=Cmap, vmax=Vmax, vmin=Vmin, extent=extent)
    #ax.set_xlim(- 0.5 * width, 0.5 * width)
    #ax.set_ylim(- 0.5 * width, 0.5 * width)
    #ax.set_xticklabels(np.linspace(-0.5*width, 0.5*width, 5))
    #ax.set_yticklabels(np.linspace(-0.5*width, 0.5*width, 5))
    ax.set_xlabel(r"$\rm{{x [{0}]}}$".format(params["DistUnits"]))
    ax.set_ylabel(r"$\rm{{y [{0}]}}$".format(params["DistUnits"]))

    cbar = _colorbar(im)
    cbar.ax.set_ylabel(r"$\rm{DM\ [pc\ cm^{-3}]}$")

    if passed_ax:
        return ax
    else:
        output_path = os.path.join(params["PlotDir"],
                                   params["ColDensMapFileName"])
        output_file = "{0}_z_{1:.3f}.png".format(output_path, z)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

