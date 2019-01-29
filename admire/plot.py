import matplotlib.pyplot as plt
import os
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 18

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


def dmz_2dhist(zvals, dmvals, bins, density=True, passed_ax=None, fn=None, **kwargs):

    if passed_ax:
        ax = passed_ax
    else:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)


    ax.hist2d(zvals, dmvals, bins=50)
    ax.set_xlabel(r"$\rm{Redshift}$")
    ax.set_ylabel(r"$\rm{DM\ [pc\ cm^{-3}]}$")

    if passed_ax:
        return ax
    else:
        plt.tight_layout()
        plt.savefig("TESTING_MASTER_PLOT_DM.png")
        plt.close()




def projection(data, z, params, logfile=None, passed_ax=None):

    plt.imshow(data, cmap=params["ProjCmap"], vmax=params["ProjVmax"], 
               vmin=params["ProjVmin"])
    plt.tight_layout()

    output_file = "{0}_z_{1:.3f}.png".format(os.path.join(params["OutputDataDir"], 
                                                      params["ProjPlotName"]), z)
    if logfile:
        logfile.write("Creating projection plot: {0}".format(output_file))
                                                  
    plt.savefig(output_file)

