import numpy as np
from pyx import plot_tools
from pyx import print_tools
import os
import scipy.interpolate as interp

from model import DMzModel

class cdf_sampler(object):

    # __init__ computed the normalisation
    # factor then constructs the cumulative
    # distribution function (cdf).
    def __init__(self,x,y):
        self.x_input = x
        self.y_input = y
        self.sample  = 'None'

        pdf_fnorm = np.sum(y)
        self.cdf = np.cumsum(y/pdf_fnorm)

    # sample_n in produces a random sample
    # of n with a distribution matched to the
    # input array, y.
    def sample_n(self, n):

        self.sample = np.zeros(n)
        for i in range(n):
            tm = np.random.uniform()
            tt = np.where(np.abs(tm-self.cdf) == np.min(np.abs(tm-self.cdf)))
            if len(tt[0]) > 1:
                bi = np.random.binomial(1,.5)
                bi = int((-1.)*bi)
                self.sample[i] = self.x_input[tt[0][bi]]

            else:
                self.sample[i] = self.x_input[tt[0]]



def calc_frbs(models):

    num_frbs = [5, 10, 30, 50, 100, 500, 1000, 5000]



    for model in models:

        z_bins = model.z_bins
        dm_bins = model.DM_bins

        redshift1_idx = np.where(z_bins > 1)[0][0]

        pdf = model.Hist.T[redshift1_idx]

        sampler = cdf_sampler(dm_bins, pdf)


        frb_list = []
        for num in num_frbs:
            sampler.sample_n(num)
            mean = np.mean(sampler.sample)
            std = np.std(sampler.sample)
            frb_list.append(f"{num}, {std:.2f}, {std / np.sqrt(2 * num - 2):.2f}")

        print(frb_list, "\n")










if __name__ == "__main__":
    print_tools.print_header("DM-z Relation")
    battenNoAGN = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/shuffled_output",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",
        "label"        : "NoAGN",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    battenAGNdT9 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/shuffled_output",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_AGNdT9L0050N0752/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",
        "label"        : "AGNdT9",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }


    model_dicts = [
        battenNoAGN,
        battenAGNdT9,
    ]


    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)

    calc_frbs(all_models)

    print_tools.print_footer()