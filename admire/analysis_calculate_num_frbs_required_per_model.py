import os, sys
import numpy as np
from scipy import stats
from pyx import print_tools
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




def load_models(model_dicts):
    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)

    return all_models

def generate_N_frbs(N, dm_bins, pdf):
    sampler = cdf_sampler(dm_bins, pdf)
    sampler.sample_n(N)

    return sampler.sample






def anderson_darling_test(sample1, sample2):
    return stats.anderson_ksamp([sample1, sample2])








if __name__ == "__main__":
    print_tools.print_header("DM-z Relation")

    Ref = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/shuffled_output/",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "NoAGN",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
    }

    NoAGN = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "NoAGN",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",

    }

    AGNdT9 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_AGNdT9L0050N0752/all_snapshot_data/shuffled_output/",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "AGNdT9",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",

    }


    model_dicts = [
        Ref,
        NoAGN,
        AGNdT9,
    ]


    all_models = load_models(model_dicts)

    frb_sample_numbers = 2**np.linspace(3, 13, 22)#np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000])

    simulation_pairs = {
        #"Ref-NoAGN": (all_models[0], all_models[1]),
        #"Ref-AGNdT9": (all_models[0], all_models[2]),
        "NoAGN-AGNdT9": (all_models[1], all_models[2]),
    }


    for key, value in simulation_pairs.items():
        print(key)

        sim1, sim2 = value

        z_bins = sim1.z_bins
        dm_bins = sim1.DM_bins
        print(z_bins)


        for i, z in enumerate(z_bins):
            sim1_pdf = sim1.Hist.T[i]
            sim2_pdf = sim2.Hist.T[i]

            for N in frb_sample_numbers:
                N = int(N)
                sim1_frbs = generate_N_frbs(N, dm_bins, sim1_pdf)
                sim2_frbs = generate_N_frbs(N, dm_bins, sim2_pdf)

                ad_test = anderson_darling_test(sim1_frbs, sim2_frbs)
                stat, crit_vals, sig_level = ad_test


                print(f"{z:<5.3f} {N:<5.3f} {sig_level:<8.5f} {stat:<8.3f}")


    #calc_frbs(all_models)

    print_tools.print_footer()