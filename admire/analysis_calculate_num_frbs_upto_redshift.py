import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from scipy import stats
from pyx import print_tools
from pyx.print_tools import vprint, make_vprint
from model import DMzModel
from collections import Counter
from astropy import stats as astrostats
import matplotlib.pyplot as plt

rng = np.random.default_rng(5190)

class cdf_sampler(object):
    # __init__ computed the normalisation
    # factor then constructs the cumulative
    # distribution function (cdf).
    def __init__(self, x, y):
        self.x_input = x
        self.y_input = y
        self.sample  = 'None'

        pdf_fnorm = np.sum(y)
        self.cdf = np.cumsum(y / pdf_fnorm)

    # sample_n in produces a random sample
    # of n with a distribution matched to the
    # input array, y.
    def sample_n(self, n):
        self.sample = np.zeros(n)
        for idx in range(n):
            rvalue = rng.uniform()

            # Find where the random value intersects the cdf.
            cdf_idx = np.where(np.abs(rvalue - self.cdf) == np.min(np.abs(rvalue - self.cdf)))

            # If double peak distribution then there could be multiple
            # cdf_idx values, choose with binomial.
            if len(cdf_idx[0]) > 1:
                binom = np.random.binomial(1, 0.5)
                binom = int((-1.) * binom)
                self.sample[idx] = self.x_input[cdf_idx[0][binom]]
            else:
                self.sample[idx] = self.x_input[cdf_idx[0]]


def load_models(model_dicts):
    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)
    return all_models

def generate_N_frb_dms(N, dm_bins, pdf):
    sampler = cdf_sampler(dm_bins, pdf)
    sampler.sample_n(N)
    return sampler.sample


def generate_frb_dm(dm_bins, pdf):
    return generate_N_frb_dms(1, dm_bins, pdf)


def generate_N_redshifts(N, zmin, zmax, zbins):
    """

    Parameters
    ----------
    N : int
        The number of redshifts to generate.

    zmin : float
        The minimum redshift to sample.

    zmax : float
        The maximum redshift to sample.

    zbins : np.ndarray
        The redshift array of avaliable redshifts.

    Returns
    -------
    zidx: np.ndarray
        Length N. The index in zbins randomly generated.

    redshifts: np.ndarray
        Length N. The redshift in zbins randomly sampled.

    """

    z_idx = np.zeros(N, dtype=int)
    redshifts = np.zeros(N, dtype=float)

    for i in range(N):
        rfloat = rng.uniform(zmin, zmax, 1)[0]
        z_idx[i] = int(np.where(rfloat <= zbins)[0][0])
        redshifts[i] = zbins[z_idx[i]]

    return z_idx, redshifts

def anderson_darling_test(sample1, sample2):
    return stats.anderson_ksamp([sample1, sample2])




def perform_bootstrap_resampling(test_stat, bootnum):
    boot_resamples = astrostats.bootstrap(test_stat, bootnum=bootnum)
    boot_mean = np.mean(boot_resamples, axis=0)
    boot_std = np.std(boot_mean)
    return boot_mean, boot_std


if __name__ == "__main__":
    vprint = make_vprint(verbose=True)
    print_tools.print_header()

    #############
    # Load Models
    #############
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
        #AGNdT9,
    ]

    all_models = load_models(model_dicts)
    ######

    # Number of AD Tests & Bootstrap Resamplings
    num_tests_per_sample = 100

    frb_sample_sizes = np.array([5, 10, 30, 50, 100, 150, 200, 250, 300, 350, 
                                400, 450, 500, 600, 700, 800, 900, 1000,])
                                #2000, 3000, 4000, 5000, 6000, 7000, 8000, 
                                #9000, 10000])

    simulation_pairs = {
        "Ref-NoAGN": (all_models[0], all_models[1]),
        #Ref-AGNdT9": (all_models[0], all_models[2]),
        #"NoAGN-AGNdT9": (all_models[1], all_models[2]),
    }

    ad_stat_mean = np.empty(len(frb_sample_sizes))
    ad_stat_std = np.empty(len(frb_sample_sizes))
 
    #fig, ax = plt.subplots(constrained_layout=True)

    z_samp_min, z_samp_max = 0.5, 1

    # Iterate of the Model Pairs
    for key, value in simulation_pairs.items():
        vprint(f"Simulation Pair: {key}")

        # Create new output file per simulation pair
        output_filename = f"./analysis_outputs/{key}_num_frbs_upto_redshift.txt"
        output = open(output_filename, "w")
        header = "{:<15} {:<15}\n".format("Redshift", key)
        output.write(header)

        # Load both the models
        model1, model2 = value
        model1_pdf_arr = model1.Hist.T
        model2_pdf_arr = model2.Hist.T
        vprint("Length Model1 PDF:", len(model1_pdf_arr[0]))

        # Load the DM and Redshift Data
        # Same for both models
        z_bins = model1.z_bins
        dm_bins = model1.DM_bins

        for Nidx, N in enumerate(frb_sample_sizes):
            #N = int(N)  # Ensure that N is an int, because it will break otherwise.
            vprint(f"Number of FRBs: {N}")

            # Set up empty arrays
            model1_frbs = np.empty(N)
            model2_frbs = np.empty(N)
            test_results = np.empty(num_tests_per_sample)

            for test_idx in range(num_tests_per_sample):
                # Random Redshift Value
                z_idxs, redshifts = generate_N_redshifts(N, z_samp_min, z_samp_max, z_bins)

                for sample_idx in range(N):

                    # Extract the PDF from the models at the random redshift
                    model1_pdf = model1_pdf_arr[z_idxs[sample_idx]]
                    model2_pdf = model1_pdf_arr[z_idxs[sample_idx]]

                    # Sample the PDF 1 time
                    model1_frbs[sample_idx] = generate_frb_dm(dm_bins, model1_pdf)
                    model2_frbs[sample_idx] = generate_frb_dm(dm_bins, model2_pdf)

                # Perform AD test
                ad_test = anderson_darling_test(model1_frbs, model2_frbs)
                vprint("Model1 FRBs 1-5", model1_frbs[0:5])
                stat, crit_values, sig_level = ad_test
                test_results[test_idx] = stat

            # Calculate Mean and Std of test statistics
            test_results_mean = np.mean(test_results)
            test_results_std = np.std(test_results)

            # Bootstrap Resample the AD Test Statistics
            boot_mean, boot_std = perform_bootstrap_resampling(test_results, bootnum=num_tests_per_sample)

            ad_stat_mean[Nidx] = test_results_mean
            ad_stat_std[Nidx] = boot_std

            # Print out the results
            #vprint(test_results)
            vprint("Stat Mean", test_results_mean)
            vprint("Stat Std", boot_std)
            vprint("Crit Values", crit_values)
            vprint("\n")

            row = f"{N:<15.2f} {test_results_mean:<15.2f} {boot_std:<15.2f}\n"
            output.write(row)
        output.close()

        #ax.plot(frb_sample_numbers, ad_stat_mean, label=key)
        #ax.fill_between(frb_sample_numbers, ad_stat_mean - ad_stat_std, ad_stat_mean + ad_stat_std, alpha=0.2)

    #for i, crit in enumerate(crit_vals[:-1]):
    #    crit_labels = ["25%", "10%", "5%", "2.5%", "1%", "0.5%"]
    #    ax.axhline(crit, linestyle=":", alpha=0.3)
    #    ax.text(30, crit+0.1, crit_labels[i])

    #ax.set_ylim(-0.5, 5)
    #ax.legend(frameon=False, loc="lower right")
    #ax.set_xlabel("Number of FRBs")
    #ax.set_ylabel("Anderson Darling Test Statistic")

    #plt.savefig("./analysis_plots/shuffled/AD_Test_FRBs_Ref_NoAGN_z_sample.png")


    print_tools.print_footer()