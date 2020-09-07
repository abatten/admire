import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy import stats
from glob import glob
import os
import lmfit
from pyx import print_tools

from scipy.integrate import quad


def lmfit_expon_chisq(pars, xvals, yvals):
    A = pars['coeff'].value
    B = pars['exponent'].value
    C = pars['intercept'].value

    model = A * np.exp(B * xvals) + C


    return yvals - model

def lmfit_expon2_chisq(pars, xvals, yvals):
    A = pars['coeff'].value
    B = pars['exponent'].value

    model = A * (1 - np.exp(B * xvals))


    return yvals - model


def fit_expon_model(data_files, output_file, init_guess=(1000, -1, 0), stat_type="sigmaVar"):
    idx_dict = {
        "Redshift": 0,
        "Mean": 1,
        "Median": 2,
        "SigmaVar": 4,
        "CIWidth": 7,
        "SigmaCI": 8,
    }
    output = open(output_file, "w")

    output.write(
        "{:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "SimName",
            "Coeff",
            "Coeff_Err",
            "Expon",
            "Expon_Err",
            "Inter",
            "Inter_Err",
            "ChiSq",
            "DoF",
            "RedChiSq",
            )
    )

    for filename in data_files:
        sim_name = os.path.basename(filename).split("_")[1]
        print(sim_name)

        data = np.loadtxt(filename, unpack=True, skiprows=2)

        redshift = data[idx_dict["Redshift"]]
        stat = data[idx_dict[stat_type]]


        params = lmfit.Parameters()
        params.add('coeff', value=1000)
        params.add('exponent', value=-1.000)
        params.add('intercept', value=0.000, vary=True)

        out_new = lmfit.minimize(lmfit_expon_chisq, params, args=(redshift, stat))



        output.write(
            "{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(
                sim_name,
                out_new.params["coeff"].value,
                out_new.params["coeff"].stderr,
                out_new.params["exponent"].value,
                out_new.params["exponent"].stderr,
                out_new.params["intercept"].value,
                out_new.params["intercept"].stderr,
                out_new.chisqr,
                out_new.nfree,
                out_new.redchi,)
        )

    output.close()


def fit_expon2_model(data_files, output_file, init_guess=(1000, -1), stat_type="sigmaVar"):
    idx_dict = {
        "Redshift": 0,
        "Mean": 1,
        "Median": 2,
        "SigmaVar": 4,
        "CIWidth": 7,
        "SigmaCI": 8,
    }
    output = open(output_file, "w")

    output.write(
        "{:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "SimName",
            "Coeff",
            "Coeff_Err",
            "Expon",
            "Expon_Err",
            "ChiSq",
            "DoF",
            "RedChiSq",
            )
    )

    for filename in data_files:
        sim_name = os.path.basename(filename).split("_")[1]
        print(sim_name)

        data = np.loadtxt(filename, unpack=True, skiprows=2)

        redshift = data[idx_dict["Redshift"]]
        stat = data[idx_dict[stat_type]]


        params = lmfit.Parameters()
        params.add('coeff', value=200)
        params.add('exponent', value=-1.000)


        out_new = lmfit.minimize(lmfit_expon2_chisq, params, args=(redshift, stat))


        output.write(
            "{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(
                sim_name,
                out_new.params["coeff"].value,
                out_new.params["coeff"].stderr,
                out_new.params["exponent"].value,
                out_new.params["exponent"].stderr,
                out_new.chisqr,
                out_new.nfree,
                out_new.redchi,)
        )

    output.close()




if __name__ == "__main__":
    print_tools.script_info.print_header("Fit Mean and Median Models")




    sigmaCI_p0 = (1000, -1, 0)
    sigmaVar_p0 = (1000, -1, 0)

    #sigmaCI_p0 = (1000, -1)
    #sigmaVar_p0 = (1000, -1)
    all_files = sorted(glob("./analysis_outputs/shuffled/*mean_var_std_from_pdf*.txt"))
    all_files.reverse()


    sigmaCI_expon_output = "./analysis_outputs/shuffled/ANALYSIS_fit_sigma_ci_none_uncert_expon_model_least_squares.txt"
    sigmaVar_expon_output = "./analysis_outputs/shuffled/ANALYSIS_fit_sigma_var_none_uncert_expon_model_least_squares.txt"

    fit_expon_model(all_files, init_guess=sigmaCI_p0, stat_type="SigmaCI", output_file=sigmaCI_expon_output)
    fit_expon_model(all_files, init_guess=sigmaVar_p0, stat_type="SigmaVar", output_file=sigmaVar_expon_output)

    print_tools.script_info.print_footer()
