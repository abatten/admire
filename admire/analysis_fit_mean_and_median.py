import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy import stats
from glob import glob
import os
import lmfit
from pyx import print_tools

from scipy.integrate import quad

def lmfit_straight_line(pars, xvals, yvals=None, yerr=None):
    slope, intercept = pars['slope'], pars['intercept']

    model = slope * xvals + intercept

#    slope = pars['slope']

#    model = slope * xvals

    if yvals is None:
        return model
    else:
        return (model - yvals) / yerr


def lmfit_linear_chisq(params, xdata, ydata=None, yerr=None):
    slope = params["slope"].value
    intercept = params["intercept"].value

    model = slope * xdata + intercept

    if ydata is not None:
        if yerr is not None:
            return (ydata - model) / yerr
        else:
            return (ydata - model)
    else:
        return model

def fz_function(alpha, z):

    fz_array = np.zeros_like(z)

    def integrand(z, OmegaL=0.693, OmegaM=0.307):
        top = 1 + z
        bot = np.sqrt(OmegaM * (1+z)**3 + OmegaL)
        return top/bot

    for idx, z in enumerate(z):
        fz_array[idx] = quad(integrand, 0, z)[0]

    return alpha * fz_array


def lmfit_non_linear(params, xdata, ydata, yerr=None):
    alpha = params["alpha"].value

    model = fz_function(alpha, xdata)

    if ydata is not None:
        if yerr is not None:
            return (ydata - model) / yerr
        else:
            return (ydata - model)
    else:
        return model


def chi_square_linear(initial_guess, x, y, yerr):
    a, b = initial_guess

    return np.sum((y - (a * x + b))**2 / yerr**2)


def straight_line(x, m, b):
    return m * x + b


def lmfit_reviewer_model(params, xdata,ydata, yerr=None):
    A = params["A"].value
    C = params["C"].value

    model = A * xdata + C * xdata**2

    if ydata is not None:
        if yerr is not None:
            return (ydata - model) / yerr
        else:
            return (ydata - model)
    else:
        return model



def fit_non_linear_model(data_files, output_file, init_guess=(1000), stat_type="mean"):
    idx_dict = {
        "Redshift": 0,
        "Mean": 1,
        "Median": 2,
        "SigmaG": 4,
        "CIWidth": 7,
        "SigmaCI": 8,
    }
    output = open(output_file, "w")

    output.write(
        "{:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "SimName",
            "Alpha",
            "Alpha_Err",
            "ChiSq",
            "DoF",
            "RedChiSq",
            "AIC",
            "BIC")
    )

    for filename in data_files:
        sim_name = os.path.basename(filename).split("_")[1]
        print(sim_name)

        data = np.loadtxt(filename, unpack=True, skiprows=2)

        redshift = data[idx_dict["Redshift"]]
        stat = data[idx_dict[stat_type]]
        stat_err = data[idx_dict["SigmaCI"]]

        params = lmfit.Parameters()
        params.add('alpha', value=1000)

        out_new = lmfit.minimize(lmfit_non_linear, params, args=(redshift, stat))


        output.write(
            "{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(
                sim_name,
                out_new.params["alpha"].value,
                out_new.params["alpha"].stderr,
                out_new.chisqr,
                out_new.nfree,
                out_new.redchi,
                out_new.aic,
                out_new.bic)
        )

    output.close()


def fit_reviewer_model(data_files, output_file, init_guess=(1000, 0), stat_type="mean"):
    idx_dict = {
        "Redshift": 0,
        "Mean": 1,
        "Median": 2,
        "SigmaG": 4,
        "CIWidth": 7,
        "SigmaCI": 8,
    }
    output = open(output_file, "w")

    output.write(
        "{:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "SimName",
            "A",
            "A_Err",
            "C",
            "C_Err",
            "ChiSq",
            "DoF",
            "RedChiSq",
            "AIC",
            "BIC")
    )
    for filename in data_files:
        sim_name = os.path.basename(filename).split("_")[1]
        print(sim_name)

        data = np.loadtxt(filename, unpack=True, skiprows=2)

        redshift = data[idx_dict["Redshift"]]
        stat = data[idx_dict[stat_type]]
        stat_err = data[idx_dict["SigmaCI"]]

        params = lmfit.Parameters()
        params.add('A', value=1000)
        params.add('C', value=1000)
        out_new = lmfit.minimize(lmfit_reviewer_model, params, args=(redshift, stat))


        output.write(
            "{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(
                sim_name,
                out_new.params["A"].value,
                out_new.params["A"].stderr,
                out_new.params["C"].value,
                out_new.params["C"].stderr,
                out_new.chisqr,
                out_new.nfree,
                out_new.redchi,
                out_new.aic,
                out_new.bic)
        )

    output.close()


def fit_linear_model(data_files, output_file, init_guess=(1000, 0), stat_type="mean"):
    idx_dict = {
        "Redshift": 0,
        "Mean": 1,
        "Median": 2,
        "SigmaG": 4,
        "CIWidth": 7,
        "SigmaCI": 8,
    }
    output = open(output_file, "w")

    output.write(
        "{:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "SimName",
            "Slope",
            "Slope_Err",
            "Inter",
            "Inter_Err",
            "ChiSq",
            "DoF",
            "RedChiSq",
            "AIC",
            "BIC")
    )

    for filename in data_files:
        sim_name = os.path.basename(filename).split("_")[1]
        print(sim_name)

        data = np.loadtxt(filename, unpack=True, skiprows=2)
        #mean = data[idx_dict["Mean"]]
        #redshift = data[idx_dict["Redshift"]]
        #mean_err = data[idx_dict["CIWidth"]] / 2


        redshift = data[idx_dict["Redshift"]]
        stat = data[idx_dict[stat_type]]
        stat_err = data[idx_dict["SigmaG"]]


        # result = minimize(chi_square_linear, init_guess, args=(redshift, mean, mean_err))

        # result2 = curve_fit(straight_line, redshift, mean, p0=init_guess, sigma=mean_err)
        # popt, pcov= result2

        # print(result["x"])

        # model = straight_line(redshift, result['x'][0], result['x'][1])
        # chisq1 = np.sum(((mean - model) / mean_err)**2)
        # print("chi2", chisq1)
        # print("redchi", chisq1 / (len(redshift) - 2))
        # print(popt)

        # model2 = straight_line(redshift, popt[0], popt[1])
        # chisq2 = np.sum(((mean - model2)/ mean_err)**2)
        # #print(chisq2)
        # #print(chisq2 / (len(redshift) ))



        # params = lmfit.Parameters()
        # params.add('slope', value=1000.0)
        # params.add('intercept', value=0.0, vary=False)

        # min1 = lmfit.Minimizer(lmfit_straight_line, params, fcn_args=(redshift,), fcn_kws={'yvals': mean, "yerr": mean_err})

        # out1 = min1.leastsq()
        # fit1 = lmfit_straight_line(out1.params, redshift)

        # print("lmfit", out1.params["slope"].value)
        # s = out1.params["slope"].value
        #i = out1.params["intercept"].value


        #output.write(f"{sim_name:<16} {s:<8.3f} {i:<8.3f}")

        #print("FIT1", lmfit.fit_report(out1))



        params = lmfit.Parameters()
        params.add('slope', value=1000)
        params.add('intercept', value=0.000, vary=True)


        out_new = lmfit.minimize(lmfit_linear_chisq, params, args=(redshift, stat))



        output.write(
            "{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(
                sim_name,
                out_new.params["slope"].value,
                out_new.params["slope"].stderr,
                out_new.params["intercept"].value,
                out_new.params["intercept"].stderr,
                out_new.chisqr,
                out_new.nfree,
                out_new.redchi,
                out_new.aic,
                out_new.bic)
        )

    output.close()




if __name__ == "__main__":
    print_tools.script_info.print_header("Fit Mean and Median Models")




    p0 = (1000, 0)

    p1 = (1000)

    all_files = sorted(glob("./analysis_outputs/shuffled/*mean_var_std_from_pdf*.txt"))
    all_files.reverse()


    mean_linear_output = "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_linear_model_least_squares.txt"
    median_linear_output = "./analysis_outputs/shuffled/ANALYSIS_fit_median_none_uncert_linear_model_least_squares.txt"

    fit_linear_model(all_files, init_guess=p0, stat_type="Mean", output_file=mean_linear_output)
    fit_linear_model(all_files, init_guess=p0, stat_type="Median", output_file=median_linear_output)

    mean_non_linear_output = "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_non_linear_model_least_squares.txt"
    median_non_linear_output = "./analysis_outputs/shuffled/ANALYSIS_fit_median_none_uncert_non_linear_model_least_squares.txt"

    fit_non_linear_model(all_files, init_guess=p1, stat_type="Mean", output_file=mean_non_linear_output)
    fit_non_linear_model(all_files, init_guess=p1, stat_type="Median", output_file=median_non_linear_output)

    mean_reviewer_output = "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_reviewer_model_least_squares.txt"
    median_reviewer_output = "./analysis_outputs/shuffled/ANALYSIS_fit_median_none_uncert_reviewer_model_least_squares.txt"

    fit_reviewer_model(all_files, init_guess=p0, stat_type="Mean", output_file=mean_reviewer_output)
    fit_reviewer_model(all_files, init_guess=p0, stat_type="Median", output_file=median_reviewer_output)


    print_tools.script_info.print_footer()
