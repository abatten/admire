import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


from scipy.integrate import quad

import cmasher

from model import DMzModel

#the properties of the plot
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

def linear(x, m, b):
    """
    
    """
    return m * x + b

def f_function(z, m):
   
    fz_array = np.zeros(len(z))
    def integrand(z, OmegaL=0.7, OmegaM=0.3):
        top = 1 + z
        bot = np.sqrt(OmegaM * (1+z)**3 + OmegaL)
        return top/bot

    for idx, z in enumerate(z):
        fz_array[idx] = quad(integrand, 0, z)[0]
    return m * fz_array

def log(x, a, b):
    """
    """
    return a * np.log(x) + b


def inverted_parabola(x, a, b):
    return - a * x**2 + b


def power_law(x, a, b, c):
    return a * x**b + c

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b, c):
    return a * np.exp(b*x) + c

if __name__ == "__main__":
    #filename = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"
    #redshift, mean, std, percent = np.loadtxt(filename, unpack=True, skiprows=1)
    filename = "RefL0100N1504_mean_direct_values.txt"
    redshift, mean = np.loadtxt(filename, unpack=True, skiprows=1)



    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)


    w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
   

    axin = ax1.inset_axes([0.55, 0.1, 0.35, 0.35])
    axin.set_xlim(2.6, 3)
    axin.set_ylim(2600, 3000)
    axin.xaxis.set_tick_params(labelsize=12)
    axin.yaxis.set_tick_params(labelsize=12)
    # Fit the Data with linear and Non-linear
    fit_linear = curve_fit(linear, redshift, mean, bounds=([0, 0], [1200, 1e-10]))
    fit_fz = curve_fit(f_function, redshift, mean)


    mean_linear_fit = linear(redshift, fit_linear[0][0], fit_linear[0][1])
    mean_fz_fit = f_function(redshift, fit_fz[0][0])
    #mean_linear_1000 = linear(redshift, 1000, 0)
    #mean_linear_1200 = linear(redshift, 1200, 0)
    #mean_linear_855 = linear(redshift, 855, 0)
    ax1.plot(redshift, mean, color='black', label="$\mathrm{RefL0100N1504}\ \langle\mathrm{DM}\\rangle$", linewidth=4)
    axin.plot(redshift, mean, color='black', linewidth=6)


    #colours = ["#67a9cf", "#ef8a62"]
    colours = ["#0571b0", "#ca0020"]
    #colours = ["#000000", "#ca0020"]
    ax1.plot(redshift, mean_fz_fit, color=colours[0], linewidth=3, linestyle=":", label='$\mathrm{Non\\textnormal{-}linear\ fit:\ \langle DM \\rangle} = \\alpha F(z); \\alpha=%.2f$' % fit_fz[0][0])

    ax1.plot(redshift, mean_linear_fit, colours[1], linewidth=3, linestyle="--", label='$\mathrm{Linear\ fit:\ \langle DM \\rangle} = \\beta z; \\beta=%.2f$' % fit_linear[0][0])
    #ax1.plot(redshift, mean_linear_1000, colours[1], linestyle=":", linewidth=2, label=r'$\mathrm{<DM>} = 1000 z$')
    #ax1.plot(redshift, mean_linear_1200, colours[1], linestyle="--", linewidth=1, label=r'$\mathrm{<DM>} = 1200 z$')
    #ax1.plot(redshift, mean_linear_855, colours[1], linestyle="-.", linewidth=1, label=r'$\mathrm{<DM>} = 855 z$')

    axin.plot(redshift, mean_fz_fit, colours[0], linewidth=3, linestyle=":")
    axin.plot(redshift, mean_linear_fit, colours[1], linewidth=3, linestyle="--")
    #axin.plot(redshift, mean_linear_1000, colours[1], linestyle=":", linewidth=2)
    #axin.plot(redshift, mean_linear_1200, colours[1], linestyle="--", linewidth=1)
    #axin.plot(redshift, mean_linear_855, colours[1], linestyle="-.", linewidth=1)

    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linewidth=1)
    ax2.plot(redshift, mean - mean_fz_fit, color=colours[0], linestyle=':', linewidth=3)
    ax2.plot(redshift, mean - mean_linear_fit, color=colours[1], linestyle="--", linewidth=3)
    #ax2.plot(redshift, mean - mean_linear_1000, colours[1], linestyle=":", linewidth=2)
    #ax2.plot(redshift, mean - mean_linear_1200, colours[1], linestyle="--", linewidth=1)
    #ax2.plot(redshift, mean - mean_linear_855, colours[1], linestyle="-.", linewidth=1)
    
    ax1.set_ylabel("$\mathrm{Expectation\ Value}$\n$\langle\mathrm{DM}\\rangle\ \left[\mathrm{pc\ cm^{-3}}\\right]$")
    ax2.set_ylabel("$\mathrm{Residuals}$\n $\left[\mathrm{pc\ cm^{-3}}\\right]$", multialignment='center')
    ax2.set_xlabel("Redshift")

    
    ax2.set_ylim(-80, 80)
    ax2.set_xlim(-0.04, 3.04)
    ax1.legend(frameon=False, fontsize=14)
    plt.tight_layout()
    ax1.indicate_inset_zoom(axin)
    plt.savefig('dmz_relation_RefL0100N1504_mean_fit_linear_fz_2.png', dpi=175)
    plt.clf()
    print('<dm> fit', fit_linear[0])
    #print(fit2[0])
    
    #redshift, mean, std, percent = np.loadtxt(filename, unpack=True, skiprows=1)

    plt.plot(redshift, mean, 'blue', label="Data", linewidth=3)

    fit = curve_fit(f_function, redshift, mean)
    plt.plot(redshift, f_function(redshift, fit[0][0]), 'red', linewidth=2, label='Fit1')
    plt.ylabel(r"$<\mathrm{DM}>$")
    plt.xlabel("Redshift")
    plt.legend()
    plt.savefig('dmz_relation_RefL0100N1504_mean_fit_f_function.png')
    plt.clf()
    print("fz fit", fit[0])





    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)


    w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
   

    axin = ax1.inset_axes([0.55, 0.4, 0.35, 0.35])
    axin.set_xlim(2.6, 3)
    axin.set_ylim(200, 220)
    axin.xaxis.set_tick_params(labelsize=8)
    axin.yaxis.set_tick_params(labelsize=8)

    ax1.plot(redshift, std, 'black', linewidth=4, label="RefL0100N1504 $\sigma$")
    axin.plot(redshift, std, 'black', linewidth=4)
    
    fit = curve_fit(exponential, redshift, std)

    std_fit = exponential(redshift, fit[0][0], fit[0][1], fit[0][2]) 
    std_other_fit = exponential(redshift, -212, -1.00, 223)

    ax1.plot(redshift, std_fit, colours[0], linewidth=2, label='$\mathrm{Fit}\ \sigma = ae^{bz}+c; a=%.2f, b=%.2f, c=%.2f$' % (fit[0][0], fit[0][1], fit[0][2]))
    ax1.plot(redshift, std_other_fit, colours[1], linewidth=2, label='$\sigma = ae^{bz}+c; a=-212.00, b=-1.00, c=223.00$')
    
    axin.plot(redshift, std_fit, colours[0], linewidth=2, label='$\sigma = ae^{bz}+c; a=%.2f$, b=%.2f, c=%.2f$' % (fit[0][0], fit[0][1], fit[0][2]))
    axin.plot(redshift, std_other_fit, colours[1], linewidth=2, label='Test')

    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linewidth=1)
    ax2.plot(redshift, std - std_fit, linewidth=2, color=colours[0])
    ax2.plot(redshift, std - std_other_fit, linewidth=2, color=colours[1])

    ax2.set_ylim(-9.5, 9.5)
    ax2.set_xlim(-0.01, 3.01)
    #fit1 = curve_fit(power_law, redshift, std, bounds=([0, 0], [1000, 1e-5]))
    #plt.plot(redshift, power_law(redshift, fit1[0][0], fit1[0][1]), 'red', linewidth=2, label='Fit1')
    #plt.plot(redshift, exponential(redshift, fit[0][0], fit[0][1], fit[0][2]), 'green', linewidth=1, label='Fit2')
    #plt.plot(redshift, exponential(redshift, -206, -1.00, 221.5), 'red', linewidth=1, label='Test')
    ax1.set_ylabel("$\mathrm{Standard\ Deviation}$\n$\sigma\ \left[\mathrm{pc\ cm^{-3}}\\right]$")
    ax2.set_ylabel("$\mathrm{Residuals}$\n $\left[\mathrm{pc\ cm^{-3}}\\right]$", multialignment='center')
    ax2.set_xlabel("Redshift")
    ax1.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    ax1.indicate_inset_zoom(axin)
    plt.savefig('dmz_relation_RefL0100N1504_std_fit_nice.png')
    #print(fit1[0])
    #print(fit2[0])
    print('sigma fit', fit[0])
    #print(fit2[0])

