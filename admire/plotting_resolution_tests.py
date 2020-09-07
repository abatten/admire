import os
import numpy as np
from model import DMzModel
    
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
from scipy.integrate import quad

import cmasher
import e13tools
from glob import glob


from pyx import plot_tools
from pyx import print_tools

from model import DMzModel

#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=16)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=18)
plt.rc('axes', labelsize=18)




def plot_resolution_tests():
    file1 = "RefL0025N0376_Log_Normal_Fit_Mean_Std.txt"
    file2 = "RefL0025N0752_Log_Normal_Fit_Mean_Std.txt"
    file3 = "RecalL0025N0752_Log_Normal_Fit_Mean_Std.txt"
    file4 = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"
    file5 = "RefL0050N0752_log_normal_fit_mean_std_percent.txt"
    vartest = "/home/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS/correlation_varience_test.txt"

    var_data = np.loadtxt(vartest, skiprows=1, unpack=True)
    redshift = var_data[0]
    std_cumsum = var_data[5]



    z1, mean1, std1, per1 = np.loadtxt(file1, unpack=True, skiprows=1)
    z2, mean2, std2, per2 = np.loadtxt(file2, unpack=True, skiprows=1)
    z3, mean3, std3, per3 = np.loadtxt(file3, unpack=True, skiprows=1)
    z4, mean4, std4, per4 = np.loadtxt(file4, unpack=True, skiprows=1)
    z5, mean5, std5, per5 = np.loadtxt(file5, unpack=True, skiprows=1)

    var_data = np.loadtxt(vartest, skiprows=1, unpack=True)
    redshift = var_data[0]
    std_cumsum = var_data[5]

    filename = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"
    redshift, mean, std, percent = np.loadtxt(filename, unpack=True, skiprows=1)
    
    fig = plt.figure(figsize=(16,12), constrained_layout=True)
    gs = fig.add_gridspec(7, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0:4, 0])
    ax2 = fig.add_subplot(gs[4:5, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[5:6, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[6:7, 0], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    #ax1.spines['bottom'].set_linewidth(0)
    
    
    w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
    #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
    
    
    axin = ax1.inset_axes([0.55, 0.1, 0.35, 0.35])
    axin.set_xlim(2.6, 3)
    axin.set_ylim(2600, 3000)
    axin.xaxis.set_tick_params(labelsize=8)
    axin.yaxis.set_tick_params(labelsize=8)
    # Fit the Data with linear and Non-linear
    #fit_linear = curve_fit(linear, redshift, mean, bounds=([0, 0], [1200, 1e-10]))
    #fit_fz = curve_fit(f_function, redshift, mean)
    
    colours = np.array(list(map(mpl.colors.to_hex, cmasher.rainforest(np.linspace(0.20, 0.80, 4)))))[[3, 2, 1, 0]]
    colours = np.append(colours, "#000000")
    
    #mean_linear_fit = linear(redshift, fit_linear[0][0], fit_linear[0][1])
    #mean_fz_fit = f_function(redshift, fit_fz[0][0])
    #mean_linear_1000 = linear(redshift, 1000, 0)
    #mean_linear_1200 = linear(redshift, 1200, 0)
    #mean_linear_855 = linear(redshift, 855, 0)
    ax1.plot(z1, mean1, color=colours[0], label="$\mathrm{RefL0025N0376}\ <\mathrm{DM}>$", linewidth=2)
    ax1.plot(z2, mean2, color=colours[1], label="$\mathrm{RefL0025N0752}\ <\mathrm{DM}>$", linewidth=2)
    ax1.plot(z5, mean5, color=colours[3], label="$\mathrm{RefL0050N0752}\ <\mathrm{DM}>$", linewidth=2)
    ax1.plot(z3, mean3, color=colours[2], label="$\mathrm{RecalL0025N0752}\ <\mathrm{DM}>$", linewidth=2)
    ax1.plot(z4, mean4, color=colours[4], label="$\mathrm{RefL0100N1504}\ <\mathrm{DM}>$", linewidth=2)

    axin.plot(z1, mean1, color=colours[0], linewidth=2)
    axin.plot(z2, mean2, color=colours[1], linewidth=2)
    axin.plot(z3, mean3, color=colours[2], linewidth=2)
    axin.plot(z4, mean4, color=colours[4], linewidth=2)
    axin.plot(z5, mean5, color=colours[3], linewidth=2)
    
    
    #colours = ["#67a9cf", "#ef8a62"]
    #colours = ["#0571b0", "#ca0020"]
    
    #ax1.plot(redshift, mean_fz_fit, color=colours[0], linewidth=3, label=r'$\mathrm{Non\textnormal{-}linear\ fit:\ <DM>} = \alpha F(z); \alpha=923.47$')
    #ax1.plot(redshift, mean_linear_fit, colours[1], linewidth=3, label=r'$\mathrm{Linear\ fit:\ <DM>} = \beta z; \beta=1004.4$')
    #ax1.plot(redshift, mean_linear_1000, colours[1], linestyle=":", linewidth=2, label=r'$\mathrm{<DM>} = 1000 z$')
    #ax1.plot(redshift, mean_linear_1200, colours[1], linestyle="--", linewidth=1, label=r'$\mathrm{<DM>} = 1200 z$')
    #ax1.plot(redshift, mean_linear_855, colours[1], linestyle="-.", linewidth=1, label=r'$\mathrm{<DM>} = 855 z$')
    
    #axin.plot(redshift, mean_fz_fit, colours[0], linewidth=3)
    #axin.plot(redshift, mean_linear_fit, colours[1], linewidth=3)
    #axin.plot(redshift, mean_linear_1000, colours[1], linestyle=":", linewidth=2)
    #axin.plot(redshift, mean_linear_1200, colours[1], linestyle="--", linewidth=1)
    #axin.plot(redshift, mean_linear_855, colours[1], linestyle="-.", linewidth=1)
    
    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linestyle=":", linewidth=1)
    #ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linewidth=1)
    ax2.plot(redshift, mean4 - mean1[::4], color=colours[0], linestyle='-', linewidth=3, label="RefL0100N1504 - RefL0025N0376")
    ax2.plot(redshift, mean4 - mean5[::2], color=colours[3], linestyle='-', linewidth=3, label="RefL0100N1504 - RefL0050N0752")
    #ax2.plot(redshift, mean - mean_linear_fit, color=colours[1], linestyle="-", linewidth=3)
    #ax2.plot(redshift, mean - mean_linear_1000, colours[1], linestyle=":", linewidth=2)
    #ax2.plot(redshift, mean - mean_linear_1200, colours[1], linestyle="--", linewidth=1)
    #ax2.plot(redshift, mean - mean_linear_855, colours[1], linestyle="-.", linewidth=1)
    
    ax3.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linestyle=":", linewidth=1)
    ax3.plot(z2, mean2 - mean1, color=colours[1], linestyle='-', linewidth=3)
    
    ax4.plot(np.linspace(-1, 4, 100), np.zeros(100), "black", linestyle=":", linewidth=1)
    ax4.plot(z3, mean3 - mean2, color=colours[2], linestyle='-', linewidth=3)
    
    ax1.set_ylabel("$\mathrm{Expectation\ Value}$\n$<\mathrm{DM}>\ \left[\mathrm{pc\ cm^{-3}}\\right]$")
    ax2.set_ylabel("$\mathrm{Residuals}$\n $\left[\mathrm{pc\ cm^{-3}}\\right]$", multialignment='center')
    ax3.set_ylabel("$\mathrm{Residuals}$\n $\left[\mathrm{pc\ cm^{-3}}\\right]$", multialignment='center')
    ax4.set_ylabel("$\mathrm{Residuals}$\n $\left[\mathrm{pc\ cm^{-3}}\\right]$", multialignment='center')
    ax4.set_xlabel("Redshift")

    
    ax2.set_ylim(-200, 200)
    ax3.set_ylim(-100, 100)
    ax4.set_ylim(-100, 100)
    ax2.set_xlim(-0.04, 3.04)
    ax1.legend(frameon=False, fontsize=14)
    ax2.legend(frameon=False, fontsize=14)
    #plt.tight_layout()
    ax1.indicate_inset_zoom(axin)
    #plt.savefig('dmz_relation_RefL0100N1504_mean_fit_linear_fz.png', dpi=175)


#    fig = plt.figure(constrained_layout=True, figsize=(8,6))
#    gs = fig.add_gridspec(8, 16, wspace=0.0, hspace=0.0)
#    ax1 = fig.add_subplot(gs[0:9, :9])
#    ax2 = fig.add_subplot(gs[0:2, 9:], sharex=ax1)
#    ax3 = fig.add_subplot(gs[3:5, 9:], sharex=ax2)
#    ax4 = fig.add_subplot(gs[6:8, 9:], sharex=ax2)
#
#    ax2_2 = ax2.twinx()
#    ax3_2 = ax3.twinx()
#    ax4_2 = ax4.twinx()
#
#    plt.setp(ax2_2.get_yticklabels(), visible=False)
#    plt.setp(ax3_2.get_yticklabels(), visible=False)
#    plt.setp(ax4_2.get_yticklabels(), visible=False)
#
#
#    #ax1.spines['bottom'].set_linewidth(0)
#
#    ax1.set_xlabel("Redshift")
#    ax1.set_ylabel(r"$\mathrm{<DM>\ [pc\ cm^{-3}]}$")
#
#
#    ax2_2.set_ylabel("Volume Resolution")
#    ax3_2.set_ylabel("Particle Resolution")
#    ax4_2.set_ylabel("Physics Calibration")
#
#
#    w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
#    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
#
#
#    RefL25N752 = models[0]
#    RecalL25N752 = models[1] 
#
#   
    #plt.tight_layout()
    plt.savefig("TESTING_RESOLUTION_TESTS.png")



if __name__ == "__main__":    
    
    
    plot_resolution_tests()
    #plot_dmz_relation(all_models, "dmz_relation_full_RecalL0025N0752", "")

    print_tools.print_footer()


