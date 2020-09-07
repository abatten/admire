import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher

from scipy.interpolate import interp1d
import cosmolopy as cp

from pyx import math_tools
import cmasher

cosmo = cp.fidcosmo

cosmo['omega_b_0'] = 0.04825
cosmo['omega_M_0'] = 0.307
cosmo['omega_lambda_0'] = 0.693
cosmo['h'] = 0.6777
cosmo['n'] = 0.9611
cosmo['sigma_8'] = 0.8288
cosmo['tau'] = 0.0952
cosmo['Y_He'] = 0.248
cosmo['z_reion'] = 11.52


colours_lit = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.10, 0.90, 7)))))[[0, 2, 4, 3, 5, 1, 6]]


#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=14)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=14)
plt.rc('axes', labelsize=14)


def calc_sigma_r(redshifts, length, cosmology):
    radius = (length**3 / (4.*np.pi/3.))**(1./3.)
    sigma = cp.perturbation.sigma_r(radius, redshifts, **cosmology)
    return sigma[0]


file1 = "RefL0025N0376_Log_Normal_Fit_Mean_Std.txt"
file2 = "RefL0025N0752_Log_Normal_Fit_Mean_Std.txt"
file3 = "RecalL0025N0752_Log_Normal_Fit_Mean_Std.txt"
file4 = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"
file5 = "RefL0050N0752_log_normal_fit_mean_std_percent.txt"


z1, mean1, std1, per1 = np.loadtxt(file1, unpack=True, skiprows=1)
z2, mean2, std2, per2 = np.loadtxt(file2, unpack=True, skiprows=1)
z3, mean3, std3, per3 = np.loadtxt(file3, unpack=True, skiprows=1)
z4, mean4, std4, per4 = np.loadtxt(file4, unpack=True, skiprows=1)
z5, mean5, std5, per5 = np.loadtxt(file5, unpack=True, skiprows=1)

z4 = z4 + math_tools.cosmology.cMpc_to_z(100)
#colours = ['#ef8a62', '#67a9cf', '#666666']
#colours = ['#1b9e77','#d95f02','#7570b3']
colours = ['#1b9e77','#d95f02','#7570b3','#e7298a']
colours = np.array(list(map(mpl.colors.to_hex, cmasher.rainforest(np.linspace(0.20, 0.80, 4)))))[[3, 2, 1, 0]]
colours = np.append(colours, "#000000")
print(colours)


###########################################################
###########################################################
# PLOT <DM> vs Redshift
###########################################################
###########################################################
plt.plot(z1, mean1, label="RefL0025N0376", color=colours[0])
plt.plot(z2, mean2, label="RefL0025N0752", color=colours[1])
plt.plot(z5, mean5, label="RefL0050N0752", color=colours[3])
plt.plot(z3, mean3, label="RecalL0025N0752", color=colours[2])
plt.plot(z4, mean4, label="RefL0100N1504", color=colours[4], linewidth=2)

plt.xlabel("Redshift")
plt.ylabel("$<\mathrm{DM}>$")
plt.xlim(0, 3)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("dmz_relation_mean_all_simulations.png", dpi=175)

###########################################################
###########################################################
# PLOT sigma(DM) vs Redshift
###########################################################
###########################################################
plt.clf()

vartest = "/home/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS/correlation_varience_test.txt"
vartest_L50N752 = "/home/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/output/T4EOS/correlation_varience_test.txt"

var_data = np.loadtxt(vartest, skiprows=1, unpack=True)
redshift = var_data[0]
std_cumsum = var_data[5]
fit_std = var_data[10]

var_data_L50N752 = np.loadtxt(vartest_L50N752, skiprows=1, unpack=True)
redshift_L50N752 = var_data_L50N752[0]
std_cumsum_L50N752 = var_data_L50N752[5]
fit_std_L50N752 = var_data_L50N752[10]


plt.plot(z1, std1, label="RefL0025N0376", color=colours[0])
plt.plot(z2, std2, label="RefL0025N0752", color=colours[1])
plt.plot(z5, std5, label="RefL0050N0752", color=colours[3])
plt.plot(z3, std3, label="RecalL0025N0752", color=colours[2])
plt.plot(z4, std4, label="RefL0100N1504", color=colours[4])
plt.plot(redshift, std_cumsum, label="RefL0100N1504 $\sqrt{\sum \sigma^2}$", color=colours[4], linestyle=":")
plt.plot(redshift, fit_std, label="RefL0100N1504 Fit $\sqrt{\sum \sigma^2}$", color=colours[4], linestyle="-.")
plt.plot(redshift_L50N752, std_cumsum_L50N752, label="RefL0050N0752 $\sqrt{\sum \sigma^2}$ ", color=colours[3], linestyle=":")
plt.plot(redshift_L50N752, fit_std_L50N752, label="RefL0050N0752 Fit $\sqrt{\sum \sigma^2}$", color=colours[3], linestyle="-.")


plt.xlabel("Redshift")
plt.ylabel(r"$\sigma \mathrm{(DM)\ [pc\ cm^{-3}]}$")
plt.xlim(0, 0.87)
plt.ylim(0, 200)
plt.legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig("dmz_relation_std_all_simulations.png", dpi=175)

###########################################################
###########################################################
# PLOT sigma(DM)/<DM> vs Redshift
###########################################################
###########################################################

plt.clf()
plt.plot(z1, per1, label="RefL0025N0376", color=colours[0], linewidth=2)
plt.plot(z2, per2, label="RefL0025N0752", color=colours[1], linewidth=2)
plt.plot(z5, per5, label="RefL0050N0752", color=colours[3], linewidth=2)
plt.plot(z3, per3, label="RecalL0025N0752", color=colours[2], linewidth=2)
#plt.plot(z4, per4+0.3*per4, label="TestColor", color=colours[3], linewidth=2)
plt.plot(z4, per4, label="RefL0100N1504", color=colours[4], linewidth=2)

plt.xlabel("Redshift")
plt.ylabel("$\sigma(\mathrm{DM})/<\mathrm{DM}>$")
plt.xlim(0, 3)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("dmz_relation_percent_all_simulations.png", dpi=175)

###########################################################
###########################################################
# PLOT sigma(DM)/<DM> vs Redshif with Jaro and Dolag
###########################################################
###########################################################

def dolag_sigma(z):
    return 126.9*z + 2.03 - 27.4*z**3




plt.clf()

jaro_data = [(1, 12.70), (2, 8.54), (3, 6.68)]
jaro_z = [z[0] for z in jaro_data]
jaro_per = [p[1] for p in jaro_data]

dolag_z = np.array([0.066, 0.137, 0.293, 0.470, 0.672, 0.901, 1.323, 1.980])
dolag_sig = dolag_sigma(dolag_z)
dolag_mean = np.array([36.40, 87, 212, 381, 590, 839, 1325, 2139.4])
dolag_per = dolag_sig/dolag_mean * 100



mcquinn_z = np.array([1])
mcquinn_per = np.array([26])
mcquinn_err = np.array([10])
mcquinn_std_z, mcquinn_std = np.loadtxt("DM_redshift_models/mcquinn2014_std.txt", unpack=True)
mcquinn_mean_z, mcquinn_mean = np.loadtxt("DM_redshift_models/mcquinn2014_model.txt", unpack=True)

std_interp = interp1d(mcquinn_std_z, mcquinn_std)
mcquinn_std_values = std_interp(mcquinn_mean_z[3:-1])

mcquinn_per = mcquinn_std_values / mcquinn_mean[3:-1] * 100
#plt.plot(z1, per1, label="RefL0025N0376", color=colours[0], linewidth=2)
#plt.plot(z2, per2, label="RefL0025N0752", color=colours[1], linewidth=2)
#plt.plot(z5, per5, label="RefL0050N0752", color=colours[3], linewidth=2)
#plt.plot(z3, per3, label="RecalL0025N0752", color=colours[2], linewidth=2)
#plt.plot(z4, per4+0.3*per4, label="TestColor", color=colours[3], linewidth=2)
plt.plot(z4, per4 * 100, label="RefL0100N1504", color=colours_lit[0], linewidth=2)
#plt.plot(jaro_z, jaro_per, label="Jaroszynski (2019)", linewidth=2, color=colours_lit[5])
#plt.scatter(dolag_z, dolag_per, label="Dolag et al. (2015)", marker="^", color=colours_lit[4])
#plt.errorbar(mcquinn_z, mcquinn_per, yerr=mcquinn_err, label="McQuinn (2014)", marker="*")
plt.plot(mcquinn_mean_z[3:-1], mcquinn_per, label="McQuinn (2014)", color=colours_lit[3])

plt.xlabel("Redshift")
plt.ylabel("Relative $1\sigma$ scatter (\%) \n$\sigma(\mathrm{DM})/<\mathrm{DM}> \\times\ 100 $")
plt.xlim(0, 3.016)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("dmz_relation_percent_with_literature_3.png", dpi=175)

###########################################################
###########################################################
# PLOT sigma(DM)/sigma_R/sigma_R,0 vs Redshift
###########################################################
###########################################################

plt.clf()
z1_sigma_r = calc_sigma_r(z1, 25, cosmo)
z1_sigma_r_ratio = z1_sigma_r/z1_sigma_r[0]

z2_sigma_r = calc_sigma_r(z2, 25, cosmo)
z2_sigma_r_ratio = z2_sigma_r/z2_sigma_r[0]

z3_sigma_r = calc_sigma_r(z3, 25, cosmo)
z3_sigma_r_ratio = z3_sigma_r/z3_sigma_r[0]

z4_sigma_r = calc_sigma_r(z4, 100, cosmo)
z4_sigma_r_ratio = z4_sigma_r/z4_sigma_r[0]

std1_sigma_corrected = std1/z1_sigma_r_ratio
std2_sigma_corrected = std2/z2_sigma_r_ratio
std3_sigma_corrected = std3/z3_sigma_r_ratio
std4_sigma_corrected = std4/z4_sigma_r_ratio

std1_sigma_corrected_2 = std1/3.41
std2_sigma_corrected_2 = std2/3.41
std3_sigma_corrected_2 = std3/3.41
std4_sigma_corrected_2 = std4


plt.plot(z1, std1_sigma_corrected, label="RefL0025N0376", color=colours[0], linewidth=2)
plt.plot(z2, std2_sigma_corrected, label="RefL0025N0752", color=colours[1], linewidth=2)
plt.plot(z3, std3_sigma_corrected, label="RecalL0025N0752", color=colours[2], linewidth=2)
#plt.plot(z4, per4+0.3*per4, label="TestColor", color=colours[3], linewidth=2)
plt.plot(z4, std4_sigma_corrected, label="RefL0100N1504", color=colours[4], linewidth=2)

plt.xlabel("Redshift")
plt.ylabel("$\sigma(\mathrm{DM})/\sigma_R(z)/\sigma_{R,0}$")
plt.xlim(0, 3)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("dmz_relation_std_sigma_corrected_all_simulations.png", dpi=175)

###########################################################
###########################################################
# PLOT sigma(DM)/sigma_R/sigma_R,0/<DM> vs Redshift
###########################################################
###########################################################

#plt.clf()
#plt.plot(z1, std1_sigma_corrected_2/mean1, label="RefL0025N0376", color=colours[0], linewidth=2)
#plt.plot(z2, std2_sigma_corrected_2/mean2, label="RefL0025N0752", color=colours[1], linewidth=2)
#plt.plot(z3, std3_sigma_corrected_2/mean3, label="RecalL0025N0752", color=colours[2], linewidth=2)
#plt.plot(z4, per4+0.3*per4, label="TestColor", color=colours[3], linewidth=2)
#plt.plot(z4, std4_sigma_corrected_2/mean4, label="RefL0100N1504", color=colours[4], linewidth=2)

#plt.xlabel("Redshift")
#plt.ylabel("$\sigma(\mathrm{DM})/<\mathrm{DM}>/3.41$")
#plt.xlim(0, 3)
#plt.legend(frameon=False, fontsize=12)
#plt.tight_layout()
#plt.savefig("dmz_relation_percentage_sigma_corrected_all_simulations_again.png", dpi=175)




