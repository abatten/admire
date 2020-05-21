import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher

#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=12)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=12)
plt.rc('axes', labelsize=12)

file1 = "RefL0025N0376_Log_Normal_Fit_Mean_Std.txt"
file2 = "RefL0025N0752_Log_Normal_Fit_Mean_Std.txt"
file3 = "RecalL0025N0752_Log_Normal_Fit_Mean_Std.txt"
file4 = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"


z1, mean1, std1, per1 = np.loadtxt(file1, unpack=True, skiprows=1)
z2, mean2, std2, per2 = np.loadtxt(file2, unpack=True, skiprows=1)
z3, mean3, std3, per3 = np.loadtxt(file3, unpack=True, skiprows=1)
z4, mean4, std4, per4 = np.loadtxt(file4, unpack=True, skiprows=1)


#colours = ['#ef8a62', '#67a9cf', '#666666']
#colours = ['#1b9e77','#d95f02','#7570b3']
colours = ['#1b9e77','#d95f02','#7570b3','#e7298a']
colours = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.00, 0.90, 5)))))[[4, 3, 2, 1, 0]]

plt.plot(z1, mean1, label="RefL0025N0376", color=colours[0])
plt.plot(z2, mean2, label="RefL0025N0752", color=colours[1])
plt.plot(z3, mean3, label="RecalL0025N0752", color=colours[2])
plt.plot(z4, mean4, label="RefL0100N1504", color=colours[4], linewidth=3)

plt.xlabel("Redshift")
plt.ylabel("$<\mathrm{DM}>$")
plt.legend(frameon=False, fontsize=12)
plt.savefig("dmz_relation_mean_all_simulations.png")

plt.clf()
plt.plot(z1, std1, label="RefL0025N0376", color=colours[0])
plt.plot(z2, std2, label="RefL0025N0752", color=colours[1])
plt.plot(z3, std3, label="RecalL0025N0752", color=colours[2])
plt.plot(z4, std4, label="RefL0100N1504", color=colours[4])

plt.xlabel("Redshift")
plt.ylabel(r"$\sigma \mathrm{(DM)\ [pc\ cm^{-3}]}$")
plt.legend(frameon=False, fontsize=12)
plt.savefig("dmz_relation_std_all_simulations.png")

plt.clf()
plt.plot(z1, per1, label="RefL0025N0376", color=colours[0], linewidth=2)
plt.plot(z2, per2, label="RefL0025N0752", color=colours[1], linewidth=2)
plt.plot(z3, per3, label="RecalL0025N0752", color=colours[2], linewidth=2)
plt.plot(z4, per4+0.3*per4, label="TestColor", color=colours[3], linewidth=2)
plt.plot(z4, per4, label="RefL0100N1504", color=colours[4], linewidth=4)

plt.xlabel("Redshift")
plt.ylabel("$\sigma(\mathrm{DM})/<\mathrm{DM}>$")
plt.legend(frameon=False, fontsize=12)
plt.savefig("dmz_relation_percent_all_simulations.png")


