import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable



import cmasher

from model import DMzModel

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


def linear(x, m, b):
    """
    
    """
    return m * x + b


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
    filename = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"

    redshift, mean, std, percent = np.loadtxt(filename, unpack=True, skiprows=1)

    plt.plot(redshift, mean, 'blue', label="Data", linewidth=3)

    fit1 = curve_fit(linear, redshift, mean, bounds=([0, 0], [1200, 1e-10]))
    plt.plot(redshift, linear(redshift, fit1[0][0], fit1[0][1]), 'red', linewidth=2, label='Fit1')
    fit2 = curve_fit(linear, redshift, mean)
    plt.plot(redshift, linear(redshift, fit2[0][0], fit2[0][1]), 'green', linewidth=1, label='Fit2')
    plt.legend()
    plt.savefig('TESTING_MEAN_FIT.png')
    plt.clf()
    print(fit1[0])
    print(fit2[0])

    plt.plot(redshift, std, 'blue', label="Data", linewidth=3)
    #fit1 = curve_fit(power_law, redshift, std, bounds=([0, 0], [1000, 1e-5]))
    #plt.plot(redshift, power_law(redshift, fit1[0][0], fit1[0][1]), 'red', linewidth=2, label='Fit1')
    fit2 = curve_fit(exponential, redshift, std)
    plt.plot(redshift, exponential(redshift, fit2[0][0], fit2[0][1], fit2[0][2]), 'green', linewidth=1, label='Fit2')
    plt.plot(redshift, exponential(redshift, -206, -1.08, 221.5), 'red', linewidth=1, label='Test')
    plt.legend()
    plt.savefig('TESTING_STD_FIT.png')
    #print(fit1[0])
    #print(fit2[0])
    print(fit1[0])
    print(fit2[0])

