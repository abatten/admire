"""
Bootstrap: To do stats
"""
#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def bootstrap(data, resamples=10, num_samples=None):
    """
    Performs a bootstrap resampling on a dataset.
    """
    output = []
    for i in range(resamples):
        output.append(_resample(data, num_samples=num_samples))
    return output


def _sample(data):
    """
    Randomly picks a sample from an item in a list.

    Parameters
    ----------
    data: list
        The list of import data to sample.

    Returns
    -------
    sample:
        The randomly selected item from the list.
    """

    max_idx = len(data)
    rand_idx = np.random.randint(0, max_idx)

    sample = data[rand_idx]

    return sample


def _resample(data, num_samples=None):
    resample = np.empty(0)
    if not num_samples:
        num_samples = len(data)
    for i in range(num_samples):
        resample = np.append(resample, _sample(data))
    return resample


def uncert_vals(x, num_sigma=1):
    num_items = len(x)
    x.sort()

    sigma = {
            1: 0.682689492/2,
            2: 0.954499736/2,
            3: 0.997300204/2,
            4: 0.99993666/2,
            5: 0.999999426697/2
            }

    lower_sigma_idx = int((0.5 - sigma[num_sigma]) * num_items)
    upper_sigma_idx = int((0.5 + sigma[num_sigma]) * num_items)
    return x[lower_sigma_idx], x[upper_sigma_idx]


def boot_hist(x, bins=30, best_fit_line=None, best_fit_median=False, sigma_lines=False, passed_ax=None):

    if passed_ax:
        ax = passed_ax
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    ax.hist(x, bins=bins)

    if best_fit_line:
        ax.axvline(best_fit_line, color='r', linewidth=2)

    if sigma_lines:
        err_low, err_high = uncert_vals(x)
        ax.axvline(err_low, color='g', linewidth=2)
        ax.axvline(err_high, color='g', linewidth=2)

    if passed_ax:
        return ax
    else:
        #  return fig
        plt.savefig("testing_2.pdf")
