import numpy as np
import astropy.units as u
import astropy

from glob import glob


def scale_factor(redshift):
    a = (1 + redshift)**-1.0
    return a

def physical_length(dist_phys, redshift):
    length = dist_phys * scale_factor(redshift)
    return length

def physical_area(dist_phys, redshift):
    return physical_length(dist_phys, redshift)**2.0


#path = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ElectronColDensity/"
path = "/home/abatten/oz071/nwijers/maps/electrons_T4EOS/"
files = sorted(glob(path + "*"))

redshifts = np.array([3.017, 1.004, 0.0])

for i, filename in enumerate(files):
    f = np.load(filename, "r")

    total_column_density = np.sum(10**f["arr_0"])
    print(f"Total Column Density: {total_column_density}")

    total_number_of_electrons = total_column_density * (u.cm**-2) * physical_area(100 * u.Mpc, redshifts[i])

    print(f"Total Num Electrons: {total_number_of_electrons.to(u.dimensionless_unscaled)}")


    num_electrons_per_column = 10**f["arr_0"] * (u.cm**-2) * physical_area((100/32000)*u.Mpc, redshifts[i])

    print(f"Num Electrons Per Column: {num_electrons_per_column.to(u.dimensionless_unscaled)}")

    total_num_electrons_2 = np.sum(num_electrons_per_column.to(u.dimensionless_unscaled))
    print(f"Total Num Electrons: {total_num_electrons_2}")
