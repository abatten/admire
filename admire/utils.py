
import astropy.units as u


def z_to_mpc(z):
    return cosmo.comoving_distance(z)

def mpc_to_z(mpc):
    return z_at_value(cosmo.comoving_distance, mpc)
