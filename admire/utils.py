
import astropy.units as u


def z_to_mpc(z):

    if z == 0:
        return 0 * u.Mpc
    else:
        return cosmo.comoving_distance(z)

def mpc_to_z(mpc):

    if mpc == 0:
        return 0
    else:
        return z_at_value(cosmo.comoving_distance, mpc)
