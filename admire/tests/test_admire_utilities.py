import astropy.units as u
import astropy.constants as c
import admire.utilities as utils


def test_z_to_mpc_zero():
    assert utils.z_to_mpc(0) == 0.0 * u.Mpc

def test_mpc_to_z_zero():
    assert utils.mpc_to_z(0.0 * u.Mpc) == 0.0


if __name__ == "__main__":
    utils.vprint("This is a vprint")
