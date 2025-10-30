import numpy as np
import xarray as xr
from materials import Material
from .interp import interp_xs
from .lib_utils import get_xs
from scipy.constants import Avogadro
barn = 1e-24


def mass_attenuation(material: Material, energy_grid=np.ndarray, mt_list=[501], interp_method='log-log'):
    """
    Compute total mass attenuation for a material.

    Parameters
    ----------
    material : Material
        a Material
    energy_grid : array, optional
        Energy grid to interpolate onto.
    mt : int
        MT number for XS (default 501: total electron XS)

    Returns
    -------
    mu: xarray.DataArray [cm**2/g]
    The mass attenuation coefficient
    """
    material_z = set()

    for iso, _ in material.composition.items():
        material_z.add(iso.Z)
    material_z = list(material_z)

    full_attenuation_coeff = get_xs(material_z, mt_list)

    # interpolate onto requested energy grid
    # structure: [Z, MT, E]
    attenuation_coeff_z = np.zeros((len(material_z), len(mt_list), energy_grid.shape[0]))
    for z_ind, z in enumerate(material_z):
        for mt_ind, mt in enumerate(mt_list):
            attenuation_coeff_z[z_ind, mt_ind, :] = interp_xs(
                full_attenuation_coeff[z][mt],
                energy=energy_grid,
                method=interp_method,
            )

    # convert material to atomic fractions (needed for weighting)
    material_atomic = material.update_fraction_type("atomic")

    # initialize result arrays
    mu_over_rho = np.zeros_like(energy_grid)

    # weight each isotope's Z cross section by its atomic fraction
    for iso, frac in material_atomic.composition.items():
        z_ind = material_z.index(iso.Z)
        # sum over MTs
        mu_el = attenuation_coeff_z[z_ind].sum(axis=0)  # shape (E,)
        # add weighted contribution
        mu_over_rho += frac * mu_el * (Avogadro * barn / iso.mass)

    return xr.DataArray(mu_over_rho, coords={'energy':energy_grid})

def attenuation_coeff(material: Material, energy_grid=np.ndarray, mt_list=[501], interp_method='log-log'):
    """
    Compute total attenuation for a material.
    This is equivalent for the macroscopic cross section

    Parameters
    ----------
    material : Material
        a Material
    energy_grid : array, optional
        Energy grid to interpolate onto.
    mt : int
        MT number for XS (default 501: total electron XS)

    Returns
    -------
    mu: xarray.DataArray [cm**2/g]
    The attenuation coefficient
    """
    return mass_attenuation(material, energy_grid, mt_list, interp_method) * material.density