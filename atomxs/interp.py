import numpy as np
import xarray as xr
from scipy.interpolate import Akima1DInterpolator, interp1d


def interp_xs(xs_table: xr.DataArray, energy: np.ndarray, method='log-log'):
    """
    Interpolate a cross section (XS) table onto an energy grid using interpolation.

    Parameters
    ----------
    xs_table : xr.DataArray
        Cross section data with energy coordinates in eV.
    energy : np.ndarray
        1D array of energies where the interpolated XS values are desired.
    method : str, default 'log-log'
        Interpolation method:
        - 'log-log': linear interpolation in log(E) vs log(XS) space.
        - 'linear-linear': linear interpolation in log(E) vs log(XS) space.

    Returns
    -------
    xr.DataArray
        Interpolated cross-section values at the requested energies.

    Raises
    ------
    ValueError
        If `method` is not 'log-log' or 'linear-linear'

    """
    l_energy = np.log(xs_table.energy)
    l_xs = np.log(xs_table.values)
    if method == 'log-log':
        interp = interp1d(l_energy, l_xs)
        values = np.exp(interp(np.log(energy)))
    elif method == 'linear-linear':
        interp = interp1d(xs_table.energy, xs_table.values)
        values = interp(energy)
    else:
        raise ValueError("method is incorrect")

    return values