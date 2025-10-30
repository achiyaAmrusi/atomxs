import h5py
import xarray as xr
from typing import List, Dict
import importlib.resources as pkg_resources

def get_xs(Z_list: List[int], MT_list: List[int]) -> Dict[int, Dict[int, xr.DataArray]]:
    """
    Extract selected MT cross sections for multiple elements from an HDF5 file
    inside the package library.

    Parameters
    ----------
    h5_filename : str
        Name of the HDF5 file inside the package's library directory.
    Z_list : list of int
        Atomic numbers of elements to extract.
    MT_list : list of int
        MT numbers to extract (e.g., [501, 522]).

    Returns
    -------
    Dict[int, Dict[int, xr.DataArray]]
        Nested dictionary: data[Z][MT] gives an xarray.DataArray for that element
        and reaction type. If an MT is missing, it is skipped.
    """
    data = {}

    # Locate the HDF5 file inside the package library
    with pkg_resources.path('atomxs.library', 'photoatomic_xs.h5') as h5_path:
        with h5py.File(h5_path, 'r') as h5f:
            for Z in Z_list:
                strZ = str(Z)
                if strZ not in h5f:
                    continue  # skip missing elements
                gZ = h5f[strZ]
                data[Z] = {}
                for MT in MT_list:
                    strMT = str(MT)
                    if strMT not in gZ:
                        continue  # skip missing MT
                    energy = gZ[strMT]["energy"][:]
                    xs = gZ[strMT]["xs"][:]
                    da = xr.DataArray(
                        xs,
                        coords={'energy': energy},
                        dims=['energy'],
                        name=f'Z{Z}_MT{MT}'
                    )
                    data[Z][MT] = da

    return data


def get_MT(Z_list: List[int]) -> Dict[int, List[int]]:
    """
    Return all MT numbers available for a specific element Z in the HDF5 file.

    Parameters
    ----------
    Z : int
        Atomic number of the element.

    Returns
    -------
    List[int]
        List of MT numbers available for this element.
    """
    MT_dict = {}

    with pkg_resources.path("atomxs.library", "photoatomic_xs.h5") as h5_path:
        with h5py.File(h5_path, "r") as h5f:
            for Z in Z_list:
                strZ = str(Z)
                if strZ not in h5f:
                    continue  # skip missing Z instead of raising
                gZ = h5f[strZ]

                # Read MT dataset
                MTs = gZ["MT"][:].astype(int).tolist()
                MT_dict[Z] = MTs

    return MT_dict