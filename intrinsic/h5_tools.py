import h5py
from typing import Union
import numpy as np


def write_dataset(gp: h5py.Group, dset_name: str, data):
    """
    Write an HDF5 dataset that might already exists, in which case it overwrites it

    Parameters
    ----------
    gp: h5py.Group
        HDF5 group to write to
    dset_name: str
        Name of the dataset
    data: Any
        Data to write
    Return
    ------
    dset: h5py.Dataset
        The created / written dataset
    """
    if dset_name in gp.keys():
        del gp[dset_name]
    gp.create_dataset(dset_name, data=data)

    dset = gp[dset_name]
    return dset


def get_group(parent: Union[h5py.File, h5py.Group], name: str, create=True):
    """
    Get a hdf5 group from its name. If it does not exist can create it or return None

    Parameters
    ----------
    parent: h5py.File or h5py.Group
        Parent of the group to find
    name: str
        Name of the group to return
    create: bool
        Should a non-existing group be created? Default to True

    Returns
    -------
    gp : h5py.Group
        Requested group
    """
    try:
        gp = parent[name]
    except KeyError:
        if create:
            gp = parent.create_group(name)
        else:
            gp = None
    return gp


def get_dataset(gp: Union[h5py.File, h5py.Group], name: str, default_value=None):
    """
    Get a dataset from a hdf5 file
    If it does not exist can return a default value (dictionary-like)

    Parameters
    ----------
    gp: h5py.File or h5py.Group
        Parent of the dataset to find
    name: str
        Dataset name
    default_value: Any
        Default value to return. Default to None

    Returns
    -------
    dset: h5py.Dataset or default_value
        Dataset or the requested default value
    """
    try:
        return gp[name]
    except KeyError:
        return default_value

def write_roi_range(roi_grp, x_range, y_range, stack):
    '''
    we will create a dataset called roi_range if that does not exit in the roi grp
    '''
    try:
        range_ds = roi_grp.create_dataset('roi_range', data = np.array([0,0,0,0]))
        range_ds[:] = [*x_range, *y_range]

        write_dataset(roi_grp, 'normalized_stack', stack)
    except Exception as e:
        print(e)
        range_ds = roi_grp['roi_range']
        range_ds[:] = [*x_range, *y_range]


    
       

    

    



