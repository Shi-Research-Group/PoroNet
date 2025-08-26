def get_fractional_coordinates(points, ase_atoms):
    """Find the fractional coordinates for a bunch of cartesian corrdinates in a box corresponding to an ase.atoms object

    :param points: numpy array of shape (Npoints, 3)
    :type points: numpy.array
    :param ase_atoms: ASE atoms object for the box
    :type ase_atoms: ase.atoms
    :return: numpy array of fractional coordinates of shape (Npoints,3)
    :rtype: numpy.array
    """    
    import numpy as np
    import ase
    cell= ase.geometry.complete_cell(ase_atoms.get_cell()).T 
    cell_inv = np.linalg.inv(cell)
    frac_coords = np.dot(cell_inv, points.T).T
    return frac_coords

def find_coord_from_indices(indices, shape, ase_atoms):
    """Get the cartesian coordinates from the indces of a grid mapped over an ase atoms object.

    :param indices: indices of interest, can be a list of a (N,3) numpy array, where N is number of points
    :type indices: list or np.array (N,3)
    :param shape: overall shape of the 3D grid
    :type shape: np.shape output, so tuple or array of shape (1,3)
    :param ase_atoms: ASE atoms object corressponding to the grid
    :type ase_atoms: ase.atoms object
    :return: cartesian coordinates of the grid points
    :rtype: numpy arry of size (N,3) where N is the number of grid points
    """
    import ase
    import numpy as np
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    return np.dot(cell, (indices/shape).T).T

def interpolate_labels(regions):
    """Create a scipy.interpolator.RegularGridInterpolator, using 'nearest' method for the region labels (or any 3D numpy grid) 
    over a unit box ([0,1]) in all axes.

    :param regions: 3D numpy array of region labels
    :type regions: numpy.array 
    :return: a scipy RegularGridInterpolator object that works on the fractional coordinates of atoms
    :rtype: scipy.interpolate.RegularGridInterpolator
    """    
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI
    xx = np.linspace(0,  1,regions.shape[0])
    yy = np.linspace(0,  1,regions.shape[1])
    zz = np.linspace(0,  1,regions.shape[2])
    rl = RGI((xx, yy, zz), regions, method='nearest')

    return rl