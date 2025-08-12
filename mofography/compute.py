def egrid_from_atoms2(
    ase_atoms,
    spacing=0.5,
    blocksize=20000,
    forcefield_mixed=None,
    cutoff=12.5,
    probe_symbol="He",
    return_ncells=False,
    precompute_aabb=True,
):
    """
    > It takes an ASE atoms object, and returns a dask array of the vdw energy on a grid.

    The function is a bit long, but it's not too complicated.

    The first part of the function is just to get the grid points.

    The second part is to get the forcefield parameters for the atoms.

    The third part is to get the positions of the atoms in the unit cell, and to get the unit cell
    vectors.

    The fourth part is to repeat the unit cell to get the positions of the atoms in the repeated unit
    cell.

    The fifth part is to map the grid point energy function onto the grid points.

    The sixth part is to reshape the grid points into a 3D array.

    The seventh part is to return the grid points.

    The eighth part is to return the number of repetitions of the unit cell if that return_ncells parameter is turned on.

    :param ase_atoms: an ase atoms object
    :param spacing: the spacing of the grid points
    :param blocksize: the size of the chunks that the grid is split into. This is important for memory
    management, defaults to 20000 (optional)
    :param forcefield_mixed: This is a list of the forcefield parameters for each atom in the unit cell
    :param cutoff: the cutoff distance for the Lennard-Jones potential
    :param probe_symbol: the symbol of the probe atom used to calculate the van der Waals radii,
    defaults to He (optional)
    :param return_ncells: if True, returns the number of cells needed to meet the cutoff, defaults to
    False (optional)
    :param precompute: if True, then the AABB is precomputed and passed to the function. If False, then
    the positions and cell are passed to the function, defaults to True (optional)
    :return: The grid2 is a dask array, which is a lazy array. It is not computed until you call
    compute() on it.
    """

    import dask.array as da
    import numpy as np
    from .data import mix_lorentz_berthelot
    from .data import forcefields
    import ase

    # import mofography as mgr
    grid, shape = dask_grid_over_atoms(ase_atoms, spacing=spacing, chunksize=blocksize)

    # ab = aabb_for_atoms(ase_atoms)
    if forcefield_mixed is None:
        forcefield_mixed = mix_lorentz_berthelot(
            ase_atoms, forcefield=forcefields["uff"], probe_symbol=probe_symbol
        )
    # grid.rechunk(chunks=(10000,3))

    # * Now here use the the repeated box to get the framework atom positions
    # * also the grid point coordinates now only span one unit cell, since others are simply copies

    from ase.build import make_supercell
    from ase.geometry import complete_cell
    P = np.eye(3)
    repeats = get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    nrepeats = np.prod(repeats)
    np.fill_diagonal(P, val=list(repeats))

    # * let's repeat everything
    ase_atoms_repeat = make_supercell(ase_atoms, P)
    positions_repeat = ase_atoms_repeat.get_positions()
    cell_repeat = ase.geometry.complete_cell(ase_atoms_repeat.get_cell()).T
    forcefield_mixed_repeat = np.tile(forcefield_mixed, (nrepeats, 1))

    # mix_lorentz_berthelot(ase_atoms_repeat, forcefield = forcefields['uff'], probe_symbol=probe_symbol)
    if precompute_aabb:
        ab_repeat = AABB_on_atoms(ase_atoms_repeat)
        # * map the grid point energy function that uses position and cell, on to the grid points
        grid2 = da.map_blocks(
            gpe_aabb,
            grid,
            ab_repeat,
            forcefield_mixed_repeat,
            cutoff=cutoff,
            dtype=np.float64,
            drop_axis=1,
        )
        grid2 = grid2.reshape(shape)
        grid2 = (
            da.from_array(grid2.compute()).rechunk(chunks=(10, 10, 10)).reshape(shape)
        )

    else:
        # * map the grid point energy function that uses position and cell, on to the grid points
        grid2 = da.map_blocks(
            gpe_aabb2,
            grid,
            positions_repeat,
            cell_repeat,
            forcefield_mixed_repeat,
            cutoff=cutoff,
            dtype=np.float64,
            drop_axis=1,
        )
        grid2 = grid2.reshape(shape)

    if return_ncells:
        return grid2, get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    else:
        return grid2

def dask_grid_over_atoms(ase_atoms, spacing=0.1, chunksize=50000):
    """Generate a dask array of grid points over an ase atoms object with a specified grid spacing

    :param ase_atoms: ASE atoms object
    :type ase_atoms: ase.atoms obj
    :param spacing: grid fineness in angstroms, defaults to 0.1
    :type spacing: float, optional
    :param chunksize: number of grid points in each block of the dask array, defaults to 50000
    :type chunksize: int, optional
    :return: dask array of size (N_grid_points, 3) with each block of size 'chunksize'.
    :rtype: dask.array
    """
    import ase
    import dask.array as da
    import numpy as np

    # number of grid points in each direction
    [nx, ny, nz] = (ase_atoms.get_cell_lengths_and_angles()[0:3] / spacing).astype(
        int
    ) + 1
    gpoints = (
        da.stack(
            da.meshgrid(
                np.linspace(0, 1, nx),
                np.linspace(0, 1, ny),
                np.linspace(0, 1, nz),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
    ).rechunk(chunksize, 3)
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T  # cell matrix
    return da.dot(cell, gpoints.T).T, (nx, ny, nz)  # return the actual coordinates


def AABB_on_atoms(ase_atoms, points=None):
    """
    > This function takes an ASE atoms object and returns a freud AABBQuery object

    :param ase_atoms: ASE atoms object
    :param points: the positions of the atoms
    :return: The AABB tree
    """
    from freud.locality import AABBQuery
    from freud.box import Box
    import ase
    # from ase.data import vdw_radii

    if points is None:
        points = ase_atoms.get_positions()
    cell = ase.geometry.complete_cell(
        ase_atoms.get_cell()
    ).T  # this needs to be an upper triangular matrix, essential column vectors, not row vectors

    box = Box.from_matrix(cell, dimensions=3)  # Define the box for Freud
    ab = AABBQuery(box=box, points=points)  # Compute the AABB Tree
    # Perform the nearest neighbor query with n-neighbors (10 by default)

    return ab

def get_repetitions_to_meet_cutoff(ase_atoms, cutoff=12.5):
    """
    It takes an ASE atoms object and a cutoff distance, and returns the number of unit cells in each
    direction that are needed to ensure that the cutoff distance is satisfied

    :param ase_atoms: ASE atoms object
    :param cutoff: the cutoff distance for the pairwise interactions
    :return: The number of repetitions in each direction to meet the cutoff.
    """

    import numpy as np

    la, lb, lc, alpha, beta, gamma = ase_atoms.get_cell_lengths_and_angles()
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    vol = ase_atoms.get_volume()
    eA = [la, 0, 0]
    eB = [lb * np.cos(gamma), lb * np.sin(gamma), 0]
    eC = [
        lc * np.cos(beta),
        lc * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        vol / (la * lb * np.sin(gamma)),
    ]
    A = [eA, eB, eC]
    A = np.array(A)
    A = A.T
    A_inv = np.linalg.inv(A)
    lx_unit, ly_unit, lz_unit = (
        vol / np.linalg.norm(np.cross(eB, eC)),
        vol / np.linalg.norm(np.cross(eC, eA)),
        vol / np.linalg.norm(np.cross(eA, eB)),
    )
    nx_cells, ny_cells, nz_cells = (
        int(np.ceil(2 * cutoff / lx_unit)),
        int(np.ceil(2 * cutoff / ly_unit)),
        int(np.ceil(2 * cutoff / lz_unit)),
    )
    return nx_cells, ny_cells, nz_cells


def get_energy_interpolator(egrid):
    """
    This function creates a 3D regular grid interpolator for energy values given an input energy grid.

    :param egrid: a 3D numpy array representing a grid of energy values. The shape of the array should
    be (n_x, n_y, n_z), where n_x, n_y, and n_z are the number of grid points in the x, y, and z
    directions, respectively
    :return: a RegularGridInterpolator object that can be used to interpolate values from a 3D energy
    grid.
    """

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI

    xx = np.linspace(0, 1, egrid.shape[0])
    yy = np.linspace(0, 1, egrid.shape[1])
    zz = np.linspace(0, 1, egrid.shape[2])
    energy_interpolator = RGI((xx, yy, zz), egrid, method="slinear")
    return energy_interpolator

def get_energies_for_group2(
    group, regions, energy_interpolator=None, return_scaled_coords=False, **kwargs_egrid
):  # ase_atoms, forcefield_mixed, shape, cutoff=12.5, return_coords=False, block_size=20000, sample_fraction=1):
    """
    This function calculates the energies for a group of regions using an energy interpolator and
    returns the energies and scaled coordinates if specified.

    :param group: A list of labels for the regions in the group
    :param regions: A 3D array representing a segmentation of a larger 3D volume into regions or
    objects. Each region is assigned a unique integer label
    :param energy_interpolator: A function that interpolates energies for a given set of coordinates. It
    is used to calculate the energies for a group of points in a region
    :param return_scaled_coords: A boolean parameter that determines whether the function should return
    the energies and the scaled coordinates of the points in the group or just the energies. If set to
    True, the function will return a tuple of two arrays: the energies and the scaled coordinates. If
    set to False, the function will only return the, defaults to False (optional)
    :return: either the energies for all the points in the specified group or the energies and scaled
    coordinates for all the points in the specified group, depending on the value of the
    `return_scaled_coords` parameter.
    """

    import numpy as np
    import random

    from skimage.measure import regionprops

    props = regionprops(regions)
    shape = regions.shape

    # * get region indices for the regions in the group
    prop_labels = np.array([prop["label"] for prop in props])
    prop_indices = [np.where(prop_labels == g)[0][0] for g in group]

    # * create an energy interpolotor if not passed
    if energy_interpolator is None:
        egrid = egrid_from_atoms2(**kwargs_egrid)
        from scipy.interpolate import RegularGridInterpolator as RGI

        xx = np.linspace(0, 1, egrid.shape[0])
        yy = np.linspace(0, 1, egrid.shape[1])
        zz = np.linspace(0, 1, egrid.shape[2])
        energy_interpolator = RGI((xx, yy, zz), egrid, method="slinear")

    # * this group has more than one entry, assuming it is a pbc group
    if len(group) > 1:
        # * grid indices of all the points present in this group

        all_group_indices = np.vstack([props[pi]["coords"] for pi in prop_indices])
        if len(all_group_indices) > 1000000:
            all_group_indices = random.sample(list(all_group_indices), 1000000)
        # * normalize the indices to get the fractional coordinates
        all_group_scaled = np.divide(all_group_indices, shape)

        # #* get the energy for all the points in the group
        # energies = energy_interpolator(all_group_scaled.T)
    else:
        # * if it is only one entry in the pbc group

        pi = np.where(prop_labels == list(group)[0])[0][0]
        # print(pi)
        # * get coordinates and centroids
        coords_indices = props[pi]["coords"]
        all_group_scaled = np.divide(coords_indices, shape)
    # * get the energy for all the points in the group
    energies = energy_interpolator(all_group_scaled)

    if return_scaled_coords:
        return energies, all_group_scaled
    else:
        return energies

def regions_from_dgrid_with_threshold_abs(dgrid, mask_thickness=1, h=0.2, threshold_abs=1, min_distance=2, compactness=0):
    """
    The function `regions_from_dgrid` takes a density grid as input and identifies regions based on
    local maxima, with options for adjusting parameters like mask thickness and compactness.

    :param dgrid: The `dgrid` parameter is a 2D array representing the input data grid. It is used in
    the function to perform calculations and identify regions based on the data values in the grid
    :param mask_thickness: The `mask_thickness` parameter in the `regions_from_dgrid` function
    determines the thickness of the mask used in the watershed segmentation algorithm. It is used to
    create a mask where regions with values greater than the `mask_thickness` are considered as markers
    for segmentation. Increasing the `mask_thickness` value, defaults to 1 (optional)
    :param h: The `h` parameter in the `regions_from_dgrid` function is used to specify the height
    threshold for detecting local maxima in the input `dgrid` array. It is used in the
    `extrema.h_maxima` function to find the local maxima points based on the specified height
    :param min_distance: The `min_distance` parameter in the `regions_from_dgrid` function specifies the
    minimum distance between peaks in the peak_local_max function. It is used to find peaks in the input
    data that are separated by at least this distance. Peaks that are closer than `min_distance` to each
    other are, defaults to 2 (optional)
    :param compactness: The `compactness` parameter in the `regions_from_dgrid` function controls how
    uniform the region sizes are in the output. A higher value for `compactness` will result in more
    equally sized regions, while a lower value will allow for more size variation among the regions.
    Adjusting the `, defaults to 0 (optional)
    :return: The function `regions_from_dgrid` returns two values: `reg_labels` and `new_max_indices.T`.
    """

    from scipy import ndimage as ndi
    import numpy as np

    try:
        import sparse
    except ImportError:
        import collections.abc

        # sparse needs the four following aliases to be done manually.
        collections.Iterable = collections.abc.Iterable
        collections.Mapping = collections.abc.Mapping
        collections.MutableSet = collections.abc.MutableSet
        collections.MutableMapping = collections.abc.MutableMapping
        import sparse

    # from .helper import interpolate_labels, find_coord_from_indices
    from skimage import segmentation
    from skimage.morphology import extrema
    from skimage.feature import peak_local_max, corner_peaks
    from sklearn.cluster import AgglomerativeClustering
    # peak_min =2
    # max_indices = np.vstack(np.where(extrema.h_maxima(dgrid, h=0.5)))
    # max_indices = np.vstack(np.where(extrema.local_maxima(dgrid)))
    # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
    # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
    # Finding the local maxima of the corner response function.

    # ag = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=np.max(np.array(dgrid.shape))/5, compute_full_tree=True).fit(max_indices)
    # max_indices= np.vstack([max_indices[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1]).astype(int)
    # print(max_indices)
    # markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)
    from icecream import ic

    probe = mask_thickness
    import dask.array as da
    import warnings
    # reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int))

    # max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1, **kwargs)).T
    # max_indices = np.array(peak_local_max(dgrid,  **kwargs)).T
    max_indices = np.vstack(np.where(extrema.h_maxima(dgrid, h=h)))
    markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)

    probe = mask_thickness
    import dask.array as da

    reg_labels = segmentation.watershed(
        -dgrid,
        ndi.label(markers_sparse.todense())[0],
        mask=(dgrid > probe).astype(int),
        compactness=compactness,
    )

    # print(keep_these)
    # new_max_indices = max_indices[keep_these-1].T

    new_max_indices = np.array(
        peak_local_max(dgrid, labels=reg_labels,threshold_abs=threshold_abs, num_peaks_per_label=1, p_norm=2)
    ).T
    if len(new_max_indices.T) < len(max_indices.T):
        warnings.warn("Some maxima excluded as they fell inside the mask_thickness")
        # warnings.warn('Some maxima excluded as they the regions were less that 0.001 of the box')

    markers_sparse = sparse.COO(new_max_indices, 1, shape=dgrid.shape)

    probe = mask_thickness
    import dask.array as da

    reg_labels = segmentation.watershed(
        -dgrid,
        ndi.label(markers_sparse.todense())[0],
        mask=(dgrid > probe).astype(int),
        compactness=compactness,
    )

    new_max_indices = np.array(
        peak_local_max(dgrid, labels=reg_labels,threshold_abs=threshold_abs, num_peaks_per_label=1, p_norm=2)
    ).T
    # ir = interpolate_labels(reg_labels)
    # new_max_indices = (new_max_indices.T[np.argsort(ir(new_max_indices.T/dgrid.shape))]).T

    # # new_max_indices = []
    # old_labels = np.unique(reg_labels)  # This is sorted aalreadylready
    # number_of_regions = len(old_labels)
    # new_labels = range(number_of_regions)

    # # * Replace the old-labels with new
    # for i in range(number_of_regions):
    #     reg_labels[reg_labels== old_labels[i]] = new_labels[i]

    # # *  Lets clean out the regions that are too small
    # props = regionprops(reg_labels)
    # print([(p.label, p.area/dgrid.size) for p in props])
    # len([(p.label, p.area/dgrid.size) for p in props if p.area > dgrid.size/1000])
    # keep_these = [p.label for p in props if p.area > dgrid.size/1000]
    # maxima2 = np.vstack([new_max_indices.T[k-1] for k in keep_these])
    # markers_sparse = sparse.COO(maxima2.T, 1, shape=dgrid.shape)

    # # probe = mask_thickness
    # # import dask.array as da
    # reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int), compactness=False)
    # new_max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1)).T

    # * Now interpolate the region labels and sort the maxima in the correct order
    sort_index = np.argsort([reg_labels[tuple(m)] for m in new_max_indices.T])
    new_max_indices = (new_max_indices.T[sort_index]).T

    return reg_labels, new_max_indices.T

def connections_from_regions_and_dgrid(
    region_labels,
    dgrid,
    maxima,
    ase_atoms,
    wall_windows=True,
    distance_threshold=1.0,
    wall_thickness=2,
):
    """
    It takes a 3D array of region labels, a list of maxima, an ASE atoms object, and a distance
    threshold, and returns a list of connections between regions, and the centers of the windows between
    them

    :param region_labels: The labels of the regions, as returned by the skimage.measure.label function
    :param maxima: the maxima of the image
    :param ase_atoms: ASE atoms object
    :param wall_windows: If True, then the windows on the wall are also calculated, defaults to True
    (optional)
    :param distance_threshold: The distance threshold for the agglomerative clustering
    :return: A list of connections between regions.
    """

    def check_connected(regs):
        import numpy as np
        from icecream import ic

        ic.disable()
        from sklearn.cluster import AgglomerativeClustering

        i = regs[0]
        j = regs[1]
        ic.disable()
        ic(regs)

        check_flag = np.logical_or(
            np.logical_and(region_labels == i, outer.get(j)),
            np.logical_and(region_labels == j, outer.get(i)),
        )
        if np.sum(check_flag) > 0:
            # print("Edges found between "+str(i)+' and '+ str(j) + ' sum: '+str(np.sum(check_flag)))
            windices = np.vstack(np.where(check_flag)).T  # * (N,3)
            window_points = np.dot(
                A_unit, (windices / region_labels.shape).T
            ).T  # * (N,3)
            # ic(window_points)
            # * No clustering
            # window_centers = np.mean(window_points, axis=0)

            # * DBScan
            # X_scaled = StandardScaler().fit_transform(window_points)
            # db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
            # window_centers = np.vstack([window_points[db.labels_==i].mean(axis=0) for i in np.unique(db.labels_) if i != -1])

            # if len(window_points) >1:
            # * Agglomerative
            ag = AgglomerativeClustering(
                n_clusters=None,
                linkage="single",
                distance_threshold=minimum_window_separation,
                compute_full_tree=True,
            ).fit(window_points)
            # * we are taking the centroid of the clusters here as the window location
            # window_centers = np.vstack([window_points[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1])

            # * what if we took the index of the point where the distance value is the largest in the window boundary region, that way we don't have to worry bout
            # * regions at the wall
            d_windices = dgrid[
                tuple(windices.T)
            ]  # distance values at the window indices
            r_windices = region_labels[
                tuple(windices.T)
            ]  # region labels at the window indices
            # window_centers = [(window_points[np.argmax(d_windices[np.logical_and(r_windices[ag.labels_ == uwi]==regs[0],ag.labels_==uwi)])] + window_points[np.argmax(d_windices[np.logical_and(r_windices[ag.labels_==uwi]==regs[1],ag.labels_==uwi)])])/2.0 for uwi in np.unique(ag.labels_) if uwi != -1]

            # print('These are window centers', window_centers)
            window_centers = []
            for uwi in np.unique(ag.labels_):
                # for each cluster that is nota noise, do this
                if uwi != -1:
                    # np.argmax(d_windices[ag.labels_==uwi])

                    uwi_dvals = d_windices[
                        ag.labels_ == uwi
                    ]  # distance values in the cluster with label uwi
                    uwi_points = window_points[
                        ag.labels_ == uwi
                    ]  # points in the cluster with label uwi
                    uwi_r = r_windices[ag.labels_ == uwi]

                    # for points in first region

                    # uwi_r1_indices =np.transpose((uwi_r==regs[0])).nonzero()
                    uwi_r1_indices = (np.where)(uwi_r == regs[0])[0]
                    # print('uwi_r1_indices are ', uwi_r1_indices)
                    # print(uwi_r1_indices.shape)
                    uwi_r1_dvals = uwi_dvals[uwi_r1_indices]
                    uwi_r1_points = uwi_points[uwi_r1_indices]
                    uwi_r1_max_indices = np.argmax(
                        uwi_r1_dvals
                    )  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                    # print(uwi_r1_max_indices)
                    uwi_r1_max_points = uwi_r1_points[uwi_r1_max_indices]

                    # for points in second region
                    uwi_r2_indices = (np.where)(uwi_r == regs[1])[0]
                    uwi_r2_dvals = uwi_dvals[uwi_r2_indices]
                    uwi_r2_points = uwi_points[uwi_r2_indices]
                    uwi_r2_max_indices = np.where(
                        uwi_r2_dvals == np.max(uwi_r2_dvals)
                    )[
                        0
                    ]  # indices of the maximum distance values in the cluster with label uwi
                    uwi_r2_max_points = uwi_r2_points[uwi_r2_max_indices]

                    # take the centroid of the two maxima of each region close to the window
                    uwi_center = (uwi_r1_max_points + uwi_r2_max_points) / 2

                    # uwi_center = np.mean(uwi_max_points, axis=0) # mean of the points where the distance is at maximum if ther are multiple a.k.a the window is symmetric on each side
                    # add that to the list of windows
                    window_centers.append(uwi_center)

            # window_centers = np.vstack(window_centers)
            # else:
            # window_centers = window_points
            window_centers = np.vstack(window_centers)
            window_fractional = get_fractional_coordinates(window_centers, ase_atoms)
            window_fractional = np.round(window_fractional, decimals=4)
            window_radii = np.array(dinterp(window_fractional))
            return np.array(
                [regs, True, "internal", window_centers, window_radii], dtype="object"
            )  # returns a list of [[r1,r2], connected or not, array of window_centers]
        else:
            return np.array([regs, False, "internal", None, None], dtype="object")

    from skimage.segmentation import find_boundaries
    import numpy as np
    import itertools
    import ase

    # from sklearn.cluster import DBSCAN
    # from sklearn.preprocessing import StandardScaler
    from icecream import ic
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from mofography import interpolate_labels, get_fractional_coordinates
    import dask.bag as db

    A_unit = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    from icecream import ic

    minimum_window_separation = distance_threshold

    dinterp = interpolate_dgrid(dgrid)
    # * Find the outer boundary of each individual region
    regions_to_check = np.delete(np.unique(region_labels), 0)
    ic(regions_to_check)
    # outer = {reg:find_boundaries(region_labels == reg, mode='outer') for reg in regions_to_check}
    outer = {
        reg: find_boundaries(region_labels == reg, mode="outer")
        for reg in regions_to_check
    }
    ic(len(outer))
    # * Check for connectedness between regions
    check_list = np.vstack(
        list(itertools.combinations(regions_to_check, 2))
    )  # list of pairs of region labels
    connections_list = [
        check_connected(c) for c in check_list
    ]  # now this step could cause an overflow
    # connections_list = db.from_sequence(check_list).map(check_connected).compute()

    # connected_regions = [[c[0], [c[2]] for c in connections_list if c[1] == True] # Make a new list with all the

    # * Windows on the wall are to be calculated
    if wall_windows:
        wwl = get_wall_windows2(
            regions=region_labels,
            maxima=maxima,
            dgrid=dgrid,
            ase_atoms=ase_atoms,
            wall_thickness=wall_thickness,
        )
        if len(wwl) > 0:
            connections_list.append(wwl)

    # #* we have one final thing to do
    # # * if both the regions forming an internal window are on the same wall, then we need to set the window center and the radius to
    # # * that on the wall, since centroid

    # for c in connections_list:
    #     if np.logical_and(c[1], c[2]=='internal'):

    return np.vstack(connections_list)

def add_maxima_to_rag(G, maxima, maxima_radii, shape, ase_atoms):
    """
    > This function takes a graph, a list of maxima, a list of maxima radii, a shape, and an ase_atoms
    object and adds the maxima and maxima radii to the graph.

    Let's look at the function in more detail.

    The first thing we do is import the `find_coord_from_indices` function from the `helper` module.

    Next, we check to make sure that the number of nodes in the graph is equal to the number of maxima.
    If not, we raise a `ValueError`.

    If the number of nodes in the graph is equal to the number of maxima, we loop through the nodes and
    maxima and add the maxima indices, cartesian coordinates and maxima radii to the graph.



    :param G: the graph
    :param maxima: the indices of the maxima (2nd output of regions_from_grid routine)
    :param maxima_radii: the radius of the maxima
    :param shape: the shape of the image
    :param ase_atoms: The atoms object from ASE
    :return: The graph with the maxima added to the nodes
    """

    from .helper import find_coord_from_indices

    if len(G.nodes()) != len(maxima):
        raise ValueError(
            "The number of nodes in the graph and the number of maxima do not match"
        )
    else:
        # regionprops = get_region_props_for_regions(G.nodes())
        for n, m in zip(G.nodes(), maxima):
            G.nodes[n]["maxima"] = find_coord_from_indices(
                maxima[G.nodes[n]["labels"][0] - 1], shape=shape, ase_atoms=ase_atoms
            )  # * add the coordinates of the maxima to the node
            G.nodes[n]["maxima_indices"] = maxima[
                G.nodes[n]["labels"][0] - 1
            ]  # * add the indcies of the maxima
            G.nodes[n]["maxima_radii"] = maxima_radii[
                G.nodes[n]["labels"][0] - 1
            ]  # * add the indcies of the maxima
    return G

def dgrid_from_atoms_cpu_no_aabb(ase_atoms, radii=None, spacing=0.2, block_size=20000):
    """
    The function `dgrid_from_atoms_no_aabb` calculates a distance grid based on the cpu but without using 
    an AABB cell list to accelerate the calculation. Although substantially slower, this function might provide
    better smoothness of the distance grid, especially matrials with small pores. 

    :param ase_atoms: The `ase_atoms` parameter in the `dgrid_from_atoms_no_aabb` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their properties. This object
    typically contains information such as atomic positions, unit cell parameters, atomic symbols, and
    other properties of the atoms in
    :param radii: The `radii` parameter in the `dgrid_from_atoms_no_aabb` function represents the van der
    Waals radii of the atoms in the atomic structure provided as input. These radii are used in the
    calculation of the distance grid. If the `radii` parameter is not provided by the
    :param spacing: The `spacing` parameter in the `dgrid_from_atoms_no_aabb` function represents the
    distance between grid points in the generated grid. It determines the resolution of the grid where
    the distances from atoms to grid points are calculated. A smaller spacing value will result in a
    higher resolution grid with more grid points,
    :param block_size: The `block_size` parameter in the `dgrid_from_atoms_no_aabb` function determines the
    size of each block used for processing the data. In this function, it is used for chunking the data
    and processing it in parallel on the CPU. A larger `block_size` can be more efficient but too big can lead to memory issues, defaults
    to 10000 (optional)
    :return: The function `dgrid_from_atoms_no_aabb` returns a 3D numpy array representing a distance grid
    computed based on the input atomic structure provided in the form of ASE atoms. The distance grid is
    calculated by computing the minimum distance between each point on the grid and the atoms in the
    structure, taking into account the atomic radii.
    """

    from ase.io import read
    import numpy as np
    import dask.array as da
    # import cupy as cp
    import pandas as pd
    from tqdm import tqdm
    import ase

    # data    = read(path_to_cif)
    fpoints = (ase_atoms.get_scaled_positions() - 0.5)
    cell =  (ase.geometry.complete_cell(ase_atoms.get_cell())).T
    # from ase.data import vdw_radii
    # radius_series = pd.Series({'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
    #                         'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
    #                         'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    #                         'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    #                         'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
    #                         'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    #                         'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
    #                         'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
    #                         'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    #                         'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
    #                         'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
    #                         'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    #                         'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
    #                         'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
    #                         'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    #                         'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    #                         'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
    #                         'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    #                         'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
    #                         'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    #                         'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    #                         'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
    #                         'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
    #                         'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69})
    # radii = cp.asarray(radius_series[data.get_chemical_symbols()].values).reshape(-1, 1)
    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
                "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = (
            vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0
        ).reshape(-1, 1)

    # * the vdw radii mendeleev is in picometers
    # radii = cp.asarray(vdw_radii[ase_atoms.get_atomic_numbers()].values/100).reshape(-1, 1)
    [nx, ny, nz] = (ase_atoms.cell.cellpar()[0:3] / spacing).astype(
        int
    ) + 1
    #     print([nx,ny,nz])
    gpoints = (
        da.stack(
            da.meshgrid(
                np.linspace(-0.5, 0.5, nx),
                np.linspace(-0.5, 0.5, ny),
                np.linspace(-0.5, 0.5, nz),
                indexing="ij",
            ),
            -1,
        )
        .reshape(-1, 3)
        .rechunk(block_size, 3)
    )

    #     def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
    #         points = cp.asarray(points)
    #         return cp.min(cp.linalg.norm(cp.dot(cell, (cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0)-cp.around(cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0))).reshape(-1,3).T).T, axis=1).reshape(fpoints.shape[0],-1)-radii, axis=0)

    def gpd_np(points, fpoints=fpoints, cell=cell, radii=radii):
        # points = cp.asarray(points)
        # fpoints = cp.asarray(fpoints)
        # radii = cp.asarray(radii)
        diff = np.expand_dims(points, axis=1) - np.expand_dims(fpoints, axis=0)
        diff = (diff - np.around(diff)).reshape(-1, 3)
        diff = np.dot(cell, diff.T).T
        diff = np.linalg.norm(diff, axis=1).reshape(-1, fpoints.shape[0]) - radii.T
        return np.min(diff, axis=1)

    dgrid = gpoints.map_blocks(gpd_np, chunks=(block_size, ), dtype="float32", drop_axis=1)
    return dgrid.compute().reshape(nx, ny, nz)

def apply_pbc(
    regions,
    maxima,
    maxima_radii,
    ase_atoms,
    wall_thickness=1,
    return_conn=False,
    minimum_overlap_fraction=0.5,
):
    """
    It takes a 3D array of regions and returns a 3D array of regions with periodic boundary conditions
    applied

    :param regions: the regions array
    :param maxima: the list of maxima
    :param maxima_radii: the radius of each region
    :param ase_atoms: The atoms object from ASE
    :param wall_thickness: The thickness of the wall in Angstroms, defaults to 1 (optional)
    :param return_conn: if True, returns a list of lists of connected regions, defaults to False
    (optional)
    :param minimum_overlap_fraction: This is the minimum fraction of overlap at the wall region between two regions for
    them to be considered for connected via PBC. Then we check the maxima locations also to make sure, they are not separated by periodic windows.
    :return: a list of connected components. Each connected component is a list of regions a.k.a the PBC group.
    """

    import copy

    # a = 0
    import numpy as np
    from icecream import ic

    ic.disable()
    regions_pbc = copy.deepcopy(regions)
    shape = regions.shape
    import networkx as nx

    g1 = nx.Graph()
    g1.add_nodes_from(range(len(np.unique(regions))))

    # * we need to recompute the wall thickness for non orthorhombic cells
    # * for this we shall use the perpendicualr length , which is the norm of the reciprocal lattice vector in each direction
    from ase import geometry

    # * let's compute the perpendicular length of each voxel
    # * we will use the reciprocal lattice vectors to do this
    # * we need this to determine how many voxels we need to select as the wall of the unit cell in each direction
    la = ase_atoms.get_cell_lengths_and_angles()[0] / regions.shape[0]
    lb = ase_atoms.get_cell_lengths_and_angles()[1] / regions.shape[1]
    lc = ase_atoms.get_cell_lengths_and_angles()[2] / regions.shape[2]
    alpha = ase_atoms.get_cell_lengths_and_angles()[3] * (np.pi / 180.0)
    beta = ase_atoms.get_cell_lengths_and_angles()[4] * (np.pi / 180.0)
    gamma = ase_atoms.get_cell_lengths_and_angles()[5] * (np.pi / 180.0)
    vol = ase_atoms.get_volume() / regions.size
    eA = [la, 0, 0]
    eB = [lb * np.cos(gamma), lb * np.sin(gamma), 0]
    eC = [
        lc * np.cos(beta),
        lc * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        vol / (la * lb * np.sin(gamma)),
    ]

    la_perp = vol / np.linalg.norm(np.cross(eB, eC))
    lb_perp = vol / np.linalg.norm(np.cross(eC, eA))
    lc_perp = vol / np.linalg.norm(np.cross(eA, eB))

    wall_cells = np.ceil(
        [wall_thickness / la_perp, wall_thickness / lb_perp, wall_thickness / lc_perp]
    ).astype(int)

    for a in [0, 1, 2]:
        # ic(a)
        # print('a', a)

        # * Here we are comparing the left wall and the right wall, so to get th ecorrespoding regions,
        # * we have to mirror the right wall
        llr = regions_pbc.take(
            indices=np.arange(0, wall_cells[a]), axis=a
        )  # * left wall
        rrr = regions_pbc.take(
            indices=np.arange(shape[a] - 1, shape[a] - 1 - wall_cells[a], -1), axis=a
        )  # * right wall
        llf = llr.flatten()
        rrf = rrr.flatten()
        # ic(dict(zip(llr.flatten(), rrr.flatten())))
        # ic(dict(zip(rrr.flatten(),llr.flatten())))
        # dlist = [dict(sorted(zip(llr.flatten(), rrr.flatten()))),dict(sorted(zip(rrr.flatten(),llr.flatten())))]

        # * Check left vs right wall
        # map_dict = dlist[0]
        # map_dict.pop(0)
        # ic(map_dict)

        map_list = np.unique(list(zip(llr.flatten(), rrr.flatten())), axis=0)
        map_list = map_list[np.all(map_list != 0, axis=1)]
        ic(map_list)
        # print('map_list', map_list)
        if len(map_list) > 0:
            # * forwards check
            reg_numbers = np.vstack([(k, v) for k, v in map_list])

            # * let's compute and prnt percentage overlap between pairs of mapping regions
            # * let's filter those connections with less thant 10% overlap
            sumlist = [np.sum(llf == r[0]) for r in reg_numbers]
            matching = [
                np.sum(np.logical_and(llf == r[0], rrf == r[1])) for r in reg_numbers
            ]
            fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

            # * flags where there should be a left to right connection
            left_right = [f > minimum_overlap_fraction for f in fractional_overlap]

            # reg_numbers = reg_numbers[np.where([f > minimum_overlap_fraction for f in fractional_overlap])]
            # print('left-right\n')
            # print(list(zip(reg_numbers, fractional_overlap)))

            # * backwards check

            # #* let's compute and prnt percentage overlap between pairs of mapping regions
            # #* let's filter those connections with less thant minimum overlap fraction
            sumlist = [np.sum(rrf == r[1]) for r in reg_numbers]
            matching = [
                np.sum(np.logical_and(llf == r[0], rrf == r[1])) for r in reg_numbers
            ]
            fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

            # * flags where there should be a right to left connection
            right_left = [f > minimum_overlap_fraction for f in fractional_overlap]

            # print('right-left\n')
            # print(list(zip(reg_numbers, fractional_overlap)))
            # now let's combine the two flags
            pick_these = np.logical_or(left_right, right_left)

            reg_numbers = reg_numbers[pick_these]
            # print(pick_these)
            # print()
            # print(reg_numbers, 'reg numbers after picking')
            if len(reg_numbers) > 0:
                # TODO: Here we are asuuming the region indices to be in order, causes error in older skimage versions fix using props['labels]
                flags = np.vstack(
                    [
                        [
                            maxima[r[0] - 1][a] < wall_cells[a],
                            maxima[r[1] - 1][a] > (shape[a] - wall_cells[a]),
                        ]
                        for r in reg_numbers
                    ]
                )
                # flags = np.logical_or(flags[:,0],flags[:,1,])
                # flags = np.logical_or(flllags[:,0],flags[:,1,])
                flags1 = np.logical_and(flags[:, 0], flags[:, 1])
                # print('flags1\n',list(zip(reg_numbers,flags1)))

                flags2 = np.logical_xor(flags[:, 0], flags[:, 1])
                # print('flags2\n',list(zip(reg_numbers,flags2)))
                reg_radii = np.vstack(
                    [
                        [maxima_radii[r[0] - 1], maxima_radii[r[1] - 1]]
                        for r in reg_numbers
                    ]
                )
                flags3 = np.vstack(
                    [
                        reg_radii[i][np.where(f)]
                        < reg_radii[i][np.where(np.invert(f))][0]
                        if flags2[i]
                        else False
                        for i, f in enumerate(flags)
                    ]
                )
                # print('flags3\n',list(zip(reg_numbers,flags3)))
                # flags3 = np.vstack([ True if flags2[i] else False for i,f,r  in enumerate(list(zip(flags, reg_numbers)))])
                # reg_radii = [maxima_radii[r[0]-1], maxima_radii[r[1]-1] for r in reg_numbers]

                flags = [
                    flags1[i] if flags1[i] else flags2[i] if flags3[i] else False
                    for i, f in enumerate(flags)
                ]
                # print('after checks\n',list(zip(reg_numbers,flags)))
                # ic(flags1)
                # ic(flags2)
                # ic(flags3)
                # ic(flags)       # flags = [True if flags1[i] else True if np.logical_and(flags2[i], flags3[i] ) else False for i,f in enumerate(flags)]

                # * filter based on region numbers
                reg_numbers = reg_numbers[np.where(flags)]

                # g1.add_edges_from(reg_numbers[np.where(flags)])
                g1.add_edges_from(reg_numbers)

    conn = nx.connected_components(g1)
    # ic(list(conn))
    # ic(list(G.edges()))
    # ic([c for c in list(conn)])
    for i, c in enumerate(conn):
        for r in c:
            regions_pbc[np.where(regions == r)] = i
            ic("replace {0} with {1}".format(r, i))

    # if reindex:
    #     # * Re-index the regions so that it runs from zero to N-regions.
    #     old_labels = np.unique(regions_pbc)  # This is sorted aalreadylready
    #     number_of_regions = len(old_labels)
    #     new_labels = range(number_of_regions)

    #     # * Replace the old-labels with new
    #     for i in range(number_of_regions):
    #         regions_pbc[regions_pbc== old_labels[i]] = new_labels[i]
    if return_conn:
        return list(nx.connected_components(g1))[1:]
    else:
        return regions_pbc

def add_pbc_to_rag(rag, pbc_groups):
    """
    It takes a RAG and a list of PBC groups, and adds a 'pbc_group' attribute to each node in the RAG

    :param rag: the region adjacency graph
    :param pbc_groups: a list of lists of labels that are in the same periodic boundary condition group
    :return: A rag with the pbc_group attribute added to each node.
    """
    import numpy as np

    for node in rag.nodes:
        pbc_index = np.where(
            [rag.nodes[node]["labels"][0] in list(g) for g in pbc_groups]
        )[0][0]
        rag.nodes[node]["pbc_group"] = pbc_index
    return rag


def gpe_aabb2(grid_points, positions, cell, forcefield_mixed, cutoff):
    """
    The function `gpe_aabb2` calculates Lennard-Jones energy contributions for grid points based on
    neighboring particles within a specified cutoff distance using positions instead of neighborlist passed as 
    argument as in `gpe_aabb' subroutine.
    
    :param grid_points: The function `gpe_aabb2` seems to be calculating the Lennard-Jones potential
    energy for a set of grid points based on the positions of particles, a cell, a mixed force field,
    and a cutoff distance
    :param positions: The `positions` parameter in the `gpe_aabb2` function represents the positions of
    particles in a simulation. These positions are used to calculate the interactions between the
    particles and the grid points within a certain cutoff distance. The function performs calculations
    based on the positions of the particles and the grid points
    :param cell: The `cell` parameter in the `gpe_aabb2` function represents the simulation box or cell
    in which the particles are contained. It defines the periodic boundaries and the size of the
    simulation box in three dimensions. The `cell` parameter is used in the AABBQuery to specify the
    simulation box
    :param forcefield_mixed: The function `gpe_aabb2` seems to be calculating the Lennard-Jones
    potential energy for a given set of grid points based on the provided force field parameters. The
    key steps involved in the calculation include querying neighboring points within a specified cutoff
    distance, computing the Lennard-Jones
    :param cutoff: The `cutoff` parameter in the `gpe_aabb2` function represents the maximum distance
    within which to consider neighboring points when performing calculations. It is used to filter out
    points that are beyond this distance from each grid point during the computation
    :return: The function `gpe_aabb2` is returning the calculated Lennard-Jones potential energy values
    `v` for each grid point based on the provided input parameters.
    """

    import numpy as np

    from freud.locality import AABBQuery

    ab_r = AABBQuery(cell, positions).query(
        grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
    )
    # ab_r=aabb_object.query(grid_points,query_args=dict(mode='ball',r_max=cutoff,exclude_ii=True))
    ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)  # convertquerytoneighborlist
    # nls=ab_r_nl.point_indices.reshape(-1,n_neighbors)#gettheindicesoftheneighborsforeachgridpt.tolookuptheradiilater
    # dists=ab_r_nl.distances.reshape(-1,n_neighbors)#getthedistancestotheneighborsofeachgridpt.
    # print(ab,ab_r,ab_r_nl,nls,[radii[nl]fornlinnls])#,np.min(dists-[radii[nl]fornlinnls],axis=1))
    # returntheminimum(shortestdistancetothesurface)ineachrow

    indices = ab_r_nl.point_indices  # * frame indices within cutoff at each grid point
    distances = ab_r_nl.distances  # * distances to frame indices at each grid point
    segments = ab_r_nl.segments  # * number of frame indices at each grid point
    inter_params = np.take(
        forcefield_mixed, indices, axis=0
    )  # * get the parameters for each frame index at each grid point
    sig = inter_params[
        :, 1
    ]  # * get the sigma values for each frame index at each grid point
    eps = inter_params[
        :, 0
    ]  # * get the epsilon values for each frame index at each grid point
    sd = np.power(np.divide(sig, distances), 6)  # * (sigma/distance)^6
    lj = 4 * eps * (sd ** 2 - sd)  # * Lennard-Jones terms

    ncounts = ab_r_nl.neighbor_counts  # * number of neighbors at each grid point
    grid_point_indices = np.repeat(
        np.arange(len(ncounts)), ncounts
    )  # * repeat the grid point indices for each neighbor

    v = np.bincount(
        grid_point_indices, weights=lj, minlength=len(ncounts)
    )  # * sum the energies for each grid point

    # sig = np.array([forcefield[i][1] for i in indices])
    # eps = np.array([forcefield[i][0] for i in indices])

    # print(len(indices),len(distances),len(segments),len(sig),len(eps)) #, np.take(forcefield, indices))
    # v = np.add.reduceat(
    #     np.multiply(4 * eps, ((sd) ** 2 - (sd))), segments
    # )  # * sum the energy contributions for each grid point
    # E = np.add.reduceat(4*eps*((sig/distances)**12-(sig/distances)**6), segments, axis=0)

    # E = [lj_sum(d, s, e) for d, s, e in zip(distances, sig, eps)]

    # return np.vstack(E).reshape(-1,1)
    return v

def gpe_aabb(grid_points, aabb_object, forcefield_mixed, cutoff):
    """
    The function `gpe_aabb` calculates Lennard-Jones potentials for grid points based on their distances
    to neighboring points within a specified cutoff distance.
    
    :param grid_points: The `grid_points` parameter in the `gpe_aabb` function represents the
    coordinates of points in a grid. These points are used for calculations involving a AABB
    (Axis-Aligned Bounding Box) object, a force field, and a specified cutoff distance. The function
    calculates Lennard-J
    :param aabb_object: The function `gpe_aabb` you provided seems to calculate the Lennard-Jones
    potentials for a set of grid points based on an axis-aligned bounding box (AABB) object, a force
    field, and a cutoff distance
    :param forcefield_mixed: The `forcefield_mixed` parameter in the `gpe_aabb` function seems to
    represent a mixed force field containing interaction parameters for each pair of particles. The
    function calculates Lennard-Jones potentials between grid points and their neighboring particles
    based on these interaction parameters
    :param cutoff: The `cutoff` parameter in the `gpe_aabb` function represents the maximum distance
    within which interactions between grid points and AABB (Axis-Aligned Bounding Box) objects will be
    considered. Any interactions beyond this distance will not be included in the calculations
    :return: The function `gpe_aabb` returns an array `v` containing the Lennard-Jones potentials
    calculated for each grid point based on the provided input parameters `grid_points`, `aabb_object`,
    `forcefield_mixed`, and `cutoff`.
    """


    import numpy as np
    ab_r = aabb_object.query(
        grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
    )

    ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)  # Convert query to neighbor list
    indices = ab_r_nl.point_indices
    distances = ab_r_nl.distances
    # segments = ab_r_nl.segments
    inter_params = np.take(forcefield_mixed,indices, axis=0)
    sig = inter_params[:, 1]
    eps = inter_params[:, 0]
    # Initialize energies array
    # segments = np.cumsum(ab_r_nl.neighbor_counts)
    # segments = np.insert(segments, 0, 0)
    # print(segments)
    # Calculate the scaled distances (sigma/distance)^6
    ncounts = ab_r_nl.neighbor_counts
    # print('number of entries',  len(distances))
    # print('len of neighbors ', len(ncounts))
    # print('total numbero f neighbors', np.sum(ncounts))
    # print('ncounts',    ab_r_nl.neighbor_counts)
    # print('number of grid points', len(grid_points))

    # compute the corresponding Lennard-Jones terms
    sd = np.power(np.divide(sig,distances), 6)
    lj_potentials = 4 * eps * (sd ** 2 - sd)
    # repeat the grid point indices for each neighbor
    grid_point_indices = np.repeat(np.arange(len(ncounts)), ncounts)

    # Use np.bincount to sum energies for each grid point
    # Weights are the energies, and minlength ensures the result includes all grid points
    v = np.bincount(grid_point_indices, weights=lj_potentials, minlength=len(ncounts))

    # Use np.add.reduceat to sum potentials for each segment
    # v = np.add.reduceat(lj_potentials, segments[:-1])
    # print('Len of energies ', len(v))
    # print('min of energies ', v.min())
    # print('grid point with min energy', grid_points[v.argmin()])
    return v

def interpolate_dgrid(dgrid):
    """
    The function `interpolate_dgrid` interpolates a 3D grid using linear interpolation.

    :param dgrid: The `interpolate_dgrid` function you provided seems to be interpolating a 3D grid
    using linear interpolation. To use this function, you need to pass a 3D grid (`dgrid`) as a
    parameter. The grid should be a numpy array representing the values at different points in
    :return: The function `interpolate_dgrid` returns a RegularGridInterpolator object `dl` that is
    created using the input `dgrid` and linear interpolation method.
    """

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI

    xx = np.linspace(0, 1, dgrid.shape[0])
    yy = np.linspace(0, 1, dgrid.shape[1])
    zz = np.linspace(0, 1, dgrid.shape[2])
    dl = RGI((xx, yy, zz), dgrid, method="linear")

    return dl

def get_wall_windows2(regions, maxima, dgrid, ase_atoms, wall_thickness=2):
    import numpy as np
    from icecream import ic
    from .helper import find_coord_from_indices, interpolate_labels

    wall_windows = []
    shape = regions.shape
    dinterp = interpolate_labels(dgrid)
    for a in [0, 1, 2]:
        ic.disable()
        ic(a)
        llr = regions.take(indices=0, axis=a)  # * left wall
        rrr = regions.take(indices=shape[a] - 1, axis=a)  # * right wall

        map_list = np.unique(list(zip(llr.flatten(), rrr.flatten())), axis=0)
        not_background = [m for m in map_list if np.all(m != 0)]

        # * there should be some non background mappings in the direction
        if len(not_background) > 0:
            map_list = np.vstack([m for m in map_list if np.all(m != 0)])
            flags = np.vstack(
                [
                    [
                        maxima[k - 1][a] > wall_thickness,
                        maxima[v - 1][a] < (shape[a] - wall_thickness),
                    ]
                    for k, v in map_list
                ]
            )
            flags = np.logical_and(flags[:, 0], flags[:, 1])
            # window_indices = [np.vstack(np.where(llr==m[0])).mean(axis=1) for i,m in enumerate(map_list) if flags[i]]
            wregs_l = [
                (np.array([m[0], m[1]]), True, "periodic")
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            wregs_r = [
                (np.array([m[1], m[0]]), True, "periodic")
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            window_indices = [
                np.vstack(np.where(llr == m[0])).mean(axis=1).T
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            ic(window_indices)
            if np.any(flags):
                window_indices_l = np.insert(
                    np.vstack(window_indices),
                    [a],
                    np.zeros((len(window_indices), 1)),
                    axis=1,
                )
                window_indices_r = np.insert(
                    np.vstack(window_indices),
                    [a],
                    (shape[a] - 1) * np.ones((len(window_indices), 1)),
                    axis=1,
                )
                window_radii_l = np.vstack(dinterp(window_indices_l / regions.shape))
                window_radii_r = np.vstack(dinterp(window_indices_r / regions.shape))
                window_coords_l = find_coord_from_indices(
                    window_indices_l, shape=shape, ase_atoms=ase_atoms
                )
                window_coords_r = find_coord_from_indices(
                    window_indices_r, shape=shape, ase_atoms=ase_atoms
                )
                ic(window_coords_l)
                ic(window_coords_r)
                window_rows_l = [
                    [*w, np.array([window_coords_l[i]]), window_radii_l[i]]
                    for i, w in enumerate(wregs_l)
                ]
                window_rows_r = [
                    [*w, np.array([window_coords_r[i]]), window_radii_r[i]]
                    for i, w in enumerate(wregs_r)
                ]
                wall_windows.extend(window_rows_l)
                wall_windows.extend(window_rows_r)

    if len(wall_windows) > 0:
        wall_windows = np.array(wall_windows, dtype=object)
    return wall_windows

############

def rag_from_connections_pixel_multi(regions, connections, maxima, only_use_internal=True):
    """
    This Python function creates a graph representation of regions of interest based on connections and
    maximum values, with an option to include only internal connections.

    :param connections: Connections is a list of tuples where each tuple contains information about a
    connection. Each tuple has the following format: (node_id, is_connected, connection_type)
    :param maxima: The `maxima` parameter in the `rag_from_connections` function seems to represent a
    list of nodes or regions in a graph. It is used to determine the number of nodes in the graph
    created by the function. Each element in `maxima` likely corresponds to a node in the graph
    :param only_use_internal: The `only_use_internal` parameter in the `rag_from_connections` function
    is a boolean parameter that determines whether only internal connections should be used when
    creating the graph. If `only_use_internal` is set to `True`, only connections with the type
    'internal' will be added to the graph, defaults to True (optional)
    :return: The function `rag_from_connections` is returning a NetworkX graph `G` that is constructed
    based on the input connections, maxima, and the optional parameter only_use_internal. The graph `G`
    is built by adding nodes and edges from the input connections based on certain conditions. The nodes
    of the graph have a 'labels' attribute assigned to them.
    """

    import networkx as nx
    import numpy as np

    G = nx.MultiGraph()
    # number_of_regions = np.max([c[0] for c  in connections])
    number_of_nodes = len(maxima)  # np.max([c[0] for c  in connections])
    # print(number_of_nodes)
    # nx.set_node_attributes(G, values={i+1:i+1 for i in range(number_of_nodes)}, name='labels')
    G.add_nodes_from(range(1, number_of_nodes + 1))
    if only_use_internal:  # range stops one short
        G.add_edges_from(
            [c[0] for c in connections if np.logical_and(c[1], c[2] == "internal")]
        )
    else:
        G.add_edges_from([c[0] for c in connections if c[1]])
    # add labels property to rag
    nx.set_node_attributes(
        G, values={i + 1: [i + 1] for i in range(len(G.nodes))}, name="labels"
    )
    for i in G.nodes():
      pixel_counts = np.count_nonzero(regions == i)
      G.nodes[i]['pixel_counts']=pixel_counts
      
    return G

def add_pixel_ratio_to_rag(G, ase_atoms, spacing):
    import ase
    import dask.array as da
    import numpy as np
    [nx, ny, nz] = (ase_atoms.get_cell_lengths_and_angles()[0:3] / spacing).astype(int) + 1
    n=nx * ny * nz
    for i in G.nodes():
        G.nodes[i]['total_pixel_amounts']=n
        G.nodes[i]['pixel_ratio']=G.nodes[i]['pixel_counts']/n
    return G

def egrid_transfer(egrid_K):
    from scipy.constants import Boltzmann, Avogadro
    kb = Boltzmann
    na = Avogadro
    egrid_J=egrid_K*kb
    egrid_kJ=egrid_J/1000
    egrid_kJpermol=egrid_kJ*na
    return egrid_kJpermol 

def add_vdw_hist_to_rag_probability_right(rag, regions, energy_interpolator=None,upper_bound=1000, use_pbc=False, pbc_groups=None, bins_energy=None, **kwargs_egrid):
    import numpy as np
    import networkx as nx
    # if bins_energy is None:
    # bins_energy = np.linspace(-3000, 100, 25)
    if energy_interpolator is None:
        egrid =egrid_from_atoms2(**kwargs_egrid )
        energy_interpolator = get_energy_interpolator(egrid)
    if use_pbc:
        if pbc_groups is None:
            raise ValueError('pbc_groups must be provided if use_pbc=True')

        else:
            energies_list = [get_energies_for_group2(list(
                g), regions, energy_interpolator=energy_interpolator) for g in pbc_groups]
            
            if bins_energy is None:
                bins_energy = np.linspace(
                    np.min(np.hstack(energies_list)), upper_bound, 25)
            
            histograms_bins =[energy_histogram_right_inclusive(e, bins=bins_energy, weights=1/len(e)) for e in energies_list]
            ehistograms = [hb for hb in histograms_bins]
            ebincenters = [(bins_energy[:-1]+bins_energy[1:])/2]

            for i, node in enumerate(rag.nodes()):
                rag.nodes[node]['vdw_hist_y_pbc'] = ehistograms[rag.nodes[node]['pbc_group']]
                rag.nodes[node]['vdw_hist_x_pbc'] = ebincenters

    else:
        energies_list = [get_energies_for_group2(
            [node], regions, energy_interpolator=energy_interpolator) for node in rag.nodes()]
             
        # energies_list = [get_energies_for_group(
        #     [node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for node in rag.nodes()]
        if bins_energy is None:
            bins_energy = np.linspace(
                np.min(np.hstack(energies_list)), upper_bound, 25)
        histograms_bins =[energy_histogram_right_inclusive(e, bins=bins_energy, weights=1/len(e)) for e in energies_list] 
        ehistograms = [hb for hb in histograms_bins]
        ebincenters = [(bins_energy[:-1]+bins_energy[1:])/2]

        for i, node in enumerate(rag.nodes()):
            rag.nodes[node]['vdw_hist_y_pbc'] = ehistograms[i]
            rag.nodes[node]['vdw_hist_x_pbc'] = ebincenters
    return rag

def energy_histogram_right_inclusive(energies,bins,weights):
    import numpy as np
    histogram=np.zeros(len(bins) - 1)
    for energy in energies:
        for i in range(len(bins) - 1):
            if bins[i]<energy<=bins[i+1]:
               histogram[i]+=weights
               break
    return histogram

def add_volume_to_rag(G, ase_atoms):
    import ase
    import dask.array as da
    import numpy as np
    vol = ase_atoms.get_volume()
    vol_L=vol/(1E27)
    for i in G.nodes():
        G.nodes[i]['cell_volume']=vol_L
    return G