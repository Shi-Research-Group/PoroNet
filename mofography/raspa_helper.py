def get_atoms_list_for_pdb(pdb_file, index_list=None, symbol_map=None):
    """
    It reads a RASPA pdb file and returns a list of ASE atoms objects
    
    :param pdb_file: the name of the pdb file
    :param index_list: a list of indices of the configurations you want to extract from the pdb file. If
    you want all of them, just leave it as None
    :param symbol_map: Optional dictionary mapping custom symbols to standard element symbols.
    :return: A list of atoms objects.
    """

    from ase import Atoms
    import numpy as np
    from .data import read_raspa_pdb
    from rich.progress import track

    config_data = read_raspa_pdb(pdb_file, symbol_map=symbol_map)
    atoms_list = []
    
    if index_list is None:
        index_list = np.arange(len(config_data['coords']))
    
    for index in track(index_list, description='Reading Configs'):
        coords = config_data['coords'][index]
        sym = config_data['symbols'][index]
        
        # Check if coords is empty or not
        if coords.size == 0:
            print(f"Warning: No coordinates found for index {index}. Skipping this configuration.")
            continue
        
        # Ensure symbols are in the correct format
        sym = [str(c) for c in sym]
        
        # Check if cell dimensions are available
        if index < len(config_data['cells']):
            cell = config_data['cells'][index]
        else:
            print(f"Warning: No cell dimensions found for index {index}. Skipping this configuration.")
            continue
        
        # Convert to ASE Atoms object
        try:
            atoms = Atoms(symbols=sym, positions=coords, cell=cell, pbc=[1, 1, 1])
            atoms_list.append(atoms)
        except Exception as e:
            print(f"Error creating Atoms object for index {index}: {e}")
    
    return atoms_list


def apply_region_map_to_raspa_pdb_wrapped(pdb_file, regions_cluster, unit_cell, natoms_per_molecule=1, index_list=None, symbol_map=None):

    """
    Applies a region map to a RASPA PDB file, wrapping coordinates to a single unit cell.

    :param pdb_file: the pdb file you want to analyze
    :param regions_cluster: the region map for a single unit cell
    :param unit_cell: the unit cell matrix or ASE Atoms object containing the unit cell information
    :param natoms_per_molecule: number of atoms per molecule. This is currently only implemented for 1,
    defaults to 1 (optional)
    :param index_list: list of indices of atoms to use in the pdb file. If None, all atoms are used
    :return: A dictionary with the number of atoms in each cluster type and the total number of atoms.
    """
    import numpy as np
    from .helper import interpolate_labels
    from ase.geometry import wrap_positions
    from ase import Atoms

    atoms_list = get_atoms_list_for_pdb(pdb_file, index_list=index_list, symbol_map=symbol_map)
    unique_cluster_types = np.unique(regions_cluster)
    # remove background 0
    unique_cluster_types = unique_cluster_types[unique_cluster_types!=0]

    # * make a dictionary of cluster types to store their counts and means
    cluster_type_counts = {str(c-1):[] for c in unique_cluster_types}
    cluster_type_counts['total']= []

    # this is the interpolator function for a single unit cell
    region_interp = interpolate_labels(regions_cluster)

    if natoms_per_molecule != 1:
        print('ERROR: natoms_per_molecule is not 1, currently only implemented for 1')

    # Get the unit cell
    if isinstance(unit_cell, Atoms):
        cell = unit_cell.get_cell()
    else:
        cell = unit_cell

    # loop over the snapshots in the pdb
    for snap in atoms_list:
        # Get the positions from the snapshot
        positions = snap.get_positions()
        
        # Wrap the positions to the unit cell
        wrapped_positions = wrap_positions(positions, cell)
        
        # Convert to fractional coordinates
        scaled_positions = cell.scaled_positions(wrapped_positions)

        # shift any scaled positions if they are out of bounds
        scaled_positions = scaled_positions%1
        # Apply the region map
        cluster_indices = region_interp(scaled_positions)
        
        number_total = len(cluster_indices)
        for c in unique_cluster_types:
            cluster_type_counts[str(c-1)].append(len(np.where(cluster_indices == c)[0]))
        cluster_type_counts['total'].append(number_total)

    for c in unique_cluster_types:
        cluster_type_counts[str(c-1)+'_mean'] = np.mean(cluster_type_counts[str(c-1)])
        cluster_type_counts[str(c-1)+'_std'] = np.std(cluster_type_counts[str(c-1)])
    cluster_type_counts['total_mean'] = np.mean(cluster_type_counts['total'])

    return cluster_type_counts
