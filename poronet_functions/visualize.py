def get_isosurface_mesh(data, isoval,ase_atoms, spacing=(1,1,1), stepsize=1 ):
    """
    The function `get_isosurface_mesh` generates a triangular mesh representing an isosurface from input
    data using the marching cubes algorithm.
    
    :param data: The `data` parameter in the `get_isosurface_mesh` function is likely a 3D array
    representing the volumetric data from which the isosurface mesh will be generated. This data could
    be obtained from various sources such as medical imaging, scientific simulations, or any other 3D
    :param isoval: The `isoval` parameter in the `get_isosurface_mesh` function represents the isovalue
    at which the isosurface is extracted from the input data. It is a scalar value that determines the
    surface geometry based on the data values
    :param ase_atoms: The `ase_atoms` parameter in the `get_isosurface_mesh` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their properties. It typically
    includes information such as atomic positions, unit cell dimensions, and atomic species. This object
    is used to define the
    :param spacing: The `spacing` parameter in the `get_isosurface_mesh` function represents the voxel
    spacing in each dimension for the input `data`. It is a tuple of three values representing the
    spacing along the x, y, and z axes respectively. This parameter determines the resolution of the
    isosurface mesh
    :param stepsize: The `stepsize` parameter in the `get_isosurface_mesh` function you provided
    controls the step size used during the marching cubes algorithm, defaults to 1 (optional)
    :return: The function `get_isosurface_mesh` returns a trimesh object representing the isosurface
    mesh generated from the input data at the specified isovalue.
    """

    import numpy as np
    from skimage.measure import marching_cubes
    from ase.geometry import complete_cell 
    cell = complete_cell(ase_atoms.get_cell()).T
    verts, faces, vertex_normals, _ = marching_cubes(data, isoval,spacing=spacing, step_size=stepsize)
    verts = np.dot(cell, (verts/data.shape).T).T # * transform to cartesian coordinates
    import trimesh 
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals)
    return mesh


def get_isosurface_submeshes(dgrid, isoval, ase_atoms, region_labels,
                            steps=(1, 1, 1), stepsize=1, colorscale=None):
    """Get submeshes of isosurface based on region labels.
    
    Args:
        dgrid: Distance grid
        region_labels: Region labels array
        ase_atoms: ASE atoms object
        isoval: Isovalue for surface
        steps: Tuple of steps in x,y,z (default: (1,1,1))
        stepsize: Step size for interpolation (default: 1)
    
    Returns:
        List of submeshes corresponding to each region
    """
    
    from .helper import get_fractional_coordinates, interpolate_labels
    import numpy as np
    import plotly.express as px
    dist_iso_surf = get_isosurface_mesh(dgrid, isoval, ase_atoms)
    vertices = dist_iso_surf.vertices
    faces = dist_iso_surf.faces
    
    # Find fractional coordinates of vertices
    frac_vertices = get_fractional_coordinates(vertices, ase_atoms)
    
    # Make interpolator for regions with PBC
    rpbc_interp = interpolate_labels(region_labels)
    
    # Assign region labels to vertices
    rl_verts = rpbc_interp(frac_vertices % 1)
    
    # Get faces with same region label for all vertices
    flags_submeshes = np.vstack([
        np.all(rl_verts[faces] == r, axis=1) 
        for r in np.unique(region_labels)
    ])
    
    # Create submeshes for each region (excluding background r=0)
    dist_iso_submeshes = [
        dist_iso_surf.submesh([np.where(f)[0]], append=True) 
        for f in flags_submeshes[1:]
    ]
    # let's add some vertex colors before sendingthis over
    if colorscale is None:
        colorscale = px.colors.qualitative.Pastel
    def parse_rgb(rgb_str):
        # Extract RGB values from 'rgb(r, g, b)' string
        rgb = rgb_str.strip("rgb()").split(",")
        return [int(x) for x in rgb] + [255]
    sampled_colors = px.colors.sample_colorscale(colorscale, np.unique(region_labels)/region_labels.max())
    mesh_colors = [parse_rgb(s) for s in sampled_colors]
    print('mesh colors', mesh_colors)
    for i,m in enumerate(dist_iso_submeshes):
        m.visual.vertex_colors = mesh_colors[i] 
    return dist_iso_submeshes
