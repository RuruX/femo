"""
The ``pyvista_plotter`` module
------------------------------
Contains the basic settings for 3D interactive plotting in DolfinX, by `pyvista` module.
Installation of the external package is required for usage of this module.
To install, run the command in terminal `pip3 install pyvista`.
"""

from dolfinx import *
import dolfinx.plot
import numpy as np
### =================  3D Plotting with pyvista ==================================== ====================================================================================

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo; install pyvista by `pip install pyvista`")
    exit(0)

def plotter_3d(mesh, vertex_values):
    #pyvista.OFF_SCREEN = True
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        from pyvista.utilities.xvfb import start_xvfb
        start_xvfb(wait=0.1)

    # Set some global options for all plots
    transparent = False
    figwidth = 1200
    figlength = 800
    pyvista.rcParams["background"] = [0.5, 0.5, 0.5]


    # Extract mesh data from dolfin-X (only plot cells owned by the
    # processor) and create a pyvista UnstructuredGrid
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_entities = np.arange(num_cells, dtype=np.int32)
    pyvista_cells, cell_types, mesh_geometry_x = dolfinx.plot.create_vtk_mesh(
                                    mesh, mesh.topology.dim, cell_entities)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh_geometry_x)

    # Compute the function values at the vertices, this is equivalent to a
    # P1 Lagrange interpolation, and can be directly attached to the Pyvista
    # mesh. Discard complex value if running dolfin-X with complex PETSc as
    # backend


    if np.iscomplexobj(vertex_values):
        vertex_values = vertex_values.real

    # Create point cloud of vertices, and add the vertex values to the cloud
    grid.point_data["u"] = vertex_values
    grid.set_active_scalars("u")

    # Create a new plotter, and plot the values as a surface over the mesh
    plotter = pyvista.Plotter()
    plotter.add_text("Function values over the surface of a mesh",
                     position="upper_edge", font_size=14, color="black")

    # Define some styling arguments for a colorbar
    sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1,
                 position_y=0.05, fmt="%1.2e",
                 title_font_size=40, color="black", label_font_size=25)


    # Add mesh with edges
    plotter.add_mesh(grid, show_edges=True, scalars="u", scalar_bar_args=sargs)
    return plotter
    
    
    
    
    
    
