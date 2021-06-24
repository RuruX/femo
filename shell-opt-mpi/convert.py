import meshio

#mesh = meshio.read(
#    "conventional-co3.STL",  # string, os.PathLike, or a buffer/open file
#    file_format="stl"  # optional if filename is a path; inferred from extension
#)
#
#msh = meshio.read("conventional-co3.STL")
#meshio.write("mesh2.xdmf",msh)

mesh = meshio.read(
    "plate.stl",  # string, os.PathLike, or a buffer/open file
    file_format="stl"  # optional if filename is a path; inferred from extension
)

msh = meshio.read("plate.stl")
meshio.write("plate_quad.xdmf",msh)
