import dolfinx
from mpi4py import MPI
import numpy as np
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 1)
Q = dolfinx.VectorFunctionSpace(mesh, ("DG", 1))
num_dofs = mesh.topology.index_map(mesh.topology.dim).size_local
block_size = Q.dofmap.index_map_bs 
local_cell_dofs = np.zeros((num_dofs, Q.dofmap.dof_layout.num_dofs * block_size))
print(Q.dofmap.bs,num_dofs, block_size)
for i in range(num_dofs):
    cell_blocks = Q.dofmap.cell_dofs(i)
    for (j,dof) in enumerate(cell_blocks):
        for k in range(block_size):
            local_cell_dofs[i, j*block_size+k] = dof*block_size + k
print(local_cell_dofs)
