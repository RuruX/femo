from dolfin import *
import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt
from mpi4py import MPI as pympi

import ufl
from set_fea import *

class uhatComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = self.options['fea']
        self.comm = pympi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        num_el = self.fea.num_elements
        N_fac = 8
        self.nx = nx = num_el*N_fac+1
        self.ny = ny = num_el+1
        self.ind = self.index(nx,ny)
        self.add_input('points', shape=num_el*8+1)
        self.add_output('uhat', shape=self.fea.dof_f)
        """
        This component uses the displacements in y-axis of the bottom nodes
        to linearly interpolate the displacements of all nodes.
        
        points: u_y on the bottom edge
        uhat: displacements of the whole mesh
        
        """
        data, rows, cols = self.partials()
        self.declare_partials('uhat', 'points',
                              val=data,
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs):
        points = inputs['points']
        nx = self.nx
        ny = self.ny
        uhat_y = np.zeros(nx*ny)
        uhat_x = np.zeros(nx*ny)
        y0 = 0.0
        y1 = 1.0
        self.factor = np.linspace(y1,y0,ny)[:-1]

        for i in range(ny-1):
            uhat_y[i*nx:(i+1)*nx] = points*self.factor[i]

        uhat = np.zeros(nx*ny*2)
        ind = self.ind
        for i in range(ny):
            for j in range(nx):
                uhat[(ind[i,j]-1)*2+1] = uhat_y[i*nx+j]

        outputs['uhat'] = uhat

    def partials(self):
        nx = self.nx
        ny = self.ny
        ind = self.ind
        y0 = 0.0
        y1 = 1.0
        self.factor = np.linspace(y1,y0,ny)[:-1]
        data = np.repeat(self.factor,nx)
        cols = np.zeros(nx*(ny-1))
        rows = np.zeros(nx*(ny-1))
        for i in range(ny-1):
            cols[i*nx:(i+1)*nx] = np.arange(nx)
            for j in range(nx):
                rows[i*nx+j] = ind[i,j]*2-1

        return data, rows, cols


    def index(self,nx,ny):
        """ sorting the index of the nodes """
        comm = self.comm
        rank = self.rank
        root = 0
        local_nodes = self.fea.VHAT.tabulate_dof_coordinates()
        local_nodes = np.delete(local_nodes, range(1, local_nodes.shape[0], 2), axis=0)
        sendbuf1 = np.ascontiguousarray(local_nodes[:,0])
        sendbuf2 = np.ascontiguousarray(local_nodes[:,1])
        sendcounts1 = np.array(comm.allgather(local_nodes.shape[0]))
        sendcounts2 = np.array(comm.allgather(local_nodes.shape[0]))
        if rank == root:
            recvbuf1 = np.empty(sum(sendcounts1), dtype=float)
            recvbuf2 = np.empty(sum(sendcounts2), dtype=float)
        else:
            recvbuf1 = None
            recvbuf2 = None
        
        comm.Gatherv(sendbuf1, (recvbuf1, sendcounts1), root)
        comm.Gatherv(sendbuf2, (recvbuf2, sendcounts2), root)
        
        nodes = np.zeros((nx*ny,2))
        nodes[:,0] = comm.bcast(recvbuf1, root)
        nodes[:,1] = comm.bcast(recvbuf2, root)
        
        
        sortdofs = np.zeros(nx*ny)
        
        for i in range(nx*ny):
            sortdofs[i] = nodes[i,0]+nodes[i,1]*nx
        
        ind = np.argsort(sortdofs)+1
        newind = ind.reshape(ny,nx)
        return newind



if __name__ == '__main__':

    num_elements = 3
    N_fac = 8
    nx = num_elements*N_fac+1
    ny = num_elements+1
    
    fea = set_fea(num_elements=num_elements)
    prob = Problem()

    comp = uhatComp(fea=fea)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['uhat'])
    print('check_partials:')
    prob.check_partials(compact_print=True)


# maybe another approach for sorting the indices

#from dolfin import *
#import numpy as np
#
#mesh = UnitSquareMesh(2, 2)
#V = FunctionSpace(mesh, 'CG', 2)
#v = Function(V)
#
#dofmap = V.dofmap()
#nvertices = mesh.ufl_cell().num_vertices()
#
## Set up a vertex_2_dof list
#indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]
#
#vertex_2_dof = dict()
#[vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0),dofmap.cell_dofs(cell.index())[indices])))for cell in cells(mesh)]
#
## Get the vertex coordinates
#X = mesh.coordinates()
#
## Set the vertex coordinate you want to modify
#xcoord, ycoord = 0.5, 0.5
#
## Find the matching vertex (if it exists)
#vertex_idx = np.where((X == (xcoord,ycoord)).all(axis = 1))[0]
#if not vertex_idx:
#    print('No matching vertex!')
#else:
#    vertex_idx = vertex_idx[0]
#    dof_idx = vertex_2_dof[vertex_idx]
#    print(dof_idx)
#    v.vector()[dof_idx] = 1.
#
#plot(v)
#from matplotlib import pyplot as plt
#plt.show()
