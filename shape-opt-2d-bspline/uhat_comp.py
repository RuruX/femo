from __future__ import division
from six.moves import range

import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea

class uhatComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = self.options['fea']
        num_el = self.fea.num_elements
        N_fac = 8
        self.nx = num_el*N_fac+1
        self.ny = num_el+1
        self.add_input('points', shape=num_el*8+1)
        self.add_output('uhat', shape=len(self.fea.VHAT.dofmap().dofs()))

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
        ind = self.index(nx,ny)
        for i in range(ny):
            for j in range(nx):
                uhat[(ind[i,j]-1)*2+1] = uhat_y[i*nx+j]

        outputs['uhat'] = uhat

    def partials(self):
        nx = self.nx
        ny = self.ny
        ind = self.index(nx,ny)
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

        return data, rows, cols,


    def index(self,nx,ny):
    #sorting the index of the nodes

        nodes = self.fea.VHAT.tabulate_dof_coordinates()
        sortdofs = np.zeros(nx*ny)
        for i in range(nx*ny):
            sortdofs[i] = nodes[2*i,0]+nodes[2*i,1]*nx
        
        ind = np.argsort(sortdofs)+1
        newind = ind.reshape(ny,nx)
        return newind



if __name__ == '__main__':

    num_elements = 2
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
