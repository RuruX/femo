from __future__ import division
from six.moves import range

import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import *
from mpi4py import MPI as pympi



class ObjectiveComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('fea')
        
    def setup(self):
        self.fea = self.options['fea']
        self.comm = pympi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.local_var_size = self.fea.local_dof_f
        self.local_states_size = self.fea.local_dof_u
        self.global_ind_f = self.fea.ind_f
        self.global_ind_u = self.fea.ind_u
        self.add_input('displacements', shape=self.fea.dof_u)
        self.add_output('objective')
        self.declare_partials('objective', 'displacements')

    def compute(self, inputs, outputs):
        update(self.fea.u, inputs['displacements'][self.global_ind_u])
        outputs['objective'] = assemble(self.fea.objective(self.fea.u,self.fea.uhat))

    def compute_partials(self, inputs, partials):
        update(self.fea.u, inputs['displacements'][self.global_ind_u])
        rank = self.rank
        comm = self.comm
        dJ_du_petsc = v2p(assemble(self.fea.dJ_du))
        dJ_du_petsc.assemble()
        dJ_du_petsc.ghostUpdate()
        
        root = 0
        sendbuf1 = dJ_du_petsc.getArray()
        sendcounts1 = np.array(comm.allgather(self.local_states_size))
        
        if rank == root:
            recvbuf1 = np.empty(sum(sendcounts1), dtype=float)
        else:
            recvbuf1 = None
        comm.Gatherv(sendbuf1, (recvbuf1, sendcounts1), root)
        partials['objective', 'displacements'] = comm.bcast(recvbuf1, root)


if __name__ == '__main__':
    
    num_elements = 3
    fea = set_fea(num_elements=num_elements)
    prob = Problem()
    
    comp = ObjectiveComp(fea=fea)
    prob.model = comp
    prob.setup()
    fea.uhat.vector().set_local(np.random.rand(comp.fea.dof_f))
    prob.run_model()    
    print('check_partials:')
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
#    print(prob['displacements'])
#    print(prob['uhat'])




