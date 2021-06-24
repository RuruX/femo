from __future__ import division

from dolfin import *
import ufl
from set_fea import *
#from set_fea_3d import *
import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt
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
        self.add_input('f', shape=self.fea.dof_f)
        self.add_input('displacements', shape=self.fea.dof_u)
        self.add_output('objective')
        self.declare_partials('objective', 'displacements')
        self.declare_partials('objective', 'f')

    def compute(self, inputs, outputs):
        self.fea.updateF(inputs['f'][self.global_ind_f])
        self.fea.updateU(inputs['displacements'][self.global_ind_u])
        outputs['objective'] = assemble(self.fea.objective(self.fea.u,self.fea.f))

    def compute_partials(self, inputs, partials):
        self.fea.updateF(inputs['f'][self.global_ind_f])
        self.fea.updateU(inputs['displacements'][self.global_ind_u])
        rank = self.rank
        comm = self.comm
        dJ_du_petsc = v2p(assemble(self.fea.dJ_du))
        dJ_df_petsc = v2p(assemble(self.fea.dJ_df))
        dJ_du_petsc.assemble()
        dJ_du_petsc.ghostUpdate()
        dJ_df_petsc.assemble()
        dJ_df_petsc.ghostUpdate()
        
        root = 0
        sendbuf1 = dJ_du_petsc.getArray()
        sendbuf2 = dJ_df_petsc.getArray()
        sendcounts1 = np.array(comm.allgather(self.local_states_size))
        sendcounts2 = np.array(comm.allgather(self.local_var_size))
        
        if rank == root:
            recvbuf1 = np.empty(sum(sendcounts1), dtype=float)
            recvbuf2 = np.empty(sum(sendcounts2), dtype=float)
        else:
            recvbuf1 = None
            recvbuf2 = None
        comm.Gatherv(sendbuf1, (recvbuf1, sendcounts1), root)
        comm.Gatherv(sendbuf2, (recvbuf2, sendcounts2), root)
        partials['objective', 'displacements'] = comm.bcast(recvbuf1, root)
        partials['objective', 'f'] = comm.bcast(recvbuf2, root)

    
if __name__ == '__main__':

    num_elements = 2
    fea = set_fea(num_elements=num_elements)
    prob = Problem()

    comp = ObjectiveComp(fea=fea)
    prob.model = comp
    prob.setup()
    prob.run_model()
    print('*'*20)
    print('Processor #', pympi.COMM_WORLD.Get_rank(), '-- check_partials:')
    print('*'*20)
#    prob.check_partials(compact_print=False)

    data = prob.check_partials(out_stream=None)

    from openmdao.utils.assert_utils import assert_check_partials
    try:
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)
    except ValueError as err:
        print(str(err))

