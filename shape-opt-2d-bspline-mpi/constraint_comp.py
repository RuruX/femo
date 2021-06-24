from dolfin import *
import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt
from mpi4py import MPI as pympi

import ufl
from set_fea import *

class ConstraintComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = self.options['fea']
        self.comm = pympi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lam = 1.
        self.fea.lam.interpolate(Constant(self.lam))
        self.local_var_size = self.fea.local_dof_f
        self.global_ind_f = self.fea.ind_f
        self.add_input('uhat', shape=self.fea.dof_f)
        self.add_output('constraint')
        
        self.declare_partials('constraint', 'uhat')
    
    def compute(self, inputs, outputs):
        update(self.fea.uhat, inputs['uhat'][self.global_ind_f])
        outputs['constraint'] = assemble(self.fea.constraint(self.fea.uhat,self.fea.lam))

    def compute_partials(self, inputs, partials):
        update(self.fea.uhat, inputs['uhat'][self.global_ind_f])
        rank = self.rank
        comm = self.comm
        dC_df_petsc = v2p(assemble(self.fea.dC_df))
        dC_df_petsc.assemble()
        dC_df_petsc.ghostUpdate()
        
        root = 0
        sendbuf = dC_df_petsc.getArray()
        sendcounts = np.array(comm.allgather(self.local_var_size))
        if rank == root:
            recvbuf = np.empty(sum(sendcounts), dtype=float)
        else:
            recvbuf = None
        comm.Gatherv(sendbuf, (recvbuf, sendcounts), root)
        partials['constraint', 'uhat'] = comm.bcast(recvbuf, root)


if __name__ == '__main__':
    
    num_elements = 5
    fea = set_fea(num_elements=num_elements)
    prob = Problem()
    
    comp = ConstraintComp(fea=fea)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['constraint'])
    print('check_partials:')
    prob.check_partials(compact_print=True)

