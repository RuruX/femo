from __future__ import division
from six.moves import range

import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea

class ObjectiveComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = self.options['fea']
        
        self.add_input('displacements', shape=len(self.fea.V.dofmap().dofs()))
        self.add_input('uhat', shape=len(self.fea.VHAT.dofmap().dofs()))
        self.add_output('objective')
        self.declare_partials('objective', 'displacements')
        self.declare_partials('objective', 'uhat')
    
    
    def compute(self, inputs, outputs):
        self.fea.u.vector().set_local(inputs['displacements'])
        self.fea.uhat.vector().set_local(inputs['uhat'])
        
        outputs['objective'] = assemble(self.fea.objective(self.fea.u,self.fea.uhat))

    def compute_partials(self, inputs, partials):
        self.fea.u.vector().set_local(inputs['displacements'])
        self.fea.uhat.vector().set_local(inputs['uhat'])

        dJ_du = assemble(derivative(self.fea.objective(self.fea.u,self.fea.uhat), self.fea.u))
        dJ_du_array = dJ_du.get_local()
        dJ_df = assemble(derivative(self.fea.objective(self.fea.u,self.fea.uhat), self.fea.uhat))
        dJ_df_array = dJ_df.get_local()
        partials['objective', 'displacements'] = dJ_du_array
        partials['objective', 'uhat'] = dJ_df_array

if __name__ == '__main__':
    
    num_elements = 5
    fea = set_fea(num_elements=num_elements)
    prob = Problem()
    
    comp = ObjectiveComp(fea=fea)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['displacements'])
    print(prob['uhat'])
    print('check_partials:')
    prob.check_partials(compact_print=True)



