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

        self.add_input('f',
            shape=len(self.fea.F1.dofmap().dofs())+len(self.fea.F2.dofmap().dofs()))
        self.add_input('displacements',
            shape=len(self.fea.V1.dofmap().dofs())+len(self.fea.V2.dofmap().dofs()))
        self.add_output('objective')
        self.declare_partials('objective', 'displacements')
        self.declare_partials('objective', 'f')

    def compute(self, inputs, outputs):
        f = inputs['f']
        u = inputs['displacements']
        self.fea.updateF(f)
        self.fea.updateU(u)
        
        outputs['objective'] = self.fea.objective()
#        print(outputs['objective'])

    def compute_partials(self, inputs, partials):
        f = inputs['f']
        u = inputs['displacements']
        self.fea.updateF(f)
        self.fea.updateU(u)
        
        dJ_du1 = assemble(derivative(self.fea.err(self.fea.u1,self.fea.d1,self.fea.f1), self.fea.u1))
        dJ_du2 = assemble(derivative(self.fea.err(self.fea.u2,self.fea.d2,self.fea.f2), self.fea.u2))
        dJ_du1_array = dJ_du1.get_local()
        dJ_du2_array = dJ_du2.get_local()
        dJ_df1 = assemble(derivative(self.fea.err(self.fea.u1,self.fea.d1,self.fea.f1), self.fea.f1))
        dJ_df2 = assemble(derivative(self.fea.err(self.fea.u2,self.fea.d2,self.fea.f2), self.fea.f2))
        dJ_df1_array = dJ_df1.get_local()
        dJ_df2_array = dJ_df2.get_local()
        
        partials['objective', 'displacements'] = np.append([dJ_du1_array],[dJ_du2_array])
        partials['objective', 'f'] = np.append([dJ_df1_array],[dJ_df2_array])

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
    print(prob['f'])
    print('check_partials:')
    prob.check_partials(compact_print=True)


