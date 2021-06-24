from __future__ import division
from six.moves import range

import numpy as np
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea

class ConstraintComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = self.options['fea']
        self.lam = 1.
        
        self.add_input('uhat', shape=len(self.fea.VHAT.dofmap().dofs()))
        self.add_output('constraint')
        
        self.declare_partials('constraint', 'uhat')
    
    def compute(self, inputs, outputs):
        self.fea.uhat.vector().set_local(inputs['uhat'])
        outputs['constraint'] = assemble(self.fea.constraint(self.fea.uhat,self.lam))

    def compute_partials(self, inputs, partials):
        self.fea.uhat.vector().set_local(inputs['uhat'])
        dC_df = assemble(derivative(self.fea.constraint(self.fea.uhat,self.lam), self.fea.uhat))
        
        dC_df_array = dC_df.get_local()
        partials['constraint', 'uhat'] = dC_df_array



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

