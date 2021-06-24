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
        self.add_input('f', shape=len(self.fea.F.dofmap().dofs()))
        self.add_output('objective')
        self.declare_partials('objective', 'displacements')
        self.declare_partials('objective', 'f')


    def compute(self, inputs, outputs):
        self.fea.u.vector().set_local(inputs['displacements'])
        self.fea.f.vector().set_local(inputs['f'])
        outputs['objective'] = assemble(self.fea.objective(self.fea.u,self.fea.f))

    def compute_partials(self, inputs, partials):
        self.fea.u.vector().set_local(inputs['displacements'])
        self.fea.f.vector().set_local(inputs['f'])
        dJ_du = assemble(derivative(self.fea.objective(self.fea.u,self.fea.f), self.fea.u))
        dJ_df = assemble(derivative(self.fea.objective(self.fea.u,self.fea.f), self.fea.f))
        partials['objective', 'displacements'] = dJ_du.get_local()
        partials['objective', 'f'] = dJ_df.get_local()

if __name__ == '__main__':

    num_elements = 2
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


