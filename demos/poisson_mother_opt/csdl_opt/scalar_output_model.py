
from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from fea import *

class ScalarOutputModel(Model):

    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size_1 = self.fea.total_dofs_u
        self.input_size_2 = self.fea.total_dofs_f
        u = self.declare_variable('u',
                        shape=(self.input_size_1,),
                        val=1.0)
        f = self.declare_variable('f',
                        shape=(self.input_size_2,),
                        val=1.0)
        e = ScalarOutputOperation(fea=self.fea)
        objective = csdl.custom(u, f, op=e)
        self.register_output('objective', objective)


class ScalarOutputOperation(CustomExplicitOperation):
    """
    input: u, f
    output: objective
    """
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size_1 = self.fea.total_dofs_u
        self.input_size_2 = self.fea.total_dofs_f
        self.add_input('u',
                        shape=(self.input_size_1,),
                        val=0.0)
        self.add_input('f',
                        shape=(self.input_size_2,),
                        val=0.0)
        self.add_output('objective')
        self.declare_derivatives('*', '*')
        
    def compute(self, inputs, outputs):
        update(self.fea.u, inputs['u'])
        update(self.fea.f, inputs['f'])
        objective = assembleScalar(self.fea.objective())
        outputs['objective'] = objective

    def compute_derivatives(self, inputs, derivatives):
        update(self.fea.u, inputs['u'])
        update(self.fea.f, inputs['f'])
        dcdu = assembleVector(self.fea.dC_du)
        dcdf = assembleVector(self.fea.dC_df)
        derivatives['objective', 'u'] = dcdu
        derivatives['objective', 'f'] = dcdf
        
if __name__ == "__main__":
    n = 2
    mesh = createUnitSquareMesh(n)
    fea = FEA(mesh)

    f_ex = fea.f_ex
    u_ex = fea.u_ex
    
    from matplotlib import pyplot as plt
    print("CSDL: Running the model...")
    
    sim = Simulator(ScalarOutputModel(fea=fea))
    # setFuncArray(fea.f, getFuncArray(f_ex))
    # sim['u'] = getFuncArray(u_ex)
    sim.run()
    print(sim['objective'])
    
    print("CSDL: Running check_partials()...")
    sim.check_partials()


        
        
        
        
        
        
        
        
        
        
        
        
        
