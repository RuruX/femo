# Import `dolfin` first to avoid segmentation fault
from dolfin import *

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
        self.input_size = self.fea.total_dofs_u
        u = self.declare_variable('u',
                        shape=(self.input_size,),
                        val=np.zeros(self.input_size).reshape(self.input_size,))

        e = ScalarOutputOperation(fea=self.fea)
        objective = csdl.custom(u, op=e)
        self.register_output('objective', objective)


class ScalarOutputOperation(CustomExplicitOperation):
    """
    input: u
    output: objective, magnet_area, steel_area
    """
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size = self.fea.total_dofs_u
        self.add_input('u',
                        shape=(self.input_size,),
                        val=0.0)
        self.add_output('objective')
        self.declare_derivatives('*', '*')
        
    def compute(self, inputs, outputs):
        update(self.fea.u, inputs['u'])
        objective = assemble(self.fea.objective())
        outputs['objective'] = objective

    def compute_derivatives(self, inputs, derivatives):
        update(self.fea.u, inputs['u'])
        dcdu = assemble(derivative(self.fea.objective(), self.fea.u))
        derivatives['objective', 'u'] = dcdu
        
if __name__ == "__main__":
    n = 16
    mesh = UnitSquareMesh(n, n)
    fea = FEA(mesh)
    x = SpatialCoordinate(fea.mesh)
    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    alpha = Constant(1e-6)
    f_analytic = Expression("1/(1+alpha*4*pow(pi,4))*w", w=w, alpha=alpha, degree=3)
    u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
    f_ex = interpolate(f_analytic, fea.VF)
    u_ex = interpolate(u_analytic, fea.V)
    
    from matplotlib import pyplot as plt
    print("CSDL: Running the model...")
    
    sim = Simulator(ScalarOutputModel(fea=fea))
    fea.f.assign(f_ex)
    sim['u'] = u_ex.vector().get_local()
    sim.run()
    print(sim['objective'])
    
    # print("CSDL: Running check_partials()...")
    # sim.check_partials()


        
        
        
        
        
        
        
        
        
        
        
        
        
