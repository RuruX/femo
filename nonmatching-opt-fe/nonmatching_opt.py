from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 
from dolfin import *
from set_fea import set_fea
from states_comp import StatesComp 
from objective_comp import ObjectiveComp


class NonmatchingGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        fea = set_fea(self.options['num_elements'])

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('f',
                shape=len(fea.F1.dofmap().dofs())+len(fea.F2.dofmap().dofs()))
        self.add_subsystem('inputs_comp', inputs_comp)
        comp_1 = StatesComp(fea=fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=fea)
        self.add_subsystem('objective_comp', comp_2)
        self.connect('inputs_comp.f', 'states_comp.f')
        self.connect('states_comp.displacements', 'objective_comp.displacements')
        self.connect('inputs_comp.f', 'objective_comp.f')
        self.add_design_var('inputs_comp.f')
        self.add_objective('objective_comp.objective')


if __name__ == '__main__':

    num_el = 16
    prob = om.Problem(model=NonmatchingGroup(num_elements=num_el))

    prob.driver = om.ScipyOptimizeDriver()
    
#----------------------------- Run Optimization -------------------------------

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"

    prob.setup()
    prob.run_driver()

#----------------------------- Run Model Test -------------------------------
    from mshr import *
    from petsc4py import PETSc
    import sympy as sy
    from sympy.printing import ccode

    dx = dx(metadata={"quadrature_degree":2})

    N = num_el
    mesh_1 = UnitSquareMesh(N,N)
    mesh_2 = UnitSquareMesh(N+7,N)
    ALE.move(mesh_2,Constant((0,-1)))
    x1 = SpatialCoordinate(mesh_1)
    x2 = SpatialCoordinate(mesh_2)
    F1 = FunctionSpace(mesh_1,"DG",0)
    F2 = FunctionSpace(mesh_2,"DG",0)

    x_ = sy.Symbol('x[0]')
    y_ = sy.Symbol('x[1]')
    u_ = sy.sin(pi*x_)*sy.sin(pi*(y_-x_))*(sy.sin(pi*x_)*sy.sin(pi*x_))*(sy.cos(0.5*pi*y_)*sy.cos(0.5*pi*y_))
    f_ = - sy.diff(u_, x_, x_) - sy.diff(u_, y_, y_) + u_
    f = Expression(ccode(f_), degree=2)
    f_ex_1 = interpolate(f,F1)
    f_ex_2 = interpolate(f,F2)
    f_ex = np.append(f_ex_1.vector().get_local(), f_ex_2.vector().get_local())
#
#    prob.setup()
#    prob['inputs_comp.f'] = f_ex
#    prob.run_model()

#------------------------------ Print Out Results -----------------------------



    fea = set_fea(num_el)
    u = prob['states_comp.displacements']
    fea.u1.vector().set_local(u[:len(fea.V1.dofmap().dofs())])
    fea.u2.vector().set_local(u[-len(fea.V2.dofmap().dofs()):])
    x1 = SpatialCoordinate(fea.mesh_1)
    e1 = fea.u1-fea.u_ex(x1)
    print(sqrt(assemble((e1**2)*fea.dx)))
    print(fea.u1.vector().get_local())
    from matplotlib import pyplot as plt
    plt.figure()
    plot(fea.u1)
    plt.show()
    plot(fea.u2)
    plt.show()
