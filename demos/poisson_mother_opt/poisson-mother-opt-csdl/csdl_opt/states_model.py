# Import `dolfin` first to avoid segmentation fault
from dolfin import *

from csdl import Model, CustomImplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from fea import *

class StatesModel(Model):

    def initialize(self):
        print("="*40)
        print("CSDL: Running initialize()...")
        print("="*40)
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size = self.fea.total_dofs_f
        self.output_size = self.fea.total_dofs_u
        f = self.declare_variable('f',
                        shape=(self.input_size,),
                        val=np.zeros(self.input_size).reshape(self.input_size,))
        e = StatesOperation(fea=self.fea)
        u = csdl.custom(f, op=e)
        self.register_output('u', u)

class StatesOperation(CustomImplicitOperation):
    """
    input: f
    output: u
    """
    def initialize(self):
        print("="*40)
        print("CSDL: Running initialize()...")
        print("="*40)
        self.parameters.declare('fea')

    def define(self):
        print("="*40)
        print("CSDL: Running define()...")
        print("="*40)

        self.fea = self.parameters['fea']
        self.input_size = self.fea.total_dofs_f
        self.output_size = self.fea.total_dofs_u
        self.add_input('f',
                        shape=(self.input_size,),
                        val=np.zeros(self.input_size).reshape(self.input_size,))
        self.add_output('u',
                        shape=(self.output_size,),
                        val=np.zeros(self.output_size).reshape(self.output_size,))
        self.declare_derivatives('u', 'f')
        self.declare_derivatives('u', 'u')
        self.bcs = self.fea.bc()

    def evaluate_residuals(self, inputs, outputs, residuals):
        print("="*40)
        print("CSDL: Running evaluate_residuals()...")
        print("="*40)
        update(self.fea.f, inputs['f'])
        update(self.fea.u, outputs['u'])

        R = assemble(self.fea.R())
        # self.bcs.apply(R)
        residuals['u'] = R.get_local()

    def solve_residual_equations(self, inputs, outputs):
        print("="*40)
        print("CSDL: Running solve_residual_equations()...")
        print("="*40)
        update(self.fea.f, inputs['f'])
        self.fea.solve()

        outputs['u'] = self.fea.u.vector().get_local()
        update(self.fea.u, outputs['u'])

    def compute_derivatives(self, inputs, outputs, derivatives):
        print("="*40)
        print("CSDL: Running compute_derivatives()...")
        print("="*40)
        update(self.fea.f, inputs['f'])
        update(self.fea.u, outputs['u'])

        self.dRdu = assemble(self.fea.dR_du)
        self.dRdf = assemble(self.fea.dR_df)
        self.A,_ = assemble_system(self.fea.dR_du, self.fea.R(), bcs=self.bcs)

    def compute_jacvec_product(self, inputs, outputs,
                                d_inputs, d_outputs, d_residuals, mode):
        print("="*40)
        print("CSDL: Running compute_jacvec_product()...")
        print("="*40)
        update(self.fea.f, inputs['f'])
        update(self.fea.u, outputs['u'])
        if mode == 'fwd':
            if 'u' in d_residuals:
                if 'u' in d_outputs:
                    update(self.fea.du, d_outputs['u'])
                    d_residuals['u'] += computeMatVecProductFwd(
                            self.dRdu, self.fea.du)
                if 'f' in d_inputs:
                    update(self.fea.df, d_inputs['f'])
                    d_residuals['u'] += computeMatVecProductFwd(
                            self.dRdf, self.fea.df)

        if mode == 'rev':
            if 'u' in d_residuals:
                update(self.fea.dR, d_residuals['u'])
                if 'u' in d_outputs:
                    d_outputs['u'] += computeMatVecProductBwd(
                            self.dRdu, self.fea.dR)
                if 'f' in d_inputs:
                    d_inputs['f'] += computeMatVecProductBwd(
                            self.dRdf, self.fea.dR)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        print("="*40)
        print("CSDL: Running apply_inverse_jacobian()...")
        print("="*40)

        if mode == 'fwd':
            d_outputs['u'] = self.fea.solveLinearFwd(self.A, d_residuals['u'])
        else:
            d_residuals['u'] = self.fea.solveLinearBwd(self.A, d_outputs['u'])



if __name__ == "__main__":
    n = 16
    mesh = UnitSquareMesh(n,n)
    fea = FEA(mesh)

    x = SpatialCoordinate(fea.mesh)
    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    alpha = Constant(1e-6)
    f_analytic = Expression("1/(1+alpha*4*pow(pi,4))*w", w=w, alpha=alpha, degree=3)
    u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
    f_ex = interpolate(f_analytic, fea.VF)
    u_ex = interpolate(u_analytic, fea.V)

    # fea.solveMeshMotion()
    sim = Simulator(StatesModel(fea=fea))
    sim['f'] = f_ex.vector().get_local()
    from matplotlib import pyplot as plt
    print("CSDL: Running the model...")
    sim.run()
    #    sim.visualize_implementation()
    fea.u.vector().set_local(sim['u'])
    state_error = errornorm(u_ex, fea.u)
    print("Error in solve_nonlinear:", state_error)
    plt.figure(1)
    plot(fea.u)
    plt.show()

    print("CSDL: Checking the partial derivatives...")
    sim.check_partials()
