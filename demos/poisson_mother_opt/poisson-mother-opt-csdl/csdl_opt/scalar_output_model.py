# Import `dolfin` first to avoid segmentation fault
from dolfin import *

from csdl import Model, CustomImplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from motor_fea import *

class MagneticStatesModel(Model):

    def initialize(self):
        print("="*40)
        print("CSDL: Running initialize()...")
        print("="*40)
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size = self.fea.total_dofs_uhat
        self.output_size = self.fea.total_dofs_A_z
        uhat = self.declare_variable('uhat',
                        shape=(self.input_size,),
                        val=np.zeros(self.input_size).reshape(self.input_size,))
        iq = self.declare_variable('iq',val=0.0)
        e = MagneticStates(fea=self.fea)
        A_z = csdl.custom(uhat, iq, op=e)
        self.register_output('A_z', A_z)

class MagneticStates(CustomImplicitOperation):
    """
    input: uhat
    output: A_z
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
        self.input_size = self.fea.total_dofs_uhat
        self.output_size = self.fea.total_dofs_A_z
        self.add_input('uhat',
                        shape=(self.input_size,),
                        val=np.zeros(self.input_size).reshape(self.input_size,))
        self.add_input('iq',val=0.0)
        self.add_output('A_z',
                        shape=(self.output_size,),
                        val=np.zeros(self.output_size).reshape(self.output_size,))
        self.declare_derivatives('A_z', 'uhat')
        self.declare_derivatives('A_z', 'iq')
        self.declare_derivatives('A_z', 'A_z')
        self.bcs = self.fea.setBCMagnetostatic()

    def evaluate_residuals(self, inputs, outputs, residuals):
        print("="*40)
        print("CSDL: Running evaluate_residuals()...")
        print("="*40)
        update(self.fea.uhat, inputs['uhat'])
        update(self.fea.iq, inputs['iq'])
        update(self.fea.A_z, outputs['A_z'])

        resMS = assemble(self.fea.resMS())
        self.bcs.apply(resMS)
        residuals['A_z'] = resMS.get_local()

    def solve_residual_equations(self, inputs, outputs):
        print("="*40)
        print("CSDL: Running solve_residual_equations()...")
        print("="*40)
        update(self.fea.uhat, inputs['uhat'])
        updateR(self.fea.iq, inputs['iq'])
        self.fea.solveMagnetostatic()

        outputs['A_z'] = self.fea.A_z.vector().get_local()
        update(self.fea.A_z, outputs['A_z'])

    def compute_derivatives(self, inputs, outputs, derivatives):
        print("="*40)
        print("CSDL: Running compute_derivatives()...")
        print("="*40)
        update(self.fea.uhat, inputs['uhat'])
        updateR(self.fea.iq, inputs['iq'])
        update(self.fea.A_z, outputs['A_z'])

        self.dRdu = assemble(self.fea.dR_du)
        self.dRdf1 = assemble(self.fea.dR_duhat)
        self.dRdf2 = assemble(self.fea.dR_diq)
        self.A,_ = assemble_system(self.fea.dR_du, self.fea.resMS(), bcs=self.bcs)

    def compute_jacvec_product(self, inputs, outputs,
                                d_inputs, d_outputs, d_residuals, mode):
        print("="*40)
        print("CSDL: Running compute_jacvec_product()...")
        print("="*40)

        if mode == 'fwd':
            if 'A_z' in d_residuals:
                if 'A_z' in d_outputs:
                    update(self.fea.du, d_outputs['A_z'])
                    d_residuals['A_z'] += computeMatVecProductFwd(
                            self.dRdu, self.fea.du)
                if 'uhat' in d_inputs:
                    update(self.fea.duhat, d_inputs['uhat'])
                    d_residuals['A_z'] += computeMatVecProductFwd(
                            self.dRdf1, self.fea.duhat)
                if 'iq' in d_inputs:
                    update(self.fea.diq, d_inputs['iq'])
                    d_residuals['A_z'] += computeMatVecProductFwd(
                            self.dRdf2, self.fea.diq)

        if mode == 'rev':
            if 'A_z' in d_residuals:
                update(self.fea.dR, d_residuals['A_z'])
                if 'A_z' in d_outputs:
                    d_outputs['A_z'] += computeMatVecProductBwd(
                            self.dRdu, self.fea.dR)
                if 'uhat' in d_inputs:
                    d_inputs['uhat'] += computeMatVecProductBwd(
                            self.dRdf1, self.fea.dR)
                if 'iq' in d_inputs:
                    d_inputs['iq'] += computeMatVecProductBwd(
                            self.dRdf2, self.fea.dR)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        print("="*40)
        print("CSDL: Running apply_inverse_jacobian()...")
        print("="*40)

        if mode == 'fwd':
            d_outputs['A_z'] = self.fea.solveLinearFwd(self.A, d_residuals['A_z'])
        else:
            d_residuals['A_z'] = self.fea.solveLinearBwd(self.A, d_outputs['A_z'])



if __name__ == "__main__":
    iq = 282.2
    f = open('edge_deformation_data/init_edge_coords_coarse_1.txt', 'r+')
    old_edge_coords = np.fromstring(f.read(), dtype=float, sep=' ')
    f.close()

    f = open('edge_deformation_data/edge_coord_deltas_coarse_1.txt', 'r+')
    edge_deltas = np.fromstring(f.read(), dtype=float, sep=' ')
    f.close()

    
    fea = MotorFEA(mesh_file="mesh_files/motor_mesh_coarse_1",
                                old_edge_coords=old_edge_coords)
    fea.edge_deltas = 0.1*edge_deltas
    
    # fea.solveMeshMotion()
    sim = Simulator(MagneticStatesModel(fea=fea))
    sim['iq'] = iq
    from matplotlib import pyplot as plt
    print("CSDL: Running the model...")
    sim.run()
    #    sim.visualize_implementation()
    fea.A_z.vector().set_local(sim['A_z'])
    plt.figure(1)
    # fea.moveMesh(fea.uhat)
    # plot(fea.A_z)
    plot(fea.A_z)
    # plot(fea.mesh, linewidth=0.1)
    # plot(fea.subdomains_mf)
    plt.show()

    print("CSDL: Checking the partial derivatives...")
    sim.check_partials()
