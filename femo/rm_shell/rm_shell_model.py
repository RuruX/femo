import dolfinx
from femo.fea.fea_dolfinx import FEA
from femo.csdl_alpha_opt.fea_model import FEAModel
from shell_analysis_fenicsx import *

from femo.rm_shell.rm_shell_pde import RMShellPDE
import csdl_alpha as csdl
import ufl


class RMShellModel:
    def __init__(self, mesh: dolfinx.mesh, shells: dict):
        self.mesh = mesh
        self.shells = shells

        self.m, self.rho = 1e-6, 100
        self.set_up_fea()

    def set_up_fea(self):
        mesh = self.mesh
        shells = self.shells
        shell_pde = self.shell_pde = RMShellPDE(mesh)

        E = shells['E']
        nu = shells['nu']
        rho = shells['rho']
        dss = shells['dss'] # custom ds measure for the Dirichlet BC
        dSS = shells['dSS'] # custom ds measure for the Dirichlet BC
        dxx = shells['dxx'] # custom dx measure for the tip displacement
        record = shells['record']


        PENALTY_BC = True

        fea = FEA(mesh)
        fea.PDE_SOLVER = "Newton"
        fea.REPORT = False
        fea.record = record
        fea.linear_problem = True
        # Add input to the PDE problem:
        input_name_1 = 'thicknesses'
        input_function_space_1 = shell_pde.VT
        # input_function_space_1 = FunctionSpace(mesh, ("DG", 0))
        input_function_1 = Function(input_function_space_1)
        # Add input to the PDE problem:
        input_name_2 = 'F_solid'
        input_function_space_2 = shell_pde.VF
        input_function_2 = Function(input_function_space_2)

        # Add state to the PDE problem:
        state_name = 'disp_solid'
        state_function_space = shell_pde.W
        state_function = Function(state_function_space)

        # Simple isotropic material
        g = Function(shell_pde.W)
        with g.vector.localForm() as uloc:
            uloc.set(0.)
        residual_form = shell_pde.pdeRes(h=input_function_1,
                                         w=state_function,
                                         f=input_function_2,
                                         E=E,nu=nu,
                                         penalty=PENALTY_BC, 
                                         dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        output_name_0 = 'compliance'
        u_mid, theta = ufl.split(state_function)
        output_form_0 = shell_pde.compliance(u_mid,input_function_2)
        output_name_1 = 'tip_disp'
        output_form_1 = shell_pde.tip_disp(u_mid,input_function_1, dxx)
        output_name_2 = 'mass'
        output_form_2 = shell_pde.mass(input_function_1, rho)
        output_name_3 = 'elastic_energy'
        output_form_3 = shell_pde.elastic_energy(state_function,input_function_1,E)
        output_name_4 = 'pnorm_stress'


        dx_reduced = ufl.Measure("dx", domain=mesh, 
                                 metadata={"quadrature_degree":4})
        output_form_4 = shell_pde.pnorm_stress(
                        state_function,input_function_1,E,nu,
                        dx_reduced,m=self.m,rho=self.rho,
                        alpha=None,regularization=False)
        # output_name_5 = 'von_Mises_stress'
        output_name_5 = 'stress'
        output_form_5 = shell_pde.von_Mises_stress(
                        state_function,input_function_1,E,nu,surface='Top')

        fea.add_input(input_name_1, input_function_1, init_val=0.001, record=record)
        fea.add_input(input_name_2, input_function_2, record=record)
        fea.add_state(name=state_name,
                        function=state_function,
                        residual_form=residual_form,
                        arguments=[input_name_1, input_name_2])
        fea.add_output(name=output_name_0,
                        type='scalar',
                        form=output_form_0,
                        arguments=[state_name,input_name_2])
        fea.add_output(name=output_name_1,
                        type='scalar',
                        form=output_form_1,
                        arguments=[state_name,input_name_1])
        fea.add_output(name=output_name_2,
                        type='scalar',
                        form=output_form_2,
                        arguments=[input_name_1])
        fea.add_output(name=output_name_3,
                        type='scalar',
                        form=output_form_3,
                        arguments=[input_name_1,state_name])
        fea.add_output(name=output_name_4,
                        type='scalar',
                        form=output_form_4,
                        arguments=[input_name_1,state_name])
        fea.add_field_output(name=output_name_5,
                        form=output_form_5,
                        arguments=[input_name_1,state_name],
                        record=record)
        
        self.fea = fea
        
    def evaluate(self, force_vector: csdl.Variable, thicknesses: csdl.Variable,
                 debug_mode=False) \
                    -> csdl.VariableGroup:
        """
        Args:
        ----------
        [RX]: consider a "shell_inputs" VariableGroup when the aeroelastic coupling is ready
            
            > force_vector: csdl.Variable that contains the force vector applied to the shell model
            > thicknesses: csdl.Variable that contains the thicknesses of the shell model
        
        Returns:
        ----------
        shell_outputs: csdl.VariableGroup that contains the outputs of the shell model

            > disp_solid: (vector) csdl.Variable that contains the displacements of the shell model
            > stress: (vector) csdl.Variable that contains the von Mises stress of the shell model
            > aggregated_stress: (scalar) csdl.Variable that contains the aggregated stress of the shell model
            > compliance: (scalar) csdl.Variable that contains the compliance of the shell model
            > tip_disp: (scalar) csdl.Variable that contains the tip displacement of the shell model
            > mass: (scalar) csdl.Variable that contains the mass of the shell model
        """
        shell_inputs = csdl.VariableGroup()
        shell_inputs.thicknesses = thicknesses

        # Construct the shell model in CSDL
        force_reshaping_model = ForceReshapingModel(shell_pde=self.shell_pde)
        reshaped_force = force_reshaping_model.evaluate(force_vector)
        shell_inputs.F_solid = reshaped_force

        solid_model = FEAModel(fea=[self.fea], fea_name='rm_shell')
        shell_outputs = solid_model.evaluate(shell_inputs, debug_mode=debug_mode)
        
        # disp_extraction_model = DisplacementExtractionModel(shell_pde=self.shell_pde)
        # shell_outputs.disp_extracted = disp_extraction_model.evaluate(shell_outputs.disp_solid)
        
        aggregated_stress_model = AggregatedStressModel(m=self.m, rho=self.rho)
        shell_outputs.aggregated_stress = aggregated_stress_model.evaluate(shell_outputs.pnorm_stress)
        return shell_outputs

        
class AggregatedStressModel:
    def __init__(self, m: float, rho: int):
        self.m = m
        self.rho = rho

    def evaluate(self, pnorm_stress: csdl.Variable):
        aggregated_stress = 1/self.m*pnorm_stress**(1/self.rho)
        return aggregated_stress

class DisplacementExtractionModel:
    '''
    Extract and reshape displacement vector into matrix
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, disp_vec: csdl.Variable):
        shell_pde = self.shell_pde

        disp_extraction_mats = shell_pde.construct_nodal_disp_map()
        # Both vector or tensors need to be numpy arrays
        shape = shell_pde.mesh.geometry.x.shape

        # contains nodal displacements only (CG1)
        nodal_disp_vec = csdl.matvec(disp_extraction_mats.todense(), disp_vec)
        nodal_disp_mat = csdl.reshape(nodal_disp_vec, new_shape=(shape[1],shape[0]))

        output = csdl.transpose(nodal_disp_mat)
        return output

class ForceReshapingModel:
    '''
    Reshape force matrix to vector
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, nodal_force_mat: csdl.Variable):
        shell_pde = self.shell_pde
        dummy_func = Function(shell_pde.VF)
        size = len(dummy_func.x.array)
        output = csdl.reshape(nodal_force_mat, shape=(size,))

        return output
