from femo.fea.fea_dolfinx import FEA
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel, OutputFieldModel
from shell_analysis_fenicsx import *
from shell_analysis_fenicsx.read_properties import readCLT, sortIndex
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import basix
import scipy.sparse as sp
import csdl
import numpy as np

from shell_pde import (ShellModule, ForceReshapingModel, DisplacementExtractionModel,
                        AggregatedStressModel)

class ShellResidual(ShellModule):
    '''
    Dynamic shell model

    Output:
    - residual
    '''
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties

    def define(self):
        pde = self.parameters['pde']
        shell_mesh = pde.mesh
        shells = self.parameters['shells']
        shell_name = list(shells.keys())[0]   # this is only taking the first mesh added to the solver.

        E = shells[shell_name]['E']
        nu = shells[shell_name]['nu']
        rho = shells[shell_name]['rho']
        dss = shells[shell_name]['dss']
        dSS = shells[shell_name]['dSS']
        dxx = shells[shell_name]['dxx']
        g = shells[shell_name]['g']

        PENALTY_BC = True


        fea = FEA(shell_mesh)
        fea.PDE_SOLVER = "Newton"
        fea.initialize = True
        fea.linear_problem = True
        # Add input to the PDE problem:
        input_name_1 = shell_name+'_thicknesses'
        input_function_space_1 = pde.VT
        # input_function_space_1 = FunctionSpace(shell_mesh, ("DG", 0))
        input_function_1 = Function(input_function_space_1)
        # Add input to the PDE problem:
        input_name_2 = 'F_solid'
        input_function_space_2 = pde.VF
        input_function_2 = Function(input_function_space_2)

        # Add state to the PDE problem:
        state_name = 'disp_solid'
        state_function_space = pde.W
        state_function = Function(state_function_space)


        # Simple isotropic material
        residual_form = pde.pdeRes(input_function_1,state_function,
                                input_function_2,E,nu,
                                penalty=PENALTY_BC, dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        output_name_1 = 'compliance'
        u_mid, theta = ufl.split(state_function)
        output_form_1 = pde.compliance(u_mid,input_function_1, dxx)
        output_name_2 = 'mass'
        output_form_2 = pde.mass(input_function_1, rho)
        output_name_3 = 'elastic_energy'
        output_form_3 = pde.elastic_energy(state_function,input_function_1,E)
        output_name_4 = 'pnorm_stress'
        m, rho = 1e-6, 100
        dx_reduced = ufl.Measure("dx", domain=shell_mesh, metadata={"quadrature_degree":4})
        output_form_4 = pde.pnorm_stress(state_function,input_function_1,E,nu,
                                dx_reduced,m=m,rho=rho,alpha=None,regularization=False)
        output_name_5 = 'von_Mises_stress'
        output_form_5 = pde.von_Mises_stress(state_function,input_function_1,E,nu,surface='Top')
        fea.add_input(input_name_1, input_function_1, init_val=0.001, record=True)
        fea.add_input(input_name_2, input_function_2, record=True)
        fea.add_state(name=state_name,
                        function=state_function,
                        residual_form=residual_form,
                        arguments=[input_name_1, input_name_2])
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
                        record=True)
        force_reshaping_model = ForceReshapingModel(pde=pde,
                                    input_name=shell_name+'_forces',
                                    output_name=input_name_2)
        solid_model = StateModel(fea=fea,
                                debug_mode=False,
                                state_name=state_name,
                                arg_name_list=fea.states_dict[state_name]['arguments'])
        compliance_model = OutputModel(fea=fea,
                                    output_name=output_name_1,
                                    arg_name_list=fea.outputs_dict[output_name_1]['arguments'])
        mass_model = OutputModel(fea=fea,
                                    output_name=output_name_2,
                                    arg_name_list=fea.outputs_dict[output_name_2]['arguments'])
        elastic_energy_model = OutputModel(fea=fea,
                                    output_name=output_name_3,
                                    arg_name_list=fea.outputs_dict[output_name_3]['arguments'])
        pnorm_stress_model = OutputModel(fea=fea,
                                    output_name=output_name_4,
                                    arg_name_list=fea.outputs_dict[output_name_4]['arguments'])
        von_Mises_stress_model = OutputFieldModel(fea=fea,
                                    output_name=output_name_5,
                                    arg_name_list=fea.outputs_field_dict[output_name_5]['arguments'])

        disp_extraction_model = DisplacementExtractionModel(pde=pde,
                                    input_name=state_name,
                                    output_name=shell_name+'_displacement')
        aggregated_stress_model = AggregatedStressModel(m=m, rho=rho,
                                    input_name=output_name_4,
                                    output_name=shell_name+'_aggregated_stress')

        self.add(force_reshaping_model, name='force_reshaping_model')
        self.add(solid_model, name='solid_model')
        self.add(disp_extraction_model, name='disp_extraction_model')
        self.add(compliance_model, name='compliance_model')
        self.add(von_Mises_stress_model, name='von_Mises_stress_model')
        self.add(mass_model, name='mass_model')
        self.add(elastic_energy_model, name='elastic_energy_model')
        self.add(pnorm_stress_model, name='von_mises_stress_model')
        self.add(aggregated_stress_model, name='aggregated_stress_model')


        dim = len(state_function.x.array)
        F_ssr = self.create_output('F_extended', shape = (2*dim,))
        # Need to expose the output variable F_solid
        Fi = self.register_output('Fi', csdl.reshape(F_solid, dim))

        for i in range(dim):
            F_ssr[i] = Fi[i]
            F_ssr[i+dim] = 0

        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)
        mass_matrix = self.register_output('mass_matrix', csdl.matmat(csdl.matmat(mask, sum_m), mask) + mask_eye)

        # compute inverse mass matrix
        mass_matrix_inverse = self.create_output('mass_matrix_inverse', shape=mass_matrix.shape)
        for i in range(mass_matrix.shape[0]):
            mass_matrix_inverse[i,i] = 1/mass_matrix[i,i]

        A_ssr = self.create_output('A_ssr', shape = (2*dim,2*dim))
        for i in range(dim):
            for j in range(dim):
                A_ssr[i,j] = 0
                A_ssr[i+dim,j+dim] = 0
                if i == j:
                    A_ssr[i+dim,j] = 1
                else:
                    A_ssr[i+dim,j] = 0
        A_ssr[0:dim,dim:2*dim] = -csdl.matmat(K*mass_matrix_inverse)

        delta = self.create_output('delta', shape = (2*dim,))
        delta[0:dim] = self.declare_variable('velocities', shape=(dim,), val=0)
        delta[dim:2*dim] = self.declare_variable('displacements', shape=(dim,), val=0)

        residual = self.register_output('residual', F_ssr + csdl.matvec(A_ssr, delta))


