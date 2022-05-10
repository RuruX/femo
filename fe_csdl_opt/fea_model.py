import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
from fea_dolfinx import *
from state_model import StateModel
from output_model import OutputModel

import argparse


class FEAModel(Model):
    def initialize(self):
        self.parameters.declare('fea')
        self.options.declare(
            'linear_solver_',
            default='petsc_cg_ilu',
            values=[
                'fenics_direct', 'scipy_splu', 'fenics_krylov',
                'petsc_gmres_ilu', 'scipy_cg', 'petsc_cg_ilu'
            ],
        )
        self.options.declare(
            'problem_type',
            default='nonlinear_problem',
            values=[
                'linear_problem', 'nonlinear_problem',
                'nonlinear_problem_load_stepping'
            ],
        )
        self.options.declare(
            'visualization',
            default='False',
            values=['True', 'False'],
        )

    def define(self):


        self.fea = fea = self.parameters['fea']
        linear_solver_ = self.options['linear_solver_']
        problem_type = self.options['problem_type']
        visualization = self.options['visualization']

        for input_name in fea.inputs_dict:
            input = self.create_input("{}".format(input_name),
                                        shape=fea.get_total_dof(input_name),
                                        val=fea.get_initial_guess(input_name))
        for state_name in fea.states_dict:
            arg_name_list = self.fea.states_dict[state_name][arguments]
            self.add(StateModel(fea=self.fea,
                                debug_mode=False,
                                state_name=state_name,
                                arg_name_list=arg_name_list),
                                name='{}_state_model'.format(state_name),
                                promotes=['*'])

        for output_name in fea.outputs_dict:
            arg_name_list = self.fea.outputs_dict[output_name][arguments]
            self.add(OutputModel(fea=self.fea,
                                output_name=output_name,
                                arg_name_list=arg_name_list),
                                name='{}_output_model'.format(output_name),
                                promotes=['*'])
        #
        # self.add_design_variable('f')
        # self.add_objective('output_model.objective')
