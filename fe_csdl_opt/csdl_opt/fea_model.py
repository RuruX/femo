from csdl import Model
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel

class FEAModel(Model):
    def initialize(self):
        self.parameters.declare('fea')
        self.parameters.declare(
            'linear_solver_',
            default='petsc_cg_ilu',
            values=[
                'fenics_direct', 'scipy_splu', 'fenics_krylov',
                'petsc_gmres_ilu', 'scipy_cg', 'petsc_cg_ilu'
            ],
        )
        self.parameters.declare(
            'problem_type',
            default='nonlinear_problem',
            values=[
                'linear_problem', 'nonlinear_problem',
                'nonlinear_problem_load_stepping'
            ],
        )
        self.parameters.declare(
            'visualization',
            default='False',
            values=['True', 'False'],
        )

    def define(self):


        self.fea = fea = self.parameters['fea']
        linear_solver_ = self.parameters['linear_solver_']
        problem_type = self.parameters['problem_type']
        visualization = self.parameters['visualization']

#        for input_name in fea.inputs_dict:
#            self.create_input("{}".format(input_name),
#                                        shape=fea.inputs_dict[input_name]['shape'],
#                                        val=1.0)

        for state_name in fea.states_dict:
            arg_name_list = fea.states_dict[state_name]['arguments']
            self.add(StateModel(fea=fea,
                                debug_mode=False,
                                state_name=state_name,
                                arg_name_list=arg_name_list),
                                name='{}_state_model'.format(state_name),
                                promotes=['*'])

        for output_name in fea.outputs_dict:
            arg_name_list = fea.outputs_dict[output_name]['arguments']
            self.add(OutputModel(fea=fea,
                                output_name=output_name,
                                arg_name_list=arg_name_list),
                                name='{}_output_model'.format(output_name),
                                promotes=['*'])

