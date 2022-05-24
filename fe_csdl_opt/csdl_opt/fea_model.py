from csdl import Model
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel

class FEAModel(Model):
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = fea = self.parameters['fea']

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

