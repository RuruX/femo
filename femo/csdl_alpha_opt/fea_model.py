# This file contains the class FEAModel, which is used to evaluate all of the FEA variables
from femo.csdl_alpha_opt.state_operation import StateOperation
from femo.csdl_alpha_opt.output_operation import OutputOperation, OutputFieldOperation
import csdl_alpha as csdl

class FEAModel():
    def __init__(self, fea, fea_name='fea'):
        self.parameters = {'fea': fea, 'fea_name': fea_name}

    def evaluate(self, inputs: csdl.VariableGroup, debug_mode=False):
        """
        Evaluate all of the FEA variables

        Parameters:
        ----------
        inputs (dict): A dictionary of input variables (csdl.Variable).
        debug_mode (bool, optional): If set to True, the debug mode is enabled. 
                                    Defaults to False.

        Returns:
        --------
        fea_variable_dict (dict): A dictionary of the FEA variables 
                                    including inputs, states, and outputs.
        """

        self.fea_list = fea_list = self.parameters['fea']
        fea_name = self.parameters['fea_name']

        # construct output of the model
        fea_variable_dict = inputs

        # with csdl.namespace(fea_name):
        # loop over the FEA list (there could be multiple FEA objects for coupled PDEs)
        for fea in fea_list:
            for state_name in fea.states_dict:
                args_name_list_state = fea.states_dict[state_name]['arguments']
                state_operation = StateOperation(fea=fea,
                                            state_name=state_name,
                                            args_name_list=args_name_list_state,
                                            debug_mode=debug_mode)

                state = state_operation.evaluate(fea_variable_dict)

                # add the state variable to the dictionary
                setattr(fea_variable_dict, state_name, state)

            for output_name in fea.outputs_dict:
                args_name_list_output = fea.outputs_dict[output_name]['arguments']
                output_operation = OutputOperation(fea=fea, 
                                            output_name=output_name,
                                            args_name_list=args_name_list_output)

                output = output_operation.evaluate(fea_variable_dict)

                # add the output variable to the dictionary
                setattr(fea_variable_dict, output_name, output)

            for output_name in fea.outputs_field_dict:
                args_name_list_output = fea.outputs_field_dict[output_name]['arguments']
                output_operation = OutputFieldOperation(fea=fea,
                                            output_name=output_name,
                                            args_name_list=args_name_list_output)
                output = output_operation.evaluate(fea_variable_dict)

                # add the output variable to the dictionary
                setattr(fea_variable_dict, output_name, output)

        return fea_variable_dict