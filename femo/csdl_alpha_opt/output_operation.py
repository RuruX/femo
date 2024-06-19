from femo.fea.fea_dolfinx import FEA, update, assemble, computePartials, getFuncArray
import csdl_alpha as csdl
import numpy as np


class OutputOperation(csdl.CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """

    def __init__(self, fea, args_name_list, output_name):
        super().__init__()

        # define any checks for the parameters
        csdl.check_parameter(fea, "fea", types=FEA)
        csdl.check_parameter(args_name_list, "args_name_list", types=list)
        csdl.check_parameter(output_name, "output_name", types=str)

        args_dict = dict()
        for arg_name in args_name_list:
            if arg_name in fea.inputs_dict:
                args_dict[arg_name] = fea.inputs_dict[arg_name]
            elif arg_name in fea.states_dict:
                args_dict[arg_name] = fea.states_dict[arg_name]

        # assign parameters to the class
        self.fea = fea
        self.args_dict = args_dict
        self.output_name = output_name
        self.fea_output = fea.outputs_dict[output_name]
        self.output_dim = 0 # for scalar outputs
        
    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        for arg_name in self.args_dict:
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")

        # declare output variables
        output = self.create_output(self.output_name, (1,))
        output.add_name(self.output_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.output_name, '*', dependent=True)

        return output

    def compute(self, input_vals, output_vals):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg["function"], input_vals[arg_name])

        output_vals[self.output_name] = assemble(self.fea_output["form"])

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg["function"], input_vals[arg_name])

        for arg_name in input_vals:
            derivatives[self.output_name, arg_name] = assemble(
                computePartials(
                    self.fea_output["form"], self.args_dict[arg_name]["function"]
                ),
                dim=self.output_dim + 1,
            )


class OutputFieldOperation(csdl.CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """

    def __init__(self, fea, args_name_list, output_name):
        super().__init__()

        # define any checks for the parameters
        csdl.check_parameter(fea, "fea", types=FEA)
        csdl.check_parameter(args_name_list, "args_name_list", types=list)
        csdl.check_parameter(output_name, "output_name", types=str)

        args_dict = dict()
        for arg_name in args_name_list:
            if arg_name in fea.inputs_dict:
                args_dict[arg_name] = fea.inputs_dict[arg_name]
            elif arg_name in fea.states_dict:
                args_dict[arg_name] = fea.states_dict[arg_name]

        # assign parameters to the class
        self.fea = fea
        self.args_dict = args_dict
        self.output_name = output_name
        self.fea_output = fea.outputs_field_dict[output_name]
        self.output_dim = 1 # for field outputs

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        for arg_name in self.args_dict:
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")

        # declare output variables
        output = self.create_output(self.output_name, (self.fea_output['shape'],))
        output.add_name(self.output_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.output_name, '*', dependent=True)
        return output

    def compute(self, input_vals, output_vals):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg['function'], input_vals[arg_name])

        self.fea.projectFieldOutput(self.fea_output['form'],self.fea_output['function'])
        output_vals[self.output_name] = getFuncArray(self.fea_output['function'])

        # record the function values in XDMF files
        if self.fea_output['record']:
            self.fea_output['recorder'].write_function(
                self.fea_output['function'], self.fea.opt_iter
            )


