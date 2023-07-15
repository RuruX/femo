from femo.fea.fea_dolfinx import *
from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np


class OutputModel(Model):

    def initialize(self):
        self.parameters.declare('fea', types=FEA)
        self.parameters.declare('output_name', types=str)
        self.parameters.declare('arg_name_list', types=list)

    def define(self):
        self.fea = self.parameters['fea']
        arg_name_list = self.parameters['arg_name_list']
        output_name = self.parameters['output_name']

        args_dict = dict()
        args_list = []
        for arg_name in arg_name_list:
            if arg_name in self.fea.inputs_dict:
                args_dict[arg_name] = self.fea.inputs_dict[arg_name]
            elif arg_name in self.fea.states_dict:
                args_dict[arg_name] = self.fea.states_dict[arg_name]
            arg = self.declare_variable(arg_name,
                                        shape=(args_dict[arg_name]['shape'],),
                                        val=1.0)
            args_list.append(arg)

        e = OutputOperation(fea=self.fea,
                            args_dict=args_dict,
                            output_name=output_name,
                            )
        output = csdl.custom(*args_list, op=e)
        self.register_output(output_name, output)

class OutputOperation(CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """
    def initialize(self):
        self.parameters.declare('fea')
        self.parameters.declare('args_dict')
        self.parameters.declare('output_name')

    def define(self):
        self.fea = self.parameters['fea']
        self.output_name = output_name = self.parameters['output_name']
        self.args_dict = args_dict = self.parameters['args_dict']
        for arg_name in args_dict:
            arg = args_dict[arg_name]
            self.add_input(arg_name,
                            shape=(arg['shape'],),)
        self.output = self.fea.outputs_dict[output_name]
        self.output_size = self.output['shape']
        # for field output
        self.output_dim = 1
        # for scalar output
        if self.output_size == 1:
            self.output_dim = 0
        self.add_output(output_name,
                        shape=(self.output_size,))
        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        outputs[self.output_name] = np.array(assemble(self.output['form'],
                                        dim=self.output_dim))

    def compute_derivatives(self, inputs, derivatives):
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        for arg_name in self.args_dict:
            derivatives[self.output_name,arg_name] = assemble(
                                    computePartials(
                                        self.output['form'],
                                        self.args_dict[arg_name]['function']),
                                    dim=self.output_dim+1)



class OutputFieldModel(Model):

    def initialize(self):
        self.parameters.declare('fea', types=FEA)
        self.parameters.declare('output_name', types=str)
        self.parameters.declare('arg_name_list', types=list)

    def define(self):
        self.fea = self.parameters['fea']
        arg_name_list = self.parameters['arg_name_list']
        output_name = self.parameters['output_name']

        args_dict = dict()
        args_list = []
        for arg_name in arg_name_list:
            if arg_name in self.fea.inputs_dict:
                args_dict[arg_name] = self.fea.inputs_dict[arg_name]
            elif arg_name in self.fea.states_dict:
                args_dict[arg_name] = self.fea.states_dict[arg_name]
            arg = self.declare_variable(arg_name,
                                        shape=(args_dict[arg_name]['shape'],),
                                        val=1.0)
            args_list.append(arg)

        e = OutputFieldOperation(fea=self.fea,
                            args_dict=args_dict,
                            output_name=output_name,
                            )
        output = csdl.custom(*args_list, op=e)
        self.register_output(output_name, output)

class OutputFieldOperation(CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """
    def initialize(self):
        self.parameters.declare('fea')
        self.parameters.declare('args_dict')
        self.parameters.declare('output_name')

    def define(self):
        self.fea = self.parameters['fea']
        self.output_name = output_name = self.parameters['output_name']
        self.args_dict = args_dict = self.parameters['args_dict']
        for arg_name in args_dict:
            arg = args_dict[arg_name]
            self.add_input(arg_name,
                            shape=(arg['shape'],),)
        self.output = self.fea.outputs_field_dict[output_name]
        self.output_size = self.output['shape']
        # for field output
        self.output_dim = 1

        self.add_output(output_name,
                        shape=(self.output_size,))
        # self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        self.fea.projectFieldOutput(self.output['form'],self.output['func'])
        if self.output['record']:
            self.output['recorder'].write_function(self.output['func'],
                                                    self.fea.opt_iter)

        outputs[self.output_name] = getFuncArray(self.output['func'])
