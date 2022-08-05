
from fe_csdl_opt.fea.fea_dolfinx import *
from csdl import Model, CustomImplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator

class StateModel(Model):

    def initialize(self):
        self.parameters.declare('debug_mode', default=False)
        self.parameters.declare('fea', types=FEA)
        self.parameters.declare('state_name', types=str)
        self.parameters.declare('arg_name_list', types=list)

    def define(self):
        self.fea = self.parameters['fea']
        arg_name_list = self.parameters['arg_name_list']
        state_name = self.parameters['state_name']
        self.debug_mode = self.parameters['debug_mode']
        args_dict = dict()
        args_list = []
        for arg_name in arg_name_list:
            args_dict[arg_name] = self.fea.inputs_dict[arg_name]
            arg = self.declare_variable(arg_name,
                                        shape=(args_dict[arg_name]['shape'],),
                                        val=1.0)
            args_list.append(arg)

        e = StateOperation(fea=self.fea,
                            args_dict=args_dict,
                            state_name=state_name,
                            debug_mode=self.debug_mode)
        state = csdl.custom(*args_list, op=e)
        self.register_output(state_name, state)


class StateOperation(CustomImplicitOperation):
    """
    input: input variable
    output: state
    """
    def initialize(self):
        self.parameters.declare('debug_mode')
        self.parameters.declare('fea')
        self.parameters.declare('args_dict')
        self.parameters.declare('state_name')

    def define(self):
        self.debug_mode = self.parameters['debug_mode']
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running define()...")
            print("="*40)

        self.fea = self.parameters['fea']
        self.state_name = state_name = self.parameters['state_name']
        self.args_dict = args_dict = self.parameters['args_dict']
        for arg_name in args_dict:
            arg = args_dict[arg_name]
            self.add_input(arg_name,
                            shape=(arg['shape'],),)

        self.state = self.fea.states_dict[state_name]
        self.add_output(state_name,
                        shape=(self.state['shape'],),)
        self.declare_derivatives('*', '*')
        self.bcs = self.fea.bc

    def evaluate_residuals(self, inputs, outputs, residuals):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running evaluate_residuals()...")
            print("="*40)

        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])
        update(self.state['function'], outputs[self.state_name])

        residuals[self.state_name] = assembleVector(self.state['residual_form'])


    def solve_residual_equations(self, inputs, outputs):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running solve_residual_equations()...")
            print("="*40)
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])
        self.fea.solve(self.state['residual_form'],
                        self.state['function'],
                        self.bcs)

        outputs[self.state_name] = getFuncArray(self.state['function'])


    def compute_derivatives(self, inputs, outputs, derivatives):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running compute_derivatives()...")
            print("="*40)

        for arg_name in inputs:
            update(self.args_dict[arg_name]['function'], inputs[arg_name])
        update(self.state['function'], outputs[self.state_name])

        state = self.state
        args_dict = self.args_dict
        dR_du = state['dR_du']
        if dR_du == None:
            dR_du = computePartials(state['residual_form'],state['function'])
        self.dRdu = assembleMatrix(dR_du)


        dRdf_dict = dict()
        dR_df_list = state['dR_df_list']
        arg_list = state['arguments']
        for arg_ind in range(len(arg_list)):
            arg_name = arg_list[arg_ind]
            if dR_df_list == None:
                dRdf = assembleMatrix(computePartials(
                                    state['residual_form'],
                                    args_dict[arg_name]['function']))
            else:
                dRdf = dR_df_list[arg_ind]
            df = createFunction(args_dict[arg_name]['function'])
            dRdf_dict[arg_name] = dict(dRdf=dRdf, df=df)

        self.dRdf_dict = dRdf_dict
        self.A,_ = assembleSystem(dR_du,
                                state['residual_form'],
                                bcs=self.bcs)

        self.dR = self.state['d_residual']
        self.du = self.state['d_state']

    def compute_jacvec_product(self, inputs, outputs,
                                d_inputs, d_outputs, d_residuals, mode):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running compute_jacvec_product()...")
            print("="*40)

        ######################
        # Might be redundant #
        for arg_name in inputs:
            update(self.args_dict[arg_name]['function'], inputs[arg_name])
        update(self.state['function'], outputs[self.state_name])
        ######################

        state_name = self.state_name
        args_dict = self.args_dict
        if mode == 'fwd':
            if state_name in d_residuals:
                if state_name in d_outputs:
                    update(self.du, d_outputs[state_name])
                    d_residuals[state_name] += computeMatVecProductFwd(
                            self.dRdu, self.du)
                for arg_name in self.dRdf_dict:
                    if arg_name in d_inputs:
                        update(self.dRdf_dict[arg_name]['df'], d_inputs[arg_name])
                        dRdf = self.dRdf_dict[arg_name]['dRdf']
                        d_residuals[state_name] += computeMatVecProductFwd(
                                dRdf, self.dRdf_dict[arg_name]['df'])

        if mode == 'rev':
            if state_name in d_residuals:
                update(self.dR, d_residuals[state_name])
                if state_name in d_outputs:
                    d_outputs[state_name] += computeMatVecProductBwd(
                            self.dRdu, self.dR)
                for arg_name in self.dRdf_dict:
                    if arg_name in d_inputs:
                        dRdf = self.dRdf_dict[arg_name]['dRdf']
                        d_inputs[arg_name] += computeMatVecProductBwd(
                                dRdf, self.dR)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running apply_inverse_jacobian()...")
            print("="*40)
        state_name = self.state_name
        if mode == 'fwd':
            d_outputs[state_name] = self.fea.solveLinearFwd(
                                            self.du, self.A, self.dR, d_residuals[state_name])
        else:
            d_residuals[state_name] = self.fea.solveLinearBwd(
                                            self.dR, self.A, self.du, d_outputs[state_name])


