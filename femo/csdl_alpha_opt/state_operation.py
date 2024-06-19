from femo.fea.fea_dolfinx import (FEA, update, getFuncArray, computePartials, 
                                  assembleMatrix, computeMatVecProductFwd, 
                                  computeMatVecProductBwd, createFunction,
                                  assembleSystem, setUpKSP_MUMPS)
import csdl_alpha as csdl


class StateOperation(csdl.experimental.CustomImplicitOperation):
    """
    input: input variable
    output: state variable
    """

    def __init__(self, fea, args_name_list, state_name, debug_mode=False):
        super().__init__()
        """
        Initialize the StateOperation object.

        Parameters:
        ----------
        fea (FEA): An instance of the FEA class.
        args_name_list (list): A list of argument names.
        state_name (str): The name of the state.
        debug_mode (bool, optional): If set to True, the debug mode is enabled. 
                    Defaults to False.
        """
        # define any checks for the parameters
        csdl.check_parameter(fea, "fea", types=FEA)
        csdl.check_parameter(args_name_list, "args_name_list", types=list)
        csdl.check_parameter(state_name, "state_name", types=str)

        args_dict = dict()
        for arg_name in args_name_list:
            args_dict[arg_name] = fea.inputs_dict[arg_name]

        # assign parameters to the class
        self.fea = fea
        self.args_dict = args_dict
        self.state_name = state_name
        self.debug_mode = debug_mode

        self.fea_state = self.fea.states_dict[self.state_name]
        self.fea_dR = self.fea_state['d_residual']
        self.fea_du = self.fea_state['d_state']

        self.set_up_fea_derivatives()

    def evaluate(self, inputs: csdl.VariableGroup):
        """

        Evaluate the state operation

        parameters:
        ----------
        inputs (dict): A dictionary of input variables (csdl.Variable).

        returns:
        --------
        state (csdl.Variable): The state variable.
        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running evaluate()...")
            print("=" * 40)

        # assign method inputs to input dictionary
        for arg_name in self.args_dict:  
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(
                    f"Variable {arg_name} not found in the FEA model.")

        # declare output variables

        state = self.create_output(
            self.state_name,
            shape=(self.fea_state["shape"],),
        )
        state.add_name(self.state_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.state_name, '*', dependent=True)

        return state

    def solve_residual_equations(self, input_vals, output_vals):
        """
        Solve the residual equations using FEMO solver.

        parameters:
        ----------
        input_vals (dict): A dictionary of input variable values (numpy.ndarray).
        output_vals (dict): A dictionary of output variable values.
        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running solve_residual_equations()...")
            print("=" * 40)

        # update the input values in the FEA model
        self.fea.opt_iter += 1
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg["function"], input_vals[arg_name])
            if arg["record"]:
                arg["recorder"].write_function(arg["function"], 
                                               self.fea.opt_iter)
        
        # solve the residual equation
        self.fea.solve(
            self.fea_state["residual_form"], 
            self.fea_state["function"], 
            self.fea.bc
        )

        output_vals[self.state_name] = getFuncArray(self.fea_state["function"])

        # record the function values in XDMF files
        if self.fea.record:
            if self.fea_state["function"].function_space.num_sub_spaces > 1:
                u_mid, _ = self.fea_state["function"].split()
                self.fea_state["recorder"].write_function(u_mid, 
                                                        self.fea.opt_iter)

            else:
                self.fea_state["recorder"].write_function(
                    self.fea_state["function"], self.fea.opt_iter
                )

        # assemble the derivatives based on the updated state and inputs
        if self.fea.linear_problem is False or self.fea.opt_iter == 1:
            self.assemble_derivatives(input_vals, output_vals)
        

    def compute_jacvec_product(self, input_vals, output_vals, 
                               d_inputs, d_outputs, d_residuals, mode):
        """
        Compute the product of the Jacobian matrix and a vector.

        parameters:
        ----------
        input_vals (dict): A dictionary of input variable values.
        output_vals (dict): A dictionary of output variable values.
        d_inputs (dict): A dictionary of input variable deltas.
        d_outputs (dict): A dictionary of output variable deltas.
        d_residuals (dict): A dictionary of residual variable deltas.
        mode (str): The mode of the operation. It could be either 'fwd' or 'rev'.
        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running compute_jacvec_product()...")
            print("=" * 40)

        # [RX] not sure if the inputs will be updated from the last solve call
        # self.assemble_derivatives(input_vals, output_vals)

        state_name = self.state_name
        # for mode = fwd
        # d_inputs --> d_residuals
        if mode == 'fwd':
            if state_name in d_residuals:
                if state_name in d_outputs:
                    update(self.fea_du, d_outputs[state_name])
                    d_residuals[state_name] += computeMatVecProductFwd(
                            self.dRdu, self.fea_du)
                for arg_name in self.dR_df_dict:
                    if arg_name in d_inputs:
                        update(self.dR_df_dict[arg_name]['df'],
                                d_inputs[arg_name])
                        dRdf = self.dR_df_dict[arg_name]['dRdf']
                        d_residuals[state_name] += computeMatVecProductFwd(
                                dRdf, self.dR_df_dict[arg_name]['df'])
        # for mode = rev
        # d_residuals --> d_inputs
        elif mode == 'rev':
            if state_name in d_residuals:
                update(self.fea_dR, d_residuals[state_name])
                # if state_name in d_outputs:
                #     d_outputs[state_name] += computeMatVecProductBwd(
                #             self.dRdu, self.fea_dR)
                for arg_name in self.dR_df_dict:
                    if arg_name in d_inputs:
                        dRdf = self.dR_df_dict[arg_name]['dRdf']
                        d_inputs[arg_name] += computeMatVecProductBwd(
                                dRdf, self.fea_dR)
        else:
            raise ValueError("mode must be either 'fwd' or 'rev'.")
                                
    def apply_inverse_jacobian(self, input_vals, output_vals, 
                               d_outputs, d_residuals, mode):
        """
        Solve linear system. Invoked when solving coupled linear system; 
        i.e. when solving Newton system to update implicit state variables, 
        and when computing total derivatives

        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running apply_inverse_jacobian()...")
            print("=" * 40)

        state_name = self.state_name
        # for mode = fwd
        # d_residuals --> d_outputs
        if mode == 'fwd':
            d_outputs[state_name] = self.fea.solveLinearFwd(
                            self.fea_du, self.A, self.fea_dR,
                            d_residuals[state_name],
                            self.ksp)
        # for mode = rev:
        # d_outputs --> d_residuals
        elif mode == 'rev':
            d_residuals[state_name] = self.fea.solveLinearBwd(
                            self.fea_dR, self.A, self.fea_du,
                            d_outputs[state_name],
                            self.ksp)
            # apply the boundary conditions to the derivatives
            for bc in self.fea.bc:
                d_residuals[state_name][bc.dof_indices()[0]] = 0.0
        else:
            raise ValueError("mode must be either 'fwd' or 'rev'.")

    def set_up_fea_derivatives(self):
        """
        Set up the FEA derivatives.

        Returns:
        --------
        dR_df (ufl form): derivative form of the residual w.r.t. the inputs.
        dR_du (ufl form): derivative form of the residual w.r.t. the state.
        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running set_up_fea_derivatives()...")
            print("=" * 40)

        # compute the derivative of the residual w.r.t. the state
        dR_du = self.fea_state['dR_du']
        if dR_du is None:
            dR_du = computePartials(self.fea_state['residual_form'],
                                    self.fea_state['function'])
        self.fea_state['dR_du'] = dR_du

        # compute the derivative of the residual w.r.t. the inputs
        dR_df_dict = dict()
        dR_df_list = self.fea_state['dR_df_list']
        arg_list = self.fea_state['arguments']
        for arg_ind in range(len(arg_list)):
            arg_name = arg_list[arg_ind]
            if dR_df_list is None:
                dR_df = computePartials(
                                    self.fea_state['residual_form'],
                                    self.args_dict[arg_name]['function'])
            else:
                dR_df = dR_df_list[arg_ind]

            df = createFunction(self.args_dict[arg_name]['function'])
            dR_df_dict[arg_name] = dict(dR_df=dR_df, fea_df=df)

        self.dR_df_dict = dR_df_dict
    
    def assemble_derivatives(self, input_vals, output_vals):
        """
        Assemble the derivatives.

        returns:
        --------
        dRdf (csr_matrix): assembled derivative of the residual w.r.t. the inputs.
        dRdu (csr_matrix): assembled derivative of the residual w.r.t. the state.
        A (csr_matrix): assembled Jacobian matrix. Similar to dRdu but has bc.
        ksp (PETSc.KSP): PETSc KSP object.
        """
        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running assemble_derivatives()...")
            print("=" * 40)

        # update the input values in the FEA model
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg["function"], input_vals[arg_name])
        update(self.fea_state["function"], output_vals[self.state_name])

        dR_df_dict = self.dR_df_dict
        for arg_name in dR_df_dict:
            dR_df = dR_df_dict[arg_name]['dR_df']
            dRdf = assembleMatrix(dR_df)
            self.dR_df_dict[arg_name]['dRdf'] = dRdf

        # assemble the derivative of the residual w.r.t. the state and inputs
        self.dRdu = assembleMatrix(self.fea_state['dR_du'])

        # construct the KSP solver used in the adjoint
        self.A,_ = assembleSystem(self.fea_state['dR_du'],
                        self.fea_state['residual_form'],
                        bcs=self.fea.bc)

        self.ksp = setUpKSP_MUMPS(self.A)