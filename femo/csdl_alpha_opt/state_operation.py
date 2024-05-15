from femo.fea.fea_dolfinx import FEA, update, getFuncArray
import csdl_alpha as csdl


class StateOperation(csdl.CustomExplicitOperation):
    """
    input: input variable
    output: state
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
        debug_mode (bool, optional): If set to True, the debug mode is enabled. Defaults to False.
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
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")

        # declare output variables
        self.fea_state = self.fea.states_dict[self.state_name]
        state = self.create_output(
            self.state_name,
            shape=(self.fea_state["shape"],),
        )
        state.add_name(self.state_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.state_name, "*", dependent=False)

        return state

    def compute(self, input_vals, output_vals):
        """
        Compute the state operation

        parameters:
        ----------
        input_vals (dict): A dictionary of input variable values (numpy.ndarray).
        output_vals (dict): A dictionary of output variable values (numpy.ndarray).
        """

        if self.debug_mode is True:
            print("=" * 15 + str(self.state_name) + "=" * 15)
            print("CSDL: Running compute()...")
            print("=" * 40)

        self.solve_residual_equations(input_vals)
        output_vals[self.state_name] = getFuncArray(self.fea_state["function"])
        

    def solve_residual_equations(self, input_vals):
        """
        Solve the residual equations using FEMO solver.

        parameters:
        ----------
        input_vals (dict): A dictionary of input variable values (numpy.ndarray).

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
                arg["recorder"].write_function(arg["function"], self.fea.opt_iter)
        
        # solve the residual equation
        self.fea.solve(
            self.fea_state["residual_form"], self.fea_state["function"], self.fea.bc
        )
        
        # record the function values in XDMF files
        if self.fea.record:
            if self.fea_state["function"].function_space.num_sub_spaces > 1:
                u_mid, _ = self.fea_state["function"].split()
                self.fea_state["recorder"].write_function(u_mid, self.fea.opt_iter)

            else:
                self.fea_state["recorder"].write_function(
                    self.fea_state["function"], self.fea.opt_iter
                )