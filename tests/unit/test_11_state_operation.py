"""Unit tests for StateOperation control-flow paths."""

import numpy as np
import types


def test_state_operation_core_paths(fresh_import):
    """Verify residual, solve, and inverse-jacobian paths in StateOperation."""
    mod = fresh_import("femo.csdl_opt.state_model")

    update_calls = []
    mod.update = lambda fn, arr: update_calls.append((fn, tuple(arr.tolist())))
    mod.assembleVector = lambda form: np.array([5.0])
    mod.getFuncArray = lambda fn: np.array([9.0])

    class FakeFEA:
        def __init__(self):
            self.states_dict = {
                "u": {
                    "function": object(),
                    "residual_form": "res",
                    "shape": 1,
                    "d_residual": object(),
                    "d_state": object(),
                    "record": False,
                    "arguments": ["rho"],
                }
            }
            self.inputs_dict = {}
            self.bc = []
            self.linear_problem = False
            self.record = False
            self.opt_iter = 0
            self.solve_calls = []

        def solve(self, res, fn, bcs):
            self.solve_calls.append((res, fn, bcs))

        def solveLinearFwd(self, du, A, dR, rhs, ksp):
            return np.array([11.0])

        def solveLinearBwd(self, dR, A, du, rhs, ksp):
            return np.array([12.0])

    fea = FakeFEA()
    arg_fn = object()
    args_dict = {"rho": {"shape": 1, "function": arg_fn, "record": False}}

    op = mod.StateOperation(debug_mode=False, fea=fea, args_dict=args_dict, state_name="u")
    op.define()

    residuals = {"u": np.zeros(1)}
    op.evaluate_residuals(inputs={"rho": np.array([1.0])}, outputs={"u": np.array([2.0])}, residuals=residuals)
    assert residuals["u"].tolist() == [5.0]

    outputs = {"u": np.array([0.0])}
    op.solve_residual_equations(inputs={"rho": np.array([3.0])}, outputs=outputs)
    assert fea.opt_iter == 1
    assert outputs["u"].tolist() == [9.0]
    assert len(fea.solve_calls) == 1

    op.du = object()
    op.dR = object()
    op.A = object()
    op.ksp = None

    d_outputs = {"u": np.array([0.0])}
    d_residuals = {"u": np.array([1.0])}
    op.apply_inverse_jacobian(d_outputs=d_outputs, d_residuals=d_residuals, mode="fwd")
    assert d_outputs["u"].tolist() == [11.0]

    d_outputs = {"u": np.array([2.0])}
    d_residuals = {"u": np.array([0.0])}
    op.apply_inverse_jacobian(d_outputs=d_outputs, d_residuals=d_residuals, mode="rev")
    assert d_residuals["u"].tolist() == [12.0]
