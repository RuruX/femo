"""Unit tests for scalar and field output registration."""

import types
import numpy as np


def test_add_output_handles_scalar_and_field_shapes(fresh_import):
    """Verify add_output handles both scalar and field output contracts."""
    fea_mod = fresh_import("femo.fea.fea_dolfinx")
    fea = fea_mod.FEA(mesh=object())

    fea_mod.getFormArray = lambda form: np.zeros(5)
    fea_mod.derivative = lambda form, fn: ("d", form, fn)

    input_fn = types.SimpleNamespace(function_space=object(), x=types.SimpleNamespace(array=np.zeros(2)), rename=lambda *a, **k: None)
    state_fn = types.SimpleNamespace(function_space=object(), rename=lambda *a, **k: None)

    fea_mod.getFuncArray = lambda f: np.zeros(2)
    fea.add_input("rho", input_fn)
    fea.states_dict["u"] = {"function": state_fn}

    fea.add_output("compliance", "scalar", form="f1", arguments=["rho", "u"])
    scalar = fea.outputs_dict["compliance"]
    assert scalar["shape"] == 1
    assert len(scalar["partials"]) == 2

    fea.add_output("stress", "field", form="f2", arguments=["rho"])
    field = fea.outputs_dict["stress"]
    assert field["shape"] == 5
    assert len(field["partials"]) == 1
