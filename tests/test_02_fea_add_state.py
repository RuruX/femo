"""Unit tests for FEA state registration."""

import types
import numpy as np


def test_add_state_registers_expected_keys(fresh_import):
    """Verify add_state records the expected dictionary fields."""
    fea_mod = fresh_import("femo.fea.fea_dolfinx")
    fea = fea_mod.FEA(mesh=object())

    class DummyFunction:
        def __init__(self):
            self.function_space = object()

        def rename(self, *args, **kwargs):
            return None

    fea_mod.getFuncArray = lambda f: np.zeros(4)
    fea_mod.Function = lambda fs: types.SimpleNamespace(function_space=fs)

    residual = object()
    fn = DummyFunction()
    fea.add_state("u", fn, residual, arguments=["rho"], dR_du="dRdu", dR_df_list=["dRdrho"])

    state = fea.states_dict["u"]
    assert state["function"] is fn
    assert state["residual_form"] is residual
    assert state["shape"] == 4
    assert state["dR_du"] == "dRdu"
    assert state["dR_df_list"] == ["dRdrho"]
    assert state["arguments"] == ["rho"]
