"""Unit tests for FEA input registration behavior."""

import types
import numpy as np
import pytest


def test_add_input_rejects_duplicate_name(fresh_import):
    """Verify add_input stores metadata and rejects duplicate input names."""
    fea_mod = fresh_import("femo.fea.fea_dolfinx")
    fea = fea_mod.FEA(mesh=object())

    class DummyFunction:
        def __init__(self):
            self.function_space = object()
            self.x = types.SimpleNamespace(array=np.zeros(3))

        def rename(self, *args, **kwargs):
            return None

    fea_mod.getFuncArray = lambda f: np.zeros(3)

    fn = DummyFunction()
    fea.add_input("density", fn, init_val=2.5)
    assert "density" in fea.inputs_dict
    assert fea.inputs_dict["density"]["shape"] == 3

    with pytest.raises(ValueError):
        fea.add_input("density", DummyFunction())
