import numpy as np
import types


def test_add_field_output_sets_shape_and_metadata(fresh_import):
    fea_mod = fresh_import("femo.fea.fea_dolfinx")
    fea = fea_mod.FEA(mesh=object())

    class DummyFunc:
        def __init__(self):
            self.vector = types.SimpleNamespace(getArray=lambda: np.zeros(6))

    fea_mod.FunctionSpace = lambda mesh, desc: object()
    fea_mod.Function = lambda V: DummyFunc()
    fea_mod.getFuncArray = lambda f: np.zeros(6)

    fea.add_field_output("disp", form="expr", arguments=["u"], record=True)
    out = fea.outputs_field_dict["disp"]
    assert out["shape"] == 6
    assert out["arguments"] == ["u"]
    assert out["record"] is True
