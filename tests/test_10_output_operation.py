import numpy as np
import types


def test_output_operation_compute_and_derivatives(fresh_import):
    mod = fresh_import("femo.csdl_opt.output_model")

    update_calls = []
    mod.update = lambda fn, arr: update_calls.append((fn, tuple(arr.tolist())))
    mod.assemble = lambda form, dim=0: 7.0 if dim == 0 else np.array([7.0, 8.0])
    mod.computePartials = lambda form, fn: ("partial", form, fn)

    fn = types.SimpleNamespace()
    fea = types.SimpleNamespace(outputs_dict={
        "mass": {"form": "m_form", "shape": 1}
    })
    args_dict = {"rho": {"shape": 2, "function": fn}}

    op = mod.OutputOperation(fea=fea, args_dict=args_dict, output_name="mass")
    op.define()

    outputs = {"mass": np.zeros(1)}
    op.compute(inputs={"rho": np.array([1.0, 2.0])}, outputs=outputs)
    assert outputs["mass"].tolist() == 7.0
    assert len(update_calls) == 1

    derivatives = {}
    op.compute_derivatives(inputs={"rho": np.array([2.0, 3.0])}, derivatives=derivatives)
    assert ("mass", "rho") in derivatives
