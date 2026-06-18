"""Unit tests for FEAModel CSDL wiring."""

import types


def test_fea_model_define_adds_state_and_output_models(fresh_import):
    """Verify FEAModel.define adds state, output, and field output submodels."""
    fea_model_mod = fresh_import("femo.csdl_opt.fea_model")

    made = []
    fea_model_mod.StateModel = lambda **kwargs: ("state", kwargs)
    fea_model_mod.OutputModel = lambda **kwargs: ("output", kwargs)
    fea_model_mod.OutputFieldModel = lambda **kwargs: ("field", kwargs)

    fake_fea = types.SimpleNamespace(
        states_dict={"u": {"arguments": ["rho"]}},
        outputs_dict={"mass": {"arguments": ["rho", "u"]}},
        outputs_field_dict={"disp": {"arguments": ["u"]}},
    )

    model = fea_model_mod.FEAModel()
    model.parameters["fea"] = [fake_fea]
    model.define()

    names = [name for name, _ in model._added]
    assert "u_state_model" in names
    assert "mass_output_model" in names
    assert "disp_output_model" in names
