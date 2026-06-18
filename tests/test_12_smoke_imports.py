"""Unit smoke tests for importing core FEMO modules."""

def test_smoke_imports_core_modules(fresh_import):
    """Verify core package modules import successfully."""
    assert fresh_import("femo.fea.fea_dolfinx") is not None
    assert fresh_import("femo.fea.utils_dolfinx") is not None
    assert fresh_import("femo.csdl_opt.fea_model") is not None
    assert fresh_import("femo.csdl_opt.output_model") is not None
    assert fresh_import("femo.csdl_opt.state_model") is not None
