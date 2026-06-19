"""
Unit tests for FEA input/state/output registration logic in AbstractFEA.
These tests use mock objects — no dolfinx installation required.
"""
import pytest
import sys
import types


def _make_abstract_fea(mock_mesh):
    """Import AbstractFEA with dolfinx stubbed out."""
    # Build minimal stubs so the module-level imports in fea_dolfinx don't fail
    for mod in ["dolfinx", "dolfinx.io", "dolfinx.fem", "dolfinx.fem.petsc",
                "dolfinx.mesh", "dolfinx.nls", "dolfinx.nls.petsc",
                "dolfinx.cpp", "dolfinx.cpp.mesh", "dolfinx.la",
                "ufl", "petsc4py", "petsc4py.PETSc",
                "mpi4py", "mpi4py.MPI", "matplotlib", "matplotlib.pyplot",
                "scipy", "scipy.spatial", "scipy.sparse"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Patch the symbols actually used at import time
    import ufl as _ufl
    for sym in ["Identity", "dot", "derivative", "TestFunction", "TrialFunction",
                "inner", "ds", "dS", "dx", "grad", "inv", "as_vector", "sqrt",
                "conditional", "lt", "det", "Measure", "exp", "tr", "CellDiameter"]:
        setattr(_ufl, sym, lambda *a, **kw: None)

    import dolfinx.fem as _fem
    for sym in ["form", "assemble_scalar", "Function", "FunctionSpace",
                "dirichletbc", "locate_dofs_geometrical", "Constant"]:
        setattr(_fem, sym, lambda *a, **kw: None)

    import dolfinx.mesh as _mesh
    for sym in ["create_unit_square", "create_rectangle", "create_interval",
                "locate_entities_boundary", "locate_entities", "meshtags"]:
        setattr(_mesh, sym, lambda *a, **kw: None)

    import mpi4py.MPI as _mpi
    _mpi.COMM_WORLD = object()

    import scipy.spatial as _spatial
    _spatial.KDTree = object

    # Now import the actual module
    import importlib
    import femo.fea.fea_dolfinx as _mod
    importlib.reload(_mod)
    return _mod.AbstractFEA(mock_mesh)


class MockFunction:
    def __init__(self, name="f"):
        self.name = name
    def rename(self, label, _):
        self.name = label


def test_add_input_stores_entry(mock_mesh):
    fea = _make_abstract_fea(mock_mesh)
    f = MockFunction("rho")
    fea.add_input("rho", f)
    assert "rho" in fea.inputs_dict
    assert fea.inputs_dict["rho"]["function"] is f


def test_add_input_duplicate_raises(mock_mesh):
    fea = _make_abstract_fea(mock_mesh)
    f = MockFunction("rho")
    fea.add_input("rho", f)
    with pytest.raises(ValueError, match="already been used"):
        fea.add_input("rho", MockFunction("rho2"))


def test_add_state_stores_entry(mock_mesh):
    fea = _make_abstract_fea(mock_mesh)
    u = MockFunction("u")
    fea.add_state("u", u, residual_form=None)
    assert "u" in fea.states_dict


def test_add_output_stores_entry(mock_mesh):
    fea = _make_abstract_fea(mock_mesh)
    fea.add_output("J", form=None)
    assert "J" in fea.outputs_dict


def test_add_strong_bc_appends(mock_mesh):
    fea = _make_abstract_fea(mock_mesh)
    bc = object()
    fea.add_strong_bc(bc)
    assert bc in fea.bcs_list
