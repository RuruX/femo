"""
Unit tests for FEA input/state/output registration logic in AbstractFEA.
These tests use mock objects — no dolfinx installation required.
"""
import pytest
import sys
import types


def _setup_stubs():
    """
    Populate sys.modules with empty stub modules and set every attribute
    accessed via 'from X import Y' in fea_dolfinx.py and utils_dolfinx.py.
    Safe to call multiple times — only creates stubs for modules not yet present.
    """
    _noop = lambda *a, **kw: None

    stub_names = [
        "dolfinx", "dolfinx.io", "dolfinx.fem", "dolfinx.fem.petsc",
        "dolfinx.mesh", "dolfinx.nls", "dolfinx.nls.petsc",
        "dolfinx.cpp", "dolfinx.cpp.mesh", "dolfinx.la",
        "ufl", "petsc4py", "petsc4py.PETSc",
        "mpi4py", "mpi4py.MPI", "matplotlib", "matplotlib.pyplot",
        "scipy", "scipy.spatial", "scipy.sparse",
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Wire parent.child references so `from parent import child` works
    for name in stub_names:
        parts = name.split(".")
        if len(parts) > 1:
            parent = sys.modules[".".join(parts[:-1])]
            if not hasattr(parent, parts[-1]):
                setattr(parent, parts[-1], sys.modules[name])

    # dolfinx.io
    sys.modules["dolfinx.io"].XDMFFile = _noop

    # ufl  (fea_dolfinx.py and utils_dolfinx.py imports combined)
    for sym in ["Identity", "dot", "derivative", "TestFunction", "TrialFunction",
                "inner", "ds", "dS", "dx", "grad", "inv", "as_vector", "sqrt",
                "conditional", "lt", "det", "Measure", "exp", "tr", "CellDiameter",
                "SpatialCoordinate", "FacetNormal", "div"]:
        setattr(sys.modules["ufl"], sym, _noop)

    # dolfinx.fem
    for sym in ["form", "assemble_scalar", "Function", "FunctionSpace",
                "dirichletbc", "locate_dofs_geometrical", "locate_dofs_topological",
                "Constant", "VectorFunctionSpace", "set_bc"]:
        setattr(sys.modules["dolfinx.fem"], sym, _noop)

    # dolfinx.fem.petsc
    for sym in ["assemble_vector", "assemble_matrix", "NonlinearProblem",
                "apply_lifting", "set_bc", "create_matrix", "_assemble_matrix_mat"]:
        setattr(sys.modules["dolfinx.fem.petsc"], sym, _noop)

    # dolfinx.mesh
    for sym in ["create_unit_square", "create_rectangle", "create_interval",
                "locate_entities_boundary", "locate_entities", "meshtags"]:
        setattr(sys.modules["dolfinx.mesh"], sym, _noop)

    # dolfinx.nls.petsc
    sys.modules["dolfinx.nls.petsc"].NewtonSolver = _noop

    # dolfinx.cpp.mesh
    sys.modules["dolfinx.cpp.mesh"].CellType = object()

    # mpi4py.MPI
    sys.modules["mpi4py.MPI"].COMM_WORLD = object()

    # scipy.spatial / scipy.sparse
    sys.modules["scipy.spatial"].KDTree = object
    sys.modules["scipy.sparse"].csr_matrix = _noop


def _make_abstract_fea(mock_mesh):
    """Return an AbstractFEA instance with all heavy deps stubbed out."""
    _setup_stubs()
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
