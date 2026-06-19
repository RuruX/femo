"""
Integration tests for FEA using real dolfinx/ufl/petsc4py.

These tests are skipped automatically if dolfinx is not installed.
Run them inside the official container:

    docker run --rm -v $(pwd):/repo -w /repo dolfinx/dolfinx:v0.9.0 \
        pytest tests/integration/ -v
"""
import pytest
import numpy as np

dolfinx = pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx.mesh import create_unit_square, create_unit_cube
from dolfinx.fem import (FunctionSpace, Function, dirichletbc,
                          locate_dofs_geometrical, form, assemble_scalar,
                          Constant)
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from dolfinx.cpp.mesh import CellType
import ufl
from ufl import inner, grad, dx, TestFunction, TrialFunction

from femo.fea.fea_dolfinx import FEA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def square_mesh():
    return create_unit_square(MPI.COMM_WORLD, 8, 8)


@pytest.fixture(scope="module")
def scalar_V(square_mesh):
    return FunctionSpace(square_mesh, ("Lagrange", 1))


# ---------------------------------------------------------------------------
# FEA registration
# ---------------------------------------------------------------------------

def test_add_input(square_mesh, scalar_V):
    fea = FEA(square_mesh)
    rho = Function(scalar_V)
    fea.add_input("rho", rho, init_val=1.0)

    assert "rho" in fea.inputs_dict
    assert np.allclose(fea.inputs_dict["rho"]["function"].x.array, 1.0)


def test_add_input_duplicate_raises(square_mesh, scalar_V):
    fea = FEA(square_mesh)
    fea.add_input("rho", Function(scalar_V), init_val=0.5)
    with pytest.raises(ValueError, match="already been used"):
        fea.add_input("rho", Function(scalar_V), init_val=0.5)


def test_add_state(square_mesh, scalar_V):
    fea = FEA(square_mesh)
    u = Function(scalar_V)
    v = TestFunction(scalar_V)
    res = inner(grad(u), grad(v)) * dx
    fea.add_state("u", u, res, arguments=[])
    assert "u" in fea.states_dict


def test_add_scalar_output(square_mesh, scalar_V):
    fea = FEA(square_mesh)
    u = Function(scalar_V)
    u.x.array[:] = 1.0
    fea.add_input("rho", u, init_val=1.0)
    output_form = u * dx
    fea.add_output("vol", type="scalar", form=output_form, arguments=["rho"])
    assert "vol" in fea.outputs_dict


def test_add_strong_bc(square_mesh, scalar_V):
    fea = FEA(square_mesh)
    ubc = Function(scalar_V)
    ubc.x.array[:] = 0.0

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    dofs = locate_dofs_geometrical(scalar_V, left_boundary)
    fea.add_strong_bc(ubc, [dofs])
    assert len(fea.bc) == 1


# ---------------------------------------------------------------------------
# Poisson solve — verifies the full FEA pipeline end-to-end
# ---------------------------------------------------------------------------

def test_poisson_solve(square_mesh, scalar_V):
    """
    Solve -∇²u = f on the unit square with u=0 on the boundary.
    Checks that the assembled system and Newton solve produce a non-trivial,
    physically reasonable solution.
    """
    fea = FEA(square_mesh)
    fea.PDE_SOLVER = "Newton"
    fea.REPORT = False

    u = Function(scalar_V)
    v = TestFunction(scalar_V)
    f_val = Constant(square_mesh, 1.0)

    # Nonlinear residual: R(u;v) = ∫ ∇u·∇v dx − ∫ f·v dx
    res = inner(grad(u), grad(v)) * dx - inner(f_val, v) * dx

    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    dofs = locate_dofs_geometrical(scalar_V, boundary)
    ubc = Function(scalar_V)
    ubc.x.array[:] = 0.0
    fea.add_strong_bc(ubc, [dofs])

    fea.add_state("u", u, res, arguments=[])
    fea.solve(res, u, fea.bc)

    u_vals = u.x.array
    # Solution should be positive and peak near the centre (~0.073 for unit square)
    assert np.max(u_vals) > 0.05
    assert np.max(u_vals) < 0.15
    # Boundary dofs should be (approximately) zero
    assert np.allclose(u_vals[dofs], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Utility functions from utils_dolfinx
# ---------------------------------------------------------------------------

def test_getFuncArray(scalar_V):
    from femo.fea.utils_dolfinx import getFuncArray
    f = Function(scalar_V)
    f.x.array[:] = 3.14
    arr = getFuncArray(f)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, 3.14)


def test_setFuncArray(scalar_V):
    from femo.fea.utils_dolfinx import getFuncArray, setFuncArray
    f = Function(scalar_V)
    new_vals = np.ones(len(f.x.array)) * 2.71
    setFuncArray(f, new_vals)
    assert np.allclose(getFuncArray(f), 2.71)


def test_create_unit_square_mesh():
    """Smoke test: mesh creation and basic topology."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    assert mesh.topology.dim == 2


def test_assemble_mass_matrix(square_mesh, scalar_V):
    """Verify that a mass matrix assembles without error and is non-zero."""
    u = TrialFunction(scalar_V)
    v = TestFunction(scalar_V)
    a = form(inner(u, v) * dx)
    A = assemble_matrix(a)
    A.assemble()
    # Frobenius norm should be positive
    assert A.norm() > 0
