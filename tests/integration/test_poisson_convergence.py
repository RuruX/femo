"""
Convergence test: verifies the Poisson solver achieves the expected
O(h²) L2 error rate for a manufactured solution.

This is the canonical correctness check for a finite element code —
it cannot be replicated with mocks.
"""
import pytest
import numpy as np

dolfinx = pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import (FunctionSpace, Function, dirichletbc,
                          locate_dofs_geometrical, form, assemble_scalar,
                          Constant, Expression)
from dolfinx.cpp.mesh import CellType
import ufl
from ufl import inner, grad, dx, sin, pi, SpatialCoordinate

from femo.fea.fea_dolfinx import FEA


def _solve_poisson(n):
    """
    Solve -∇²u = 2π²·sin(πx)·sin(πy) on [0,1]², u=0 on ∂Ω.
    Exact solution: u* = sin(πx)·sin(πy).
    Returns (mesh_size h, L2 error).
    """
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    x = SpatialCoordinate(mesh)

    u = Function(V)
    v = ufl.TestFunction(V)

    f = 2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    res = inner(grad(u), grad(v)) * dx - inner(f, v) * dx

    def boundary(coords):
        return (np.isclose(coords[0], 0.0) | np.isclose(coords[0], 1.0) |
                np.isclose(coords[1], 0.0) | np.isclose(coords[1], 1.0))

    dofs = locate_dofs_geometrical(V, boundary)
    ubc = Function(V)
    ubc.x.array[:] = 0.0

    fea = FEA(mesh)
    fea.REPORT = False
    fea.add_strong_bc(ubc, [dofs])
    fea.solve(res, u, fea.bc)

    # L2 error against exact solution
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    error = form((u - u_exact) ** 2 * dx)
    L2 = np.sqrt(assemble_scalar(error))
    h = 1.0 / n
    return h, L2


@pytest.mark.parametrize("n", [4, 8, 16])
def test_poisson_converges(n):
    """Each refinement should produce a non-trivial, bounded solution."""
    h, err = _solve_poisson(n)
    assert err > 0, "zero error suggests solve did not run"
    assert err < 0.1, f"error {err:.4e} is too large for n={n}"


def test_poisson_convergence_rate():
    """
    Check that doubling the mesh halves the error with roughly O(h²) rate.
    Lagrange P1 on a uniform mesh → rate ≈ 2.
    """
    _, e1 = _solve_poisson(8)
    _, e2 = _solve_poisson(16)
    rate = np.log(e1 / e2) / np.log(2.0)
    assert rate > 1.8, f"convergence rate {rate:.2f} is below expected O(h²)"
