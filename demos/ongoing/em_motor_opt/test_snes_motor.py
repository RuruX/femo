import dolfinx
from dolfinx import la
import ufl
from ufl import (Measure, TestFunction, TrialFunction,
                form, derivative, inner, grad, dx, as_vector)
from motor_pde import pdeRes
from fe_csdl_opt.fea.utils_dolfinx import import_mesh, gradx, project
import numpy as np
from dolfinx.fem import (Function, VectorFunctionSpace, dirichletbc,
                        FunctionSpace, locate_dofs_geometrical, form,)
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix,
                        apply_lifting, set_bc,
                        create_matrix, create_vector)
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from mpi4py import MPI

class NonlinearSNESProblem:

    def __init__(self, F, u, bcs,
                 J=None):
        self.L = form(F)

        # Create the Jacobian matrix, dF/du
        if J is None:
            V = u.function_space
            du = TrialFunction(V)
            J = derivative(F, u, du)

        self.a = form(J)
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, b):
        # Reset the residual vector
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)

        # Apply boundary condition
        apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()

def solveNonlinearSNES(F, w, bcs=[],
                    J_ = None,
                    abs_tol=1e-6,
                    rel_tol=1e-6,
                    max_it=30,
                    report=False):
    """
    https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/nls/test_newton.py#L182-L205
    """
    # Create nonlinear problem
    problem = NonlinearSNESProblem(F, w, bcs, J_)
    if report is True:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    W = w.function_space
    b = la.create_petsc_vector(W.dofmap.index_map, W.dofmap.index_map_bs)
    J = create_matrix(problem.a)
    # Create Newton solver and solve
    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts['snes_type'] = 'newtonls'
    opts['snes_linesearch_type'] = 'basic'
    opts['snes_monitor'] = None
    opts['snes_linesearch_monitor'] = None
    opts['snes_linesearch_damping'] = 0.8
    opts["error_on_nonconvergence"] = False
    opts['snes_solution_tolerance'] = 1.0e-16
    opts["snes_maximum_residual_evaluations"] = 2000
    snes.setTolerances(atol=abs_tol, rtol=rel_tol, max_it=max_it)
    snes.getKSP().setType("preonly")
    snes.getKSP().setTolerances(atol=abs_tol,rtol=rel_tol)
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType('mumps')


    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    snes.setFromOptions()
    J_mat = assemble_matrix(form(J_))
    J_mat.assemble()
    J_copy = J_mat.copy()
    J_copy.assemble()
    print("Jacobian norm:", J_copy.norm())
    return snes

# def solverNewtonLS(F, w, bcs=[],
#                     J_ = None,
#                     abs_tol=1e-10,
#                     rel_tol=1e-10,
#                     max_it=20,
#                     report=False):


def test_snes_motor():
    '''
    1. Define the mesh
    '''

    data_path = "motor_data/motor_mesh_1_new/"
    mesh_file = data_path + "motor_mesh_1"
    mesh, boundaries_mf, subdomains_mf, association_table = import_mesh(
        prefix=mesh_file,
        dim=2,
        subdomains=True
    )
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains_mf)
    dS = Measure('dS', domain=mesh, subdomain_data=boundaries_mf)

    '''
    2. Set up the PDE problem
    '''
    # PROBLEM SPECIFIC PARAMETERS
    Hc = 838.e3  # 838 kA/m
    p = 12
    s = 3 * p
    vacuum_perm = 4e-7 * np.pi
    angle = 0.
    iq = 282.2  / 0.00016231

    # states for mesh motion subproblem
    state_name_mm = 'uhat'
    state_function_space_mm = VectorFunctionSpace(mesh, ('CG', 1))
    state_function_mm = Function(state_function_space_mm)
    state_function_mm.vector.set(0.0)
    v_mm = TestFunction(state_function_space_mm)

    # Add state to the PDE problem:
    # states for electromagnetic equation
    state_name_em = 'u'
    state_function_space_em = FunctionSpace(mesh, ('CG', 1))
    state_function_em = Function(state_function_space_em)
    v_em = TestFunction(state_function_space_em)

    residual_form = pdeRes(state_function_em,v_em,state_function_mm,iq,dx,p,s,Hc,vacuum_perm,angle)


    '''
    3. Define the boundary conditions
    '''

    ############ Strongly enforced boundary conditions (mesh_old)#############
    ubc = Function(state_function_space_em)
    ubc.vector.set(0.0)
    locate_BC1 = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
                                lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))
    locate_BC2 = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
                                lambda x: np.isclose(x[0]**2+x[1]**2, 0.0036 ,atol=1e-6))
    locate_BC_list = [locate_BC1,locate_BC2]
    bc = []
    for locate_BC in locate_BC_list:
        bc.append(dirichletbc(ubc, locate_BC, state_function_em.function_space))

    du = TrialFunction(state_function_space_em)
    J = derivative(residual_form, state_function_em, du)

    # state_function_em.x.array[:] = 0.1
    snes = solveNonlinearSNES(residual_form, state_function_em, J_=J, bcs=bc, report=True)
    snes.solve(None, state_function_em.vector)
    # snes.view()

    print("Converged reason:", snes.getConvergedReason())
    
    ###### Solve by Newton's method ##########
    # solveNonlinear(residual_form, state_function_em, bcs=bc, report=True)
    gradA_z = gradx(state_function_em, state_function_mm)

    B_form = as_vector((gradA_z[1], -gradA_z[0]))
    dB_dAz = derivative(B_form, state_function_em)
    
    VB = VectorFunctionSpace(mesh,('DG',0))
    B = Function(VB)
    project(B_form,B)

    with XDMFFile(MPI.COMM_WORLD, "test_solutions/state_"+state_name_em+".xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        state_function_em.name = state_name_em
        xdmf.write_function(state_function_em)
    with XDMFFile(MPI.COMM_WORLD, "test_solutions/input_"+state_name_mm+".xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        state_function_mm.name = state_name_mm
        xdmf.write_function(state_function_mm)
    with XDMFFile(MPI.COMM_WORLD, "test_solutions/magnetic_flux_density.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        B.name = 'B'
        xdmf.write_function(B)

def test_snes_nonlinear_pde():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    u_bc = Function(V)
    u_bc.x.array[:] = 1.0
    bc = [dirichletbc(u_bc, locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                np.isclose(x[0], 1.0))))]
    u.x.array[:] = 0.9   
    snes = solveNonlinearSNES(F, u, bcs=bc, report=True)
    snes.solve(None, u.vector)
    # snes.view()

    print("Converged reason:", snes.getConvergedReason())


# test_snes_nonlinear_pde()
test_snes_motor()
    
    