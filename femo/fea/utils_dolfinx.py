"""
Utility functions for the PETSc and UFL operations
"""

import dolfinx
from dolfinx.io import XDMFFile
from ufl import (Identity, dot, derivative, TestFunction, TrialFunction,
                inner, ds, dS, dx, grad, inv, as_vector, sqrt, conditional, lt,
                det, Measure, exp, tr, CellDiameter)
from dolfinx.mesh import (create_unit_square, create_rectangle, create_interval,
                            locate_entities_boundary, locate_entities,
                            meshtags)
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (form, assemble_scalar, Function, FunctionSpace,
                        dirichletbc, locate_dofs_geometrical, Constant)
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix,
                        NonlinearProblem, apply_lifting, set_bc,
                        create_matrix, _assemble_matrix_mat,)
from dolfinx.nls.petsc import NewtonSolver as PETScNewtonSolver
from dolfinx import la
from petsc4py import PETSc
from scipy.spatial import KDTree
from mpi4py import MPI
import numpy as np
from scipy.spatial import KDTree
from configparser import ConfigParser
import ufl 

from scipy.spatial import KDTree

DOLFIN_EPS = 3E-16
comm = MPI.COMM_WORLD


def readFEAMesh(meshFile, format="HDF"):
    """
    Reads mesh from input meshFile, optionally display statistics
    """

    if format == "HDF": # recommended format
        # both xmdf and h5 files are available
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, meshFile, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
    elif format == "XML": # it needs to have lxml module installed
        # only xdmf file is available
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, meshFile, "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
    else:
        raise ValueError("Invalid mesh file type. Must be 'HDF' or 'XML'")

    return mesh


def gradx(f,uhat):
    """
    Convert the differential operation from the reference domain
    to the measure in the deformed configuration based on the mesh
    movement of `uhat`
    --------------------------
    f: DOLFINx function for the solution of the physical problem
    uhat: DOLFIN function for mesh movements
    """
    return dot(grad(f), inv(F(uhat)))


def J(uhat):
    """
    Compute the determinant of the deformation gradient used in the
    integration measure of the deformed configuration wrt the the
    reference configuration.
    ---------------------------
    uhat: DOLFINx function for mesh movements
    """
    return det(F(uhat))

def F(uhat):
    """
    Compute the determinant of the deformation gradient used in the
    integration measure of the deformed configuration wrt the the
    reference configuration.
    ---------------------------
    uhat: DOLFINx function for mesh movements
    """
    order = uhat.function_space.mesh.topology.dim
    I = Identity(order) 
    return I + grad(uhat)

# The dolfinx version for mesh importer from msh2xdmf module
def import_mesh(
        prefix="mesh",
        subdomains=False,
        dim=2,
        directory=".",
):
    """Function importing a dolfinx mesh.

    Arguments:
        prefix (str, optional): mesh files prefix (eg. my_mesh.msh,
            my_mesh_domain.xdmf, my_mesh_bondaries.xdmf). Defaults to "mesh".
        subdomains (bool, optional): True if there are subdomains. Defaults to
            False.
        dim (int, optional): dimension of the domain. Defaults to 2.
        directory (str, optional): directory of the mesh files. Defaults to ".".

    Output:
        - dolfinx Mesh object containing the domain;
        - dolfinx MeshFunction object containing the physical lines (dim=2) or
            surfaces (dim=3) defined in the msh file and the sub-domains;
        - association table
    """
    # Set the file name
    domain = "{}_domain.xdmf".format(prefix)
    boundaries = "{}_boundaries.xdmf".format(prefix)

    # Import the converted domain
    with XDMFFile(MPI.COMM_WORLD,
                    "{}/{}".format(directory, domain), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim-1, tdim)
    # Import the boundaries
    with XDMFFile(MPI.COMM_WORLD,
                    "{}/{}".format(directory, boundaries), "r") as xdmf:
        boundaries_mf = xdmf.read_meshtags(mesh, "Grid")
    # Import the subdomains
    if subdomains:
        with XDMFFile(MPI.COMM_WORLD,
                        "{}/{}".format(directory, domain), "r") as xdmf:
            subdomains_mf = xdmf.read_meshtags(mesh, 'Grid')
    # Import the association table
    association_table_name = "{}/{}_{}".format(
        directory, prefix, "association_table.ini")
    file_content = ConfigParser()
    file_content.read(association_table_name)
    association_table = dict(file_content["ASSOCIATION TABLE"])
    # Convert the value from string to int
    for key, value in association_table.items():
        association_table[key] = int(value)
    # Return the Mesh and the MeshFunction objects
    if not subdomains:
        return mesh, boundaries_mf, association_table
    else:
        return mesh, boundaries_mf, subdomains_mf, association_table


def findNodeIndices(node_coordinates, coordinates):
    """
    Find the indices of the closest nodes, given the `node_coordinates`
    for a set of nodes and the `coordinates` for all of the vertices
    in the mesh, by using scipy.spatial.KDTree
    """
    tree = KDTree(coordinates)
    dist, node_indices = tree.query(node_coordinates)
    return node_indices

def createUnitSquareMesh(n):
    """
    Create unit square mesh for test purposes
    """
    return create_unit_square(MPI.COMM_WORLD, n, n)

def createIntervalMesh(n, x0, x1):
    """
    Create interval mesh for test purposes
    """
    return create_interval(MPI.COMM_WORLD, n, [x0, x1])

def createRectangleMesh(pt1,pt2,nx,ny):
    """
    Create rectangle mesh for test purposes
    """
    return create_rectangle(MPI.COMM_WORLD, [pt1, pt2], [nx,ny],
                            cell_type=CellType.quadrilateral)

def getFuncArray(v):
    """
    Compute the array representation of the Function
    """
    return v.vector.getArray()

def setFuncArray(v, v_array):
    """
    Set the fuction based on the array
    """
    v.vector[:] = v_array
    v.vector.assemble()
    v.vector.ghostUpdate()

def assembleScalar(c):
    """
    Compute the array representation of the scalar form
    """
    return assemble_scalar(form(c))

def assembleVector(v):
    """
    Compute the array representation of the vector form
    """
    return assemble_vector(form(v)).array

def assembleMatrix(M, bcs=[]):
    """
    Compute the array representation of the matrix form
    """
    M_ = assemble_matrix(form(M), bcs=bcs)
    M_.assemble()
    return M_

def assembleSystem(J, F, bcs=[]):
    """
    Compute the array representations of the linear system
    """
    a = form(J)
    L = form(F)
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    L = form(F)
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    return A, b

def assemble(f, dim=0, bcs=[]):
    if dim == 0:
        return assembleScalar(f)
    elif dim == 1:
        return assembleVector(f)
    elif dim == 2:
        M = assembleMatrix(f, bcs=bcs)
        return convertToDense(M.copy())
    else:
        return TypeError("Invalid type for assembly.")


def assemble_partials(of=None, wrt=None, dim=1):
    """
    util method for assembling the partial derivative matrices for
    education or verification.
    """
    partials = derivative(of, wrt)
    return assemble(partials, dim=dim)


def errorNorm(v, v_ex, norm='L2'):
    """
    Calculate the L2 norm of two functions
    """
    comm = MPI.COMM_WORLD
    l2_error = (v - v_ex)**2 * dx
    if norm == 'L2':
        error = form(l2_error)
    elif norm == 'H1':
        h1_error = l2_error + (grad(v) - grad(v_ex))**2 * dx
        error = form(h1_error)
    E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    return E


##### Linear algebra
def transpose(A):
    """
    Transpose for matrix of DOLFIN type
    """
    return A.transpose(PETSc.Mat(MPI.COMM_WORLD))

from scipy.sparse import csr_matrix
def convertToCOO(A):
    """
    Convert a PETSc matrix to scipy.coo Matrix
    """
    A_csr = csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

    return A_csr.tocoo()

def computeMatVecProductFwd(A, x):
    """
    Compute y = A * x
    A: PETSc matrix
    x: ufl function
    """
    y = A*x.vector
    y.assemble()
    return y.getArray()

def applyBC(res, u, bcs):
    a = form(derivative(res, u))
    L = form(res)
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    return b.array

def computeMatVecProductBwd(A, R):
    """
    Compute y = A.T * R
    A: PETSc matrix
    R: ufl function
    """
    row, col = A.getSizes()
    y = PETSc.Vec().create()
    y.setSizes(col)
    y.setUp()
    A.multTranspose(R.vector,y)
    y.assemble()
    return y.getArray()


def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()


def update(v, v_values):
    """
    Update the nodal values in every dof of the DOLFIN function `v`
    according to `v_values`.
    -------------------------
    v: dolfin function
    v_values: numpy array
    """
    if len(v_values) == 1:
        v.vector.set(v_values)
    else:
        setFuncArray(v, v_values)

def computePartials(form, function):
    return derivative(form, function)

def createFunction(function):
    return Function(function.function_space)

def solveNonlinear(res, func, bc, solver, report, initialize):
    from timeit import default_timer
    start = default_timer()
    if solver == 'Newton':
        newton_solver = NewtonSolver(res, func, bc,
                                    initialize=initialize,
                                    report=report)
        newton_solver.solve(func)
    elif solver == 'SNES':
        snes_solver = SNESSolver(res, func, bc, report=report)
        snes_solver.solve(None, func.vector)
        print("Converged reason:", snes_solver.getConvergedReason())
    stop = default_timer()
    if report is True:
        print("Solve nonlinear finished in ",stop-start, "seconds")


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
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

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


def SNESSolver(F, w, bcs=[],
                    abs_tol=1e-13,
                    rel_tol=1e-13,
                    max_it=100,
                    report=False):
    """
    https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/nls/test_newton.py#L182-L205
    """
    # Create nonlinear problem

    problem = NonlinearSNESProblem(F, w, bcs)

    W = w.function_space
    b = la.create_petsc_vector(W.dofmap.index_map, W.dofmap.index_map_bs)
    J = create_matrix(problem.a)
    # Create Newton solver and solve
    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts['snes_type'] = 'newtonls'
    opts['snes_linesearch_type'] = 'basic'
    # Ru: the choice of damping parameter seems to be mesh dependent;
    # for the fine motor mesh, it is 0.8; for the coarse mesh, it is 0.61.
    # opts['snes_linesearch_damping'] = 0.8
    opts["error_on_nonconvergence"] = True
    if report is True:
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        opts['snes_monitor'] = None
        opts['snes_linesearch_monitor'] = None
    snes.setTolerances(atol=abs_tol, rtol=rel_tol, max_it=max_it)
    snes.getKSP().setType("preonly")
    snes.getKSP().setTolerances(atol=abs_tol,rtol=rel_tol)
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType('mumps')


    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    snes.setFromOptions()

    return snes


def NewtonSolver(F, w, bcs=[],
                    abs_tol=1e-50,
                    rel_tol=1e-30,
                    max_it=3,
                    initialize=False,
                    error_on_nonconvergence=False,
                    report=False):

    """
    Wrap up the nonlinear solver for the problem F(w)=0 and
    returns the solution
    """
    problem = NonlinearProblem(F, w, bcs)
    # Set the initial guess of the solution
    if initialize is True:
        with w.vector.localForm() as w_local:
            w_local.set(0.1)
    solver = PETScNewtonSolver(MPI.COMM_WORLD, problem)
    if report is True:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    # w.vector.set(0.0)
    # Set the Newton solver options
    solver.atol = abs_tol
    solver.rtol = rel_tol
    solver.max_it = max_it
    solver.error_on_nonconvergence = error_on_nonconvergence
    opts = PETSc.Options()
    opts["nls_solve_pc_factor_mat_solver_type"] = "mumps"

    return solver

def solveKSP(A, b, x):
    """
    Wrap up the KSP solver for the linear system Ax=b
    """
    ######### Set up the KSP solver ###############

    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)

    # additive Schwarz method
    pc = ksp.getPC()
    pc.setType("asm")

    ksp.setFromOptions()
    ksp.setUp()

    localKSP = pc.getASMSubKSP()[0]
    localKSP.setType(PETSc.KSP.Type.GMRES)
    localKSP.getPC().setType("lu")
    localKSP.setTolerances(1.0e-12)
    #ksp.setGMRESRestart(30)
    ksp.setConvergenceHistory()
    ksp.solve(b, x)
    history = ksp.getConvergenceHistory()

def solveKSP_mumps(A, b, x):
    """
    Implementation of KSP solution of the linear system Ax=b using MUMPS
    """

    # setup petsc for pre-only solve
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")

    # set LU w/ MUMPS
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType('mumps')

    # solve
    ksp.setUp()
    ksp.solve(b, x)

def setUpKSP_MUMPS(A):
    """
    Implementation of KSP solution of the linear system Ax=b using MUMPS
    """

    # setup petsc for pre-only solve
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")

    # set LU w/ MUMPS
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType('mumps')

    # solve
    ksp.setUp()
    return ksp

def move(mesh, u):
    x = mesh.geometry.x
    gdim = mesh.geometry.dim
    # u_x = u.compute_point_values()
    for i in range(gdim):
        ui = u.sub(i).collapse().x.array
        x[:,i] += ui

def moveBackward(mesh, u):
    x = mesh.geometry.x
    gdim = mesh.geometry.dim
    # u_x = u.compute_point_values()
    for i in range(gdim):
        ui = u.sub(i).collapse().x.array
        x[:,i] += -ui

def meshSize(mesh):
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    h = dolfinx.cpp.mesh.h(mesh, tdim, range(num_cells))
    return h

def createCustomMeasure(mesh, dim, SubdomainFunc, measure: str, tag: int):
    metadata = {"quadrature_degree":4}
    if measure == 'ds':
        subdomain = locate_entities_boundary(mesh,dim,SubdomainFunc)
    else:
        subdomain = dolfinx.mesh.locate_entities(mesh,dim,SubdomainFunc)
    subdomain_tag = dolfinx.mesh.meshtags(mesh, dim, subdomain,
                        np.full(len(subdomain),tag,dtype=np.int32))
    custom_measure = ufl.Measure(measure,domain=mesh,
                        subdomain_data=subdomain_tag,metadata=metadata)
    return custom_measure


def project(v, target_func, bcs=[], lump_mass=False):

    """
    L2 projection of an UFL object (expression) to targeted function.
    Typically used for visualization in post-processing.
    `lump_mass` is an optional boolean argument set to be False by default;
    it's set to be True when lumping is needed for preventing oscillation
    when projecting discontinous data.
    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)

    L = inner(v, w) * dx
    if(lump_mass):
        a = inner(Constant(V.mesh, 1.0),w)*dx
        A = assemble_vector(form(a))
        b = assemble_vector(form(L))
        target_func.vector.pointwiseDivide(b,A)
    else:
        a = inner(Pv,w)*dx #lhs(res)
        # a = inner(Pv, w) * dx
        # Assemble linear system
        A = assemble_matrix(form(a), bcs)
        A.assemble()
        b = assemble_vector(form(L))
        apply_lifting(b, [form(a)], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        solver.solve(b, target_func.vector)



def findNodeIndices(node_coordinates, coordinates):
    """
    Find the indices of the closest nodes, given the `node_coordinates`
    for a set of nodes and the `coordinates` for all of the vertices
    in the mesh, by using scipy.spatial.KDTree
    """
    tree = KDTree(coordinates)
    dist, node_indices = tree.query(node_coordinates)
    return node_indices

# def locateDOFs(coords,V):
#     """
#     Find the indices of the dofs for setting up the boundary condition
#     in the mesh motion subproblem
#     """
#     coordinates = V.tabulate_dof_coordinates()[:,:-1]
#
#     # Use KDTree to find the node indices of the points on the edge
#     # in the mesh object in FEniCS
#     node_indices = findNodeIndices(np.reshape(coords, (-1,2)),
#                                     coordinates)
#
#     # Convert the node indices to edge indices, where each node has 2 dofs
#     edge_indices = np.empty(2*len(node_indices))
#     for i in range(len(node_indices)):
#         edge_indices[2*i] = 2*node_indices[i]
#         edge_indices[2*i+1] = 2*node_indices[i]+1
#
#     return edge_indices.astype('int')


def locateDOFs(coords,V, input='polar'):
    """
    Find the indices of the dofs for setting up the boundary condition
    in the mesh motion subproblem
    """
    coords = np.reshape(coords[:], (-1,2))
    if input == 'polar':
        for i in range(coords.shape[0]):
            theta, r =  coords[i,:]
            coords[i,:] = np.array([
                r*np.cos(theta), r*np.sin(theta)
            ])

    coordinates = V.tabulate_dof_coordinates()[:,:-1]
    # Use KDTree to find the node indices of the points on the edge
    # in the mesh object in FEniCS
    node_indices = findNodeIndices(coords, coordinates)

    # Convert the node indices to edge indices, where each node has 2 dofs
    edge_indices = np.empty(2*len(node_indices))
    for i in range(len(node_indices)):
        edge_indices[2*i] = 2*node_indices[i]
        edge_indices[2*i+1] = 2*node_indices[i]+1

    return edge_indices.astype('int')

import meshio
def reconstructFEAMesh(filename, nodes, connectivity):
    # Generate cells (connectivity)
    # This is a placeholder, replace with your actual cell data
    cells = [("quad", np.array(connectivity))]
    # Write the mesh data to an XDMF file
    mesh = meshio.Mesh(nodes, cells)
    meshio.write(filename, mesh)
    wing_shell_mesh_dolfinx = readFEAMesh(filename)
    return wing_shell_mesh_dolfinx
