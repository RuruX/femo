"""
The ``utils`` module
--------------------
Contains problem-specific functionalities such as project higher-order dofs
to a lower order function space (for plotting), and calculate the wing volume
of an airplane model.
"""

import dolfinx
import dolfinx.io
import ufl
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (set_bc, Function, FunctionSpace, form, Constant,
                        assemble_scalar, VectorFunctionSpace)
from ufl import TestFunction, TrialFunction, dx, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


def projectPointForce(f_array, target_func, dx_=dx, bcs=[]):

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

    a = inner(Pv,w)*dx_ #lhs(res)
    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = A.createVecLeft()
    b.setArray(f_array)
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

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


def calculateSurfaceArea(mesh, boundary):

    #try to integrate a subset of the domain:
    Q = FunctionSpace(mesh, ("DG", 0))
    vq = TestFunction(Q)
    kappa = Function(Q)
    kappa.vector.setArray(np.ones(len(kappa.vector.getArray())))
    fixedCells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, boundary)

    with kappa.vector.localForm() as loc:
        loc.setValues(fixedCells, np.full(len(fixedCells), 0))

    s = assemble_scalar(form(vq*kappa*dx))
    surface_area = mesh.comm.allreduce(s, op=MPI.SUM)
    return surface_area

def computeNodalDisp(u):
    V = u.function_space
    mesh = V.mesh
    VCG1 = VectorFunctionSpace(mesh, ("CG", 1))
    u1 = Function(VCG1)
    u1.interpolate(u)
    uX = u1.sub(0).collapse().x.array
    uY = u1.sub(1).collapse().x.array
    uZ = u1.sub(2).collapse().x.array
    return uX,uY,uZ

def computeNodalDispMagnitude(u):
    uX, uY, uZ = computeNodalDisp(u)
    magnitude = np.zeros(len(uX))
    for i in range(len(uX)):
        magnitude[i] = np.sqrt(uX[i]**2+uY[i]**2+uZ[i]**2)
    return magnitude

class Delta:
    def __init__(self, x0, f_p, dist=1E-4, **kwargs):
        self.dist = dist
        self.x0 = x0
        self.f_p = f_p

    def eval(self, x):
        dist = self.dist
        values = np.zeros((3, x.shape[1]))
        for i in range(x.shape[1]):
            x_pt = np.array([x[0][i], x[1][i], x[2][i]])
            dist_ = np.linalg.norm(x_pt-self.x0)
            if dist_ < dist:
                values[0][i] = self.f_p[0]
                values[1][i] = self.f_p[1]
                values[2][i] = self.f_p[2]
                print(i)
        print(np.sum(values,axis=1))
        return values

class Delta_cpt:
    """
    Delta function on closest points
    """
    def __init__(self, x0, f_p, **kwargs):
        self.x0 = x0
        self.f_p = f_p

    def eval(self, x):
        dist = []
        values = np.zeros((3, x.shape[1]))
        closest_local = []
        for i in range(x.shape[1]):
            x_i = np.array([x[0][i], x[1][i], x[2][i]])
            dist_i = np.linalg.norm(x_i-self.x0)
            dist.append(dist_i)

        closest_local = np.argsort(np.array(dist))[:4]
        print(closest_local)
        print("applying forces to the closest point...")
        values[0][closest_local] = self.f_p[0]
        values[1][closest_local] = self.f_p[1]
        values[2][closest_local] = self.f_p[2]
        return values

class Delta_mpt:
    """
    Multi-point delta function applied on the closest points
    """
    def __init__(self, x0, f_p, **kwargs):
        self.x0 = x0
        self.f_p = f_p

    def eval(self, x):
        values = np.zeros((3, x.shape[1]))
        for j in range(self.x0.shape[0]):
            x0_j = self.x0[:][j]
            f_p_j = self.f_p[:][j]
            print(x0_j, f_p_j)

            dist = []
            closest_local = []
            for i in range(x.shape[1]):
                x_i = np.array([x[0][i], x[1][i], x[2][i]])
                dist_i = np.linalg.norm(x_i-x0_j)
                dist.append(dist_i)

            closest_local = np.argsort(np.array(dist))[:4]
            print(np.array(dist)[closest_local])
            print(closest_local)
            print("applying forces to the closest point...")
            values[0][closest_local] = f_p_j[0]
            values[1][closest_local] = f_p_j[1]
            values[2][closest_local] = f_p_j[2]
        return values


def getCellID(coord, mesh):
    # get bbt for the mesh
    mesh_bbt = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    # convert point in array with one element
    points_list_array = np.array([coord, ])
    # for each point, compute a colliding cells and append to the lists
    points_on_proc = []
    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions(mesh_bbt, points_list_array)  # get candidates
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points_list_array)  # get actual
    for i, point in enumerate(points_list_array):
        if len(colliding_cells.links(i)) > 0:
            cc = colliding_cells.links(i)[0]
            points_on_proc.append(point)
            cells.append(cc)
    # convert to numpy array
    points_on_proc = np.array(points_on_proc)
    cells = np.array(cells)
    return cells


def sortIndex(old_ind, new_ind):
    temp_ind = np.argsort(new_ind)
    ind = temp_ind[old_ind]
    row = len(new_ind)
    return ind

def applyNodalForces(f_array, mesh, W):
    """
    Applies input vertex forces to relevant DOF locations in force vector
    """

    # get vtx_to_dof map
    vtx_to_dof = getVertexToDofMap(W, mesh)

    # collapse map to vector
    vtx_to_dof = np.reshape(vtx_to_dof, (-1,1))

    # apply forces to vertex DOFs
    f_col = np.reshape(f_array, (-1,1))
    f1 = Function(W)
    f1_0,_ = f1.split()
    f_full = f1_0.vector.getArray()
    f_full[vtx_to_dof.astype('int')] = f_col
    f1_0.vector.setArray(f_full)

    return f1

def getVertexToDofMap(W, mesh):
    """
    Returns the "vertex to DOF map" with shape [nVtx,dim] containing
    the index of the DOF corresponding to each vertex/direction--used
    to directly map applied forces to force vector. This is not particularly
    straightforward but is more efficient than previous impelementations; see
    https://fenicsproject.discourse.group/t/application-of-point-forces-mapping-vertex-indices-to-corresponding-dofs/9646
    for more information
    """
   
    # extract the displacement subspace and associated dof_layout
    W0, W0_to_W = W.sub(0).collapse()
    dof_layout = W0.dofmap.dof_layout

    # use vertex/cell/dof relationships to identify the "parent" DOF
    # associated with each vertex
    num_vertices = mesh.topology.index_map(
        0).size_local + mesh.topology.index_map(0).num_ghosts
    vertex_to_par_dof_map = np.zeros(num_vertices, dtype=np.int32)
    num_cells = mesh.topology.index_map(
        mesh.topology.dim).size_local + mesh.topology.index_map(
        mesh.topology.dim).num_ghosts
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    for cell in range(num_cells):
        vertices = c_to_v.links(cell)
        dofs = W0.dofmap.cell_dofs(cell)
        for i, vertex in enumerate(vertices):
            vertex_to_par_dof_map[vertex] = dofs[dof_layout.entity_dofs(0, i)]

    # using the "parent" DOF for each vertex and dofmap block size (bs),
    # find the actual DOF index for each vertex/direction
    geometry_indices = dolfinx.cpp.mesh.entities_to_geometry(
        mesh, 0, np.arange(num_vertices, dtype=np.int32), False)
    bs = W0.dofmap.bs
    vtx_to_dof = np.zeros((num_vertices,bs), dtype=np.int32)
    for vertex, geom_index in enumerate(geometry_indices):
        par_dof = vertex_to_par_dof_map[vertex]
        for b in range(bs):
            vtx_to_dof[vertex, b] = W0_to_W[par_dof*bs+b]

    return vtx_to_dof

def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()
