"""
Reusable functions for the PETSc and UFL operations
"""
from dolfin import *
from petsc4py import PETSc
from scipy.spatial import KDTree

def errorNorm(v, v_ex):
    return errornorm(v,v_ex)

def getGlobalIndices(u_):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    u_PETSc = v2p(u_.vector())
    ind = u_PETSc.getLGMap().getIndices()
    return ind

def assembleSystem(J, F, bcs=[]):
    """
    Compute the array representations of the linear system
    ---------
    Returns: A, b
    """
    return assemble_system(J, F, bcs=bcs)

def assembleMatrix(M, bcs=[]):
    """
    Compute the matrix representation of the form
    """
    return assemble(M)

def assembleScalar(c):
    """
    Compute the array representation of the scalar form
    """
    return assemble(c)

def assembleVector(v):
    """
    Compute the array representation of the vector form
    """
    return assemble(v).get_local()

def createUnitSquareMesh(n):
    """
    Create unit square mesh for test purposes
    """
    return UnitSquareMesh(n, n)

def getFormArray(F):
    """
    Compute the array representation of the Form
    """
    return assemble(F).get_local()

def getFuncArray(v):
    """
    Compute the array representation of the Function
    """
    return v.vector().get_local()

def setFuncArray(v, v_array):
    """
    Set the fuction based on the array
    """
    v.vector().set_local(v_array)
    v2p(v.vector()).ghostUpdate()
    v2p(v.vector()).assemble()

#set_log_level(1)
def m2p(A):
    """
    Convert the matrix of DOLFIN type to a PETSc.Mat object
    """
    return as_backend_type(A).mat()

def v2p(v):
    """
    Convert the vector of DOLFIN type to a PETSc.Vec object
    """
    return as_backend_type(v).vec()

def transpose(A):
    """
    Transpose for matrix of DOLFIN type
    """
    return PETScMatrix(as_backend_type(A).mat().transpose(PETSc.Mat(MPI.comm_world)))

def computeMatVecProductFwd(A, x):
    """
    Compute y = A * x
    A: ufl form matrix
    x: ufl function
    """
    A_p = m2p(A)
    y = A_p * v2p(x.vector())
    y.assemble()
    return y.getArray()

def computeMatVecProductBwd(A, R):
    """
    Compute y = A.T * R
    A: ufl form matrix
    R: ufl function
    """
    row, col = m2p(A).getSizes()
    y = PETSc.Vec().create()
    y.setSizes(col)
    y.setUp()
    m2p(A).multTranspose(v2p(R.vector()),y)
    y.assemble()
    return y.getArray()


def zero_petsc_vec(size, comm=MPI.comm_world):
    """
    Create zero PETSc vector of size ``num_el``.
    Parameters
    ----------
    size : int
    vec_type : str, optional
        For petsc4py.PETSc.Vec types, see petsc4py.PETSc.Vec.Type.
    comm : MPI communicator
    Returns
    -------
    v : petsc4py.PETSc.Vec
    """
    v = PETSc.Vec().create(comm)
    v.setSizes(size)
    v.setUp()
    v.assemble()
    return v

def zero_petsc_mat(row, col, comm=MPI.comm_world):
    """
    Create zeros PETSc matrix with shape (``row``, ``col``).
    Parameters
    ----------
    row : int
    col : int
    mat_type : str, optional
        For petsc4py.PETSc.Mat types, see petsc4py.PETSc.Mat.Type
    comm : MPI communicator
    Returns
    -------
    A : petsc4py.PETSc.Mat
    """
    A = PETSc.Mat(comm)
    A.createAIJ([row, col], comm=comm)
    A.setUp()
    A.assemble()
    return A

def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()

def updateR(f, f_value):
    f.assign(Constant(float(f_value)))
    
def update(f, f_values):
    """
    Update the nodal values in every dof of the DOLFIN function `f`
    according to `f_values`.
    -------------------------
    f: dolfin function
    f_values: numpy array
    """
    setFuncArray(f, f_values)

def findNodeIndices(node_coordinates, coordinates):
    """
    Find the indices of the closest nodes, given the `node_coordinates`
    for a set of nodes and the `coordinates` for all of the vertices
    in the mesh, by using scipy.spatial.KDTree
    """
    tree = KDTree(coordinates)
    dist, node_indices = tree.query(node_coordinates)
    return node_indices

I = Identity(2)
def gradx(f,uhat):
    """
    Convert the differential operation from the reference domain
    to the measure in the deformed configuration based on the mesh
    movement of `uhat`
    --------------------------
    f: DOLFIN function for the solution of the physical problem
    uhat: DOLFIN function for mesh movements
    """
    return dot(grad(f), inv(I + grad(uhat)))

def J(uhat):
    """
    Compute the determinant of the deformation gradient used in the
    integration measure of the deformed configuration wrt the the
    reference configuration.
    ---------------------------
    uhat: DOLFIN function for mesh movements
    """
    return det(I + grad(uhat))


