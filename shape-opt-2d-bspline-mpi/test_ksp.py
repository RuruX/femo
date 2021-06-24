from dolfin import *
import numpy as np
import ufl
from petsc4py import PETSc

def m2p(A):
    return as_backend_type(A).mat()

def v2p(v):
    return as_backend_type(v).vec()

def transpose(A):
    """
    Transpose for matrix of DOLFIN type
    """
    return PETScMatrix(as_backend_type(A).mat().transpose(PETSc.Mat(MPI.comm_world)))

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

num_elements = 100
length = length = 1.0
N_fac = 8
height = height = length/N_fac
d = 2
k = 1
mesh = RectangleMesh(Point(0,0),Point(length,height),N_fac*num_elements,num_elements)

cell = mesh.ufl_cell()

VE = VectorElement("CG",cell,k)

I = Identity(d)
x = SpatialCoordinate(mesh)
dX = dx(metadata={"quadrature_degree":2*k})
topChar = conditional(gt(x[1],Constant(height-DOLFIN_EPS)),1.0,Constant(0.0))
h = Constant((0,-1))*topChar         # h: applied downward force on the top

V = FunctionSpace(mesh, VE)
u = Function(V)
v = TestFunction(V)

du = Function(V)
dR = Function(V)

def Ctimes(eps):
    K = Constant(1.0e6)
    mu = Constant(1.0e6)
    return K*tr(eps)*I + 2.0*mu*(eps - tr(eps)*I/3.0)

def pdeRes(u,v):
    epsu = sym(grad(u))
    sigma = Ctimes(epsu)
    return inner(sigma,grad(v))*dX - dot(h,v)*ds


def bcu():
    x = SpatialCoordinate(mesh)
    leftStr = "x[0] < DOLFIN_EPS"
    bcu = DirichletBC(V,Constant((0,0)),leftStr)
    return bcu
    

rank = MPI.comm_world.Get_rank()
R = pdeRes(u,v)
dR_du = derivative(R,u)
u.vector().set_local(np.ones(len(V.dofmap().dofs())))
du.vector().set_local(np.ones(len(V.dofmap().dofs())))
dR.vector().set_local(np.zeros(len(V.dofmap().dofs())))
A,B = assemble_system(dR_du, -R, bcs=[bcu()])

"""
solve linear system dR_du.T (A_T) * dR = du
"""
A_T = transpose(A)
A_p = m2p(A_T)
A_p.assemble()

#du_p = v2p(du.vector())
du_p = v2p(B)
du_p.assemble()
dR_p = PETSc.Vec().create()
dR_p.createNest([v2p(dR.vector())])
dR_p.setUp()
dR_p.assemble()

ksp = PETSc.KSP().create(MPI.comm_world) 
#ksp.setType(PETSc.KSP.Type.GMRES)

# operator = A_p, preconditioner = A_p;
ksp.setOperators(A_p,A_p)
pc = ksp.getPC()
pc.setType("asm")

# the overlap number is 1 by default
#pc.setASMOverlap(1)

ksp.setFromOptions()
ksp.setUp()

localKSP = pc.getASMSubKSP()[0]
localKSP.setType(PETSc.KSP.Type.GMRES)
localKSP.getPC().setType("lu")
localKSP.setTolerances(1.0e-7)
ksp.setGMRESRestart(40)
ksp.setConvergenceHistory()

import timeit
start = timeit.default_timer()
ksp.solve(du_p,dR_p)
stop = timeit.default_timer()

v2p(dR.vector()).ghostUpdate()
history = ksp.getConvergenceHistory()

#A_array = A_p.convert("dense").getDenseArray()
#print("Is A symmetric:",check_symmetric(A_array))


if rank == 0:
    print('time =', stop - start)
    print('KSP Solver Converged in', ksp.getIterationNumber(), 'iterations.')

    from matplotlib import pyplot as plt
    plt.semilogy(history)
    plt.xlabel('number of iterations')
    plt.ylabel('log(e)')
    plt.grid(True)
    plt.title('Residual Norm of the KSP Solver')
    plt.show()





