#import dolfinx
#from dolfinx import (Form, Function, FunctionSpace, NewtonSolver,
#                     NonlinearProblem, UnitSquareMesh, log)
from dolfinx import *
#from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh, fem
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_geometrical, assemble_matrix, assemble_vector,
                         apply_lifting, set_bc, locate_dofs_topological,assemble_scalar)
import dolfinx.la
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from ufl import (MixedElement, VectorElement, FiniteElement, TestFunctions,
                 TestFunction, TrialFunction,
                 derivative, diff, dx, grad, inner, variable, SpatialCoordinate,
                 CellNormal,as_vector,Jacobian,sqrt,dot,cross,as_matrix,sym,indices,
                 as_tensor,split,Cell)
import numpy as np
import dolfinx.geometry

# Import manifold mesh of topological dimension 2 and geometric dimension 3:

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "meshout_quad.xdmf", "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")

# Problem parameters:
E = Constant(mesh,4.32e8) # Young's modulus
nu = Constant(mesh,0.0) # Poisson ratio
h = Constant(mesh,0.25) # Shell thickness
f = Constant(mesh,(0,0,-10)) # Body force per unit volume

DOLFIN_EPS = Constant(mesh,1.0e-8)

# Write it out in a visualizable format for Paraview:
#File("meshout.pvd") << mesh
dolfinx.io.VTKFile("plate-meshout.pvd").write(mesh)

# Make sure the necessary finite element spaces are supported for this mesh:
cell = mesh.ufl_cell()
VE = VectorElement("Lagrange",cell,1)
VE2 = VectorElement("DG",cell,0)
#VE2 = VectorElement("Discontinuous Lagrange",cell,0)
#WE = MixedElement([VE,VE])
W1 = FunctionSpace(mesh,VE)
WE = MixedElement([VE,VE])
W = FunctionSpace(mesh,WE)
#W = FunctionSpace(mesh,VE*VE)

#W1 = FunctionSpace(mesh,("DG",0))
W2 = FunctionSpace(mesh,VE2)
w_out = Function(W2)
# Reduced one-point quadrature:
dx3 = dx
dx = dx(metadata={"quadrature_degree":0})

# Solution function:
w = Function(W)
u_mid,theta = ufl.split(w)

# Reference configuration of midsurface:
X_mid = SpatialCoordinate(mesh)

# Normal vector to each element is the third basis vector of the
# local orthonormal basis (indexed from zero for consistency with Python):
E2 = CellNormal(mesh)
print("E0_int:")
print(assemble_vector(TestFunction(FunctionSpace(mesh,("DG",0)))*E2[0]*dx3).getArray())
print("E1_int:")
print(assemble_vector(TestFunction(FunctionSpace(mesh,("DG",0)))*E2[1]*dx3).getArray())
print("E2_int:")
print(assemble_vector(TestFunction(FunctionSpace(mesh,("DG",0)))*E2[2]*dx3).getArray())
#print(assemble_vector(TestFunction(FunctionSpace(mesh,("DG",0)))*E2[2]*dx).getArray())
print("E0,E1,E2:")
#print(E2[1])
print(assemble_scalar(E2[0]*dx3))
print(assemble_scalar(E2[1]*dx3))
print(assemble_scalar(E2[2]*dx3))
#print(type(E2))

Vatt=VectorFunctionSpace(mesh,("DG",0))
uatt = TrialFunction(Vatt)
vatt = TestFunction(Vatt)
#print(E2.str(True))
nh=Function(Vatt)
a_att = inner(uatt,vatt)*dx
l_att = inner(E2,vatt)*dx
A_att=assemble_matrix(a_att)
L_att=assemble_vector(l_att)
problem=fem.LinearProblem(a_att,l_att)
nh=problem.solve()
print(nh.vector.getArray())
print(L_att.getArray())
print(assemble_vector(TestFunction(FunctionSpace(mesh,("DG",0)))*1*dx3).getArray())
    

# Local in-plane orthogonal basis vectors, with 0-th basis vector along
# 0-th parametric coordinate direction (where Jacobian[i,j] is the partial
# derivatiave of the i-th physical coordinate w.r.t. to j-th parametric
# coordinate):
A0 = as_vector([Jacobian(mesh)[j,0] for j in range(0,3)])
E0 = A0/sqrt(dot(A0,A0))
E1 = cross(E2,E0)

# Matrix for change-of-basis to/from local/global Cartesian coordinates;
# E01[i,j] is the j-th component of the i-th basis vector:
E01 = as_matrix([[E0[i] for i in range(0,3)],
                 [E1[i] for i in range(0,3)]])

# Displacement at through-thickness coordinate xi2:
def u(xi2):
    # Formula (7.1) from http://www2.nsysu.edu.tw/csmlab/fem/dyna3d/theory.pdf
    return u_mid - xi2*cross(E2,theta)

# In-plane gradient components of displacement in the local orthogonal
# coordinate system:
def gradu_local(xi2):
    gradu_global = grad(u(xi2)) # (3x3 matrix, zero along E2 direction)
    i,j,k,l = indices(4)
    return as_tensor(E01[i,k]*gradu_global[k,l]*E01[j,l],(i,j))

# In-plane strain components of local orthogonal coordinate system at
# through-thickness coordinate xi2, in Voigt notation:
def eps(xi2):
    eps_mat = sym(gradu_local(xi2))
    return as_vector([eps_mat[0,0], eps_mat[1,1], 2*eps_mat[0,1]])

# Transverse shear strains in local coordinates at given xi2, as a vector
# such that gamma_2(xi2)[i] = 2*eps[i,2], for i in {0,1}
def gamma_2(xi2):
    dudxi2_global = -cross(E2,theta)
    i,j = indices(2)
    dudxi2_local = as_tensor(dudxi2_global[j]*E01[i,j],(i,))
    gradu2_local = as_tensor(dot(E2,grad(u(xi2)))[j]*E01[i,j],(i,))
    return dudxi2_local + gradu2_local

# Voigt notation material stiffness matrix for plane stress:
D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                 [nu,   1.0,  0.0         ],
                                 [0.0,  0.0,  0.5*(1.0-nu)]])

# Elastic energy per unit volume at through-thickness coordinate xi2:
def energyPerUnitVolume(xi2):
    G = E/(2*(1+nu))
    return 0.5*(dot(eps(xi2),D*eps(xi2))
                + G*inner(gamma_2(xi2),gamma_2(xi2))) # ?!?!?

# Some code copy-pasted from tIGAr and ShNAPr for Gaussian quadrature through
# the thickness of the shell structure:
def getQuadRule(n):
    """
    Return a list of points and a list of weights for integration over the
    interval (-1,1), using ``n`` quadrature points.  
    """
    if(n==1):
        xi = [Constant(mesh,0.0),]
        w = [Constant(mesh,2.0),]
        return (xi,w)
    if(n==2):
        xi = [Constant(mesh,-0.5773502691896257645091488),
              Constant(mesh,0.5773502691896257645091488)]
        w = [Constant(mesh,1.0),
             Constant(mesh,1.0)]
        return (xi,w)
    if(n==3):
        xi = [Constant(mesh,-0.77459666924148337703585308),
              Constant(mesh,0.0),
              Constant(mesh,0.77459666924148337703585308)]
        w = [Constant(mesh,0.55555555555555555555555556),
             Constant(mesh,0.88888888888888888888888889),
             Constant(mesh,0.55555555555555555555555556)]
        return (xi,w)
    if(n==4):
        xi = [Constant(mesh,-0.86113631159405257524),
              Constant(mesh,-0.33998104358485626481),
              Constant(mesh,0.33998104358485626481),
              Constant(mesh,0.86113631159405257524)]
        w = [Constant(mesh,0.34785484513745385736),
             Constant(mesh,0.65214515486254614264),
             Constant(mesh,0.65214515486254614264),
             Constant(mesh,0.34785484513745385736)]
        return (xi,w)
    
    print("ERROR: invalid number of quadrature points requested.")
    exit()

def getQuadRuleInterval(n,L):
    """
    Returns an ``n``-point quadrature rule for the interval 
    (-``L``/2,``L``/2), consisting of a list of points and list of weights.
    """
    xi_hat, w_hat = getQuadRule(n)
    xi = []
    w = []
    for i in range(0,n):
        xi += [L*xi_hat[i]/2.0,]
        w += [L*w_hat[i]/2.0,]
    return (xi,w)

class ThroughThicknessMeasure:
    """
    Class to represent a local integration through the thickness of a shell.
    The ``__rmul__`` method is overloaded for an instance ``dxi2`` to be
    used like ``volumeIntegral = volumeIntegrand*dxi2*dx``, where
    ``volumeIntegrand`` is a python function taking a single parameter,
    ``xi2``.
    """
    def __init__(self,nPoints,h):
        """
        Integration uses a quadrature rule with ``nPoints`` points, and assumes
        a thickness ``h``.
        """
        self.nPoints = nPoints
        self.h = h
        self.xi2, self.w = getQuadRuleInterval(nPoints,h)

    def __rmul__(self,integrand):
        """
        Given an ``integrand`` that is a Python function taking a single
        ``float`` parameter with a valid range of ``-self.h/2`` to 
        ``self.h/2``, return the (numerical) through-thickness integral.
        """
        integral = 0.0
        for i in range(0,self.nPoints):
            integral += integrand(self.xi2[i])*self.w[i]
        return integral

"""
self-defined function to project the function vector onto a DG-0 space
"""
def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w2 = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = ufl.inner(Pv, w2) * dx
    L = ufl.inner(v, w2) * dx

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)


    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

# Integrate through the thickness numerically to get total energy (where dx
# is a UFL Measure, integrating over midsurface area).
dxi2 = ThroughThicknessMeasure(3,h)
elasticEnergy = energyPerUnitVolume*dxi2*dx

# Take a Gateaux derivative and add a source term to obtain the
# weak form of the problem:
dw = TestFunction(W)
du_mid,dtheta = ufl.split(dw)
n = CellNormal(mesh)
F = derivative(elasticEnergy,w,dw) - inner(f,du_mid)*dx \
    + Constant(mesh,1e-1)*E*h*dot(theta,n)*dot(dtheta,n)*dx  # !?!?!?!

# LHS of linearized problem J == -F:
J = derivative(F,w)

# Set up test problem w/ BCs that fix all DoFs to zero for all nodes
# with x[0] negative.
#bcs = [DirichletBC(W,Constant(mesh,6*(0,)),"x[0] > 550"),]

#facets = locate_entities_boundary(mesh, 1,
#                                  lambda x: np.logical_or(x[0] < np.finfo(float).eps,
#                                                          x[0] > 1.0 - np.finfo(float).eps))
#bc = DirichletBC(u0, locate_dofs_topological(V, 1, facets))

u0 = Function(W)
u0.vector.set(0.0)


#bc1
locate_fixed_BC = [locate_dofs_geometrical((W.sub(0),W.sub(0).collapse()),lambda x: np.isclose(x[0], 0. ,atol=1e-6)),#np.logical_or(np.logical_or(x[1] < -50, x[1] > 880),np.logical_or(x[0] < 700, x[0] > 1430))),
                   locate_dofs_geometrical((W.sub(1),W.sub(1).collapse()),lambda x: np.isclose(x[0], 0. ,atol=1e-6)),#np.logical_or(np.logical_or(x[1] < -50, x[1] > 880),np.logical_or(x[0] < 700, x[0] > 1430))),
                  ]
print("#bc1 ",locate_fixed_BC)

bcs = [DirichletBC(u0,locate_fixed_BC)]

#print(locate_fixed_BC)


#bc2
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()), lambda x: x[0] > -1e-6)
ubc1=  Function(W.sub(0).collapse())
with ubc1.vector.localForm() as uloc:
     uloc.set(0.)
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()), lambda x: x[0] > -1e-6)
ubc2=  Function(W.sub(1).collapse())
with ubc2.vector.localForm() as uloc:
     uloc.set(0.)
#print("#bc2 ",locate_BC1)
#print("#bc2 ",locate_BC2)
'''
bcs = [DirichletBC(ubc1, locate_BC1, W.sub(0)),
       DirichletBC(ubc2, locate_BC2, W.sub(1))]
'''

#bc3
#locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC1 = locate_dofs_geometrical(W.sub(0).collapse(), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
#print("#bc3 ",locate_BC1)
#print("#bc3 ",locate_BC2)
'''
bcs = [DirichletBC(u0,locate_BC1),
       DirichletBC(u0,locate_BC2),]
'''

#bc4
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
#print("#bc4 ",locate_BC1)
'''
bcs = [DirichletBC(u0,locate_BC1),]
'''

#bc5
locate_BC = [locate_dofs_geometrical((W.sub(0).sub(0), W.sub(0).sub(0).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6)),
             locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6)),
             locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6)),
             ]
#print("#bc5 ",locate_BC)
'''
bcs = [DirichletBC(u0,locate_BC),]
'''

#bc6
locate_BC = [locate_dofs_geometrical([(W.sub(0).sub(1).collapse()),(W.sub(0).sub(2).collapse())], lambda x: np.isclose(x[0], 0. ,atol=1e-6)),
            ]
#print("#bc6 ",locate_BC)
'''
bcs = [DirichletBC(u0,locate_BC),]
'''

#bc7
locate_BC = [locate_dofs_geometrical(W1, lambda x: np.isclose(x[0], 0. ,atol=1e-6)),
            ]
#print("#bc7 ",locate_BC)
'''
bcs = [DirichletBC(u0,locate_BC),]
'''

#bc8
#locate_dofs_geometrical((W.sub(0).collapse(),W.sub(1).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6)),

#bc9
def boundary(x):
    return np.isclose(x[0], 0. ,atol=1e-6)

facets = locate_entities_boundary(mesh, 1, boundary)
locate_BC = [locate_dofs_topological(W,1, facets),
            ]
#print("#bc9 ",locate_BC)

#bcs = [DirichletBC(u0,locate_BC),]

#bc10
"""
!!!!!!
RuntimeError: Cannot evaluate dof coordinates - this element does not have pointwise evaluation.
"""
#locate_fixed_BC = locate_dofs_geometrical(W,lambda x: np.isclose(x[0], 0. ,atol=1e-6))
#print("#bc10 ",locate_fixed_BC)

#bcs = [DirichletBC(u0,locate_fixed_BC)]

#(W.sub(0).collapse(),W.sub(1).collapse())
#print("locate_BC1",locate_BC1)
#print("locate_BC2",locate_BC2)
#print("locate_BC3",locate_BC3)
#print("locate_BC4",locate_BC4)
#print("locate_BC5",locate_BC5)

'''
locate_BC1 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()), lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC3 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()), lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC4 = locate_dofs_geometrical((W.sub(1).sub(0), W.sub(1).sub(0).collapse()), lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC5 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()), lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC6 = locate_dofs_geometrical((W.sub(0).sub(0), W.sub(0).sub(0).collapse()), lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC7 = locate_dofs_geometrical((W.sub(1).sub(1), W.sub(1).sub(1).collapse()), lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC8 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()), lambda x: np.isclose(x[0], 25. ,atol=1e-6))

#locate_fixed_BC = locate_dofs_geometrical(W.sub(0).sub(1),lambda x: np.isclose(x[0],0.0))
ubc1=  Function(W.sub(0).sub(1).collapse())
with ubc1.vector.localForm() as uloc:
     uloc.set(0.)
ubc2=  Function(W.sub(0).sub(2).collapse())
with ubc2.vector.localForm() as uloc:
     uloc.set(0.)
ubc3=  Function(W.sub(0).sub(1).collapse())
with ubc3.vector.localForm() as uloc:
     uloc.set(0.)
ubc4=  Function(W.sub(1).sub(0).collapse())
with ubc4.vector.localForm() as uloc:
     uloc.set(0.)
ubc5=  Function(W.sub(1).sub(2).collapse())
with ubc5.vector.localForm() as uloc:
     uloc.set(0.)
ubc6=  Function(W.sub(0).sub(0).collapse())
with ubc6.vector.localForm() as uloc:
     uloc.set(0.)
ubc7=  Function(W.sub(1).sub(1).collapse())
with ubc7.vector.localForm() as uloc:
     uloc.set(0.)
ubc8=  Function(W.sub(1).sub(2).collapse())
with ubc8.vector.localForm() as uloc:
     uloc.set(0.)
bcs = [DirichletBC(ubc1, locate_BC1, W.sub(0).sub(1)),
       DirichletBC(ubc2, locate_BC2, W.sub(0).sub(2)),
       DirichletBC(ubc3, locate_BC3, W.sub(0).sub(1)),
       DirichletBC(ubc4, locate_BC4, W.sub(1).sub(0)),
       DirichletBC(ubc5, locate_BC5, W.sub(1).sub(2)),
       DirichletBC(ubc6, locate_BC6, W.sub(0).sub(0)),
       DirichletBC(ubc7, locate_BC7, W.sub(1).sub(1)),
       DirichletBC(ubc8, locate_BC8, W.sub(1).sub(2))
       ]#DirichletBC(u0,locate_fixed_BC)
'''
'''
bcs = [DirichletBC(W.sub(0).sub(1),Constant(mesh,0),"abs(x[0] - 0.0) < 1e-14"),
       DirichletBC(W.sub(0).sub(2),Constant(mesh,0),"abs(x[0] - 0.0) < 1e-14"),
       DirichletBC(W.sub(0).sub(1),Constant(mesh,0),"abs(x[1] - 0.0) < 1e-14"),
       DirichletBC(W.sub(1).sub(0),Constant(mesh,0),"abs(x[1] - 0.0) < 1e-14"),
       DirichletBC(W.sub(1).sub(2),Constant(mesh,0),"abs(x[1] - 0.0) < 1e-14"),
       DirichletBC(W.sub(0).sub(0),Constant(mesh,0),"abs(x[0] - 25.0) < 1e-14"),
       DirichletBC(W.sub(1).sub(1),Constant(mesh,0),"abs(x[0] - 25.0) < 1e-14"),
       DirichletBC(W.sub(1).sub(2),Constant(mesh,0),"abs(x[0] - 25.0) < 1e-14"),
       ]
'''

problem = fem.LinearProblem(J, -F, bcs=bcs, petsc_options={"ksp_type": "gmres", "pc_type": "jacobi"})
#solve(J==-F,w,bcs,petsc_options={"ksp_type":"preonly","pc_type":"lu"})
#solve(F==0,w,bcs,J=J)
w = problem.solve()

# Write it out in a visualizable format for Paraview:
#dolfinx.io.VTKFile("w.pvd").write(w)
#dolfinx.io.VTKFile("u.pvd").write(u_mid)
#dolfinx.io.VTKFile("t.pvd").write(theta)

#file = XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
#file.write_function(w)


# Sample points along an interior line of the domain
point = np.array([100, 100, 50])

# Create boundingboxtree
#tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
#actual_cell = dolfinx.geometry.compute_colliding_cells(tree, mesh,
#                                                       point, 1)

# Evaluate function at points, and create the exact solution
#u_eval = w.sub(0).eval(point, actual_cell)
#print(u_eval)
project(w.sub(0),w_out)



#print("w: ")
w_arr=w.vector.getArray()
#print(w_arr)
np.savetxt("plate_w.txt",w_arr)

#print("sub1: ")
print('u:',w.sub(0).vector.getArray())
print('w0:',w.sub(0).compute_point_values())
#print(u_arr)
#print(u_arr.size)

u_arr_coll = w.sub(0).collapse()
u_arr_coll_vec = u_arr_coll.vector
#print(u_arr_coll_vec.size)
u_arr = u_arr_coll_vec.getArray()
#print(u_arr)
np.savetxt("plate_u.txt",u_arr)

#print("try splitting again:")
dX= u_arr_coll.sub(0).collapse()
dY= u_arr_coll.sub(1).collapse()
dZ= u_arr_coll.sub(2).collapse()
#print(dX)
#print(dY)
#print(dZ)
uX = dX.vector.getArray()
uY = dY.vector.getArray()
uZ = dZ.vector.getArray()
#print("uX:")
#print(uX)
#print(uX.size)
np.savetxt("plate_ux.txt",uX)
#print("uY:")
#print(uY)
#print(uY.size)
np.savetxt("plate_uy.txt",uY)
#print("uZ:")
#print(uZ)
#print(uZ.size)
np.savetxt("plate_uz.txt",uZ)


#print("sub2: ")
t_arr = w.sub(1).vector
#print(t_arr)
#print(t_arr.size)

t_arr_coll = w.sub(1).collapse()
t_arr_coll_vec = t_arr_coll.vector
#print(t_arr_coll_vec.size)
t_arr = t_arr_coll_vec.getArray()
#print(t_arr)
np.savetxt("plate_t.txt",t_arr)

#print("try splitting again:")
tX= t_arr_coll.sub(0).collapse()
tY= t_arr_coll.sub(1).collapse()
tZ= t_arr_coll.sub(2).collapse()
#print(tX)
#print(tY)
#print(tZ)
tX = tX.vector.getArray()
tY = tY.vector.getArray()
tZ = tZ.vector.getArray()
#print("tX:")
#print(tX)
#print(tX.size)
np.savetxt("plate_tx.txt",tX)
#print("tY:")
#print(tY)
#print(tY.size)
np.savetxt("plate_ty.txt",tY)
#print("tZ:")
#print(tZ)
#print(tZ.size)
np.savetxt("plate_tz.txt",tZ)


print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
#print("  Number of dofs     = "+str(W.dim))
#print("  Number of dofs_mio = "+str(FunctionSpace(mesh,VE).dim))
'''
print("index maps: ")
dm0 = W.sub(0).dofmap
dm1 = W.sub(1).dofmap
print(dm0)
print(dm1)
print(dm0.index_map)

print("second way of splitting:")

(uu,pp)=split(w)
print(w)
print(uu)
print(pp)
print(uu[0])

#yy2 = Function(W1)
#yy2.interpolate(uu[0])
#print(yy2)
#print(yy2.vector.array())

#print(uu[0].coefficient)
#print(pp[0].coefficient)
#d1,d2,d3 = uu.split()
#print(d1.vector.getArray())
#print(uu[0].vector.getArray())

dm0 = W.sub(0).dofmap
dm1 = W.sub(1).dofmap
print(dm0)
print(dm1)
print(dm0.index_map)

uu1,pp1 = w.split()
#(uu1,pp1)=w.split(deepcopy=True)
print(uu1)
print(pp1)
print(uu1.vector.getArray())
print(pp1.vector.getArray())
#print(pp1.compute_point_values())


y2 = Function(W1)
y2.interpolate(uu1)
print("interpolate: ")
print(y2.vector.getArray())

#v2d = vertex_to_dof_map(W.sub(0))

#u_vec = w.sub(0).compute_point_values()
#print(u_vec)
'''
# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(w.sub(0))



dolfinx.io.VTKFile("attempt.pvd").write(w.sub(0))   
#File("u.pvd") << u_mid
#File("t.pvd") << theta

