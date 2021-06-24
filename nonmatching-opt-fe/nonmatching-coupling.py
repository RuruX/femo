from dolfin import *
from mshr import *
from petsc4py import PETSc

# Limit quadrature degree:
dx = dx(metadata={"quadrature_degree":2})

# Create two unit square meshes sharing a boundary, but not with their
# elements aligned.
N = 16
mesh_1 = UnitSquareMesh(N,N)
mesh_2 = UnitSquareMesh(N+7,N) # Different number of elements in x-direction
# Translate one of the unit squares down by one:
ALE.move(mesh_2,Constant((0,-1)))

# Hack to create a 1D mesh for the "mortar space" very close to the boundary;
# really it is the full boundary of a rectangle, so the coupling penalty
# is essentially enforced twice.  More ideally, a single polyline mesh
# would be created, but I didn't see an easy way to do this without writing
# extra code to write out an XML mesh file with the custom mesh.  (Using
# UnitIntervalMesh creates a mesh with geometric dimension 1, which is not
# compatible with the 2D meshes it needs to interact with.)
EPS = 1e-10
mesh_m = BoundaryMesh(RectangleMesh(Point(0,-EPS),
                                    Point(1,EPS),10*N,1),"exterior")

# Create spaces on mesh 1, mesh 2, and the mortar mesh:
V1 = FunctionSpace(mesh_1,"CG",1)
V2 = FunctionSpace(mesh_2,"CG",1)
Vm = FunctionSpace(mesh_m,"DG",0)

# This appears to be necessary to initialize PETSc; otherwise the code crashes
# when creating transfer matrices.  Is there a cleaner way to do this?
assemble(TestFunction(Vm)*dx)

# Utility functions to avoid typing "as_backend_type" too many times:
def m2p(A):
    return as_backend_type(A).mat()
def v2p(v):
    return as_backend_type(v).vec()

# These matrices give interpolations of every shape function in the first
# space into the second space, so, if you multiply the vector of coefficients
# of a function in the first space by this matrix, you get its interpolant
# in the second space.
A_1m = PETScDMCollection.create_transfer_matrix(V1,Vm)
A_2m = PETScDMCollection.create_transfer_matrix(V2,Vm)

# Coupling penalty increases as mesh is refined.
k = Constant(1.0e0*N)

u1 = Function(V1)
u2 = Function(V2)
u1.rename("u1","u1")
u2.rename("u2","u2")
u1m = Function(Vm)
u2m = Function(Vm)
v1 = TestFunction(V1)
v2 = TestFunction(V2)

# Penalization of mismatch between solutions on the two meshes, expressed
# as an energy, and differentiated to obtain its contributions to
# subproblem residuals.
penaltyEnergy = 0.5*k*((u1m-u2m)**2)*dx
R1m = derivative(penaltyEnergy,u1m)
R2m = derivative(penaltyEnergy,u2m)

# Function for a repeated patern of matrix multiplications, A^T*R*B, needed
# to get derivatives of mortar terms.
def AT_R_B(A,R,B):
    return ((m2p(A).transposeMatMult(m2p(assemble(R))))
            .matMult(m2p(B)))

# Get linearization of penalty terms:
dR1m_du1m = derivative(R1m,u1m)
dR1m_du2m = derivative(R1m,u2m)
dR2m_du1m = derivative(R2m,u1m)
dR2m_du2m = derivative(R2m,u2m)
dR1_du1 = AT_R_B(A_1m,dR1m_du1m,A_1m)
dR1_du2 = AT_R_B(A_1m,dR1m_du2m,A_2m)
dR2_du1 = AT_R_B(A_2m,dR2m_du1m,A_1m)
dR2_du2 = AT_R_B(A_2m,dR2m_du2m,A_2m)

# Test with manufactured solutions.  Since the SpatialCoordinates of the
# two meshes are different variables, they are passed as arguments to
# Python functions giving the exact solution, source term, residual, etc.
def u_ex(x):
    xp = as_vector([x[0],x[1]-x[0]])
    return sin(pi*xp[0])*sin(pi*xp[1])*(sin(pi*x[0])**2)*(cos(0.5*pi*x[1])**2)
def f(x):
    return -div(grad(u_ex(x))) + u_ex(x)

def pdeRes(u,v,x):
    return inner(grad(u),grad(v))*dx + u*v*dx - f(x)*v*dx

x1 = SpatialCoordinate(mesh_1)
x2 = SpatialCoordinate(mesh_2)

# PDE contributions to subproblem residuals and linearizations:
R1 = v2p(assemble(-pdeRes(u1,v1,x1)))
R2 = v2p(assemble(-pdeRes(u2,v2,x2)))
A11 = m2p(assemble(derivative(pdeRes(u1,v1,x1),u1)))
A22 = m2p(assemble(derivative(pdeRes(u2,v2,x2),u2)))
dR1_du1 += A11
dR2_du2 += A22

# Glue everything together with PETSc nesting functionality:
A = PETSc.Mat()
A.createNest([[dR1_du1,dR1_du2],
              [dR2_du1,dR2_du2]])
A.setUp()
b = PETSc.Vec()
b.createNest([R1,R2])
b.setUp()
u = PETSc.Vec()
u.createNest([v2p(u1.vector()),v2p(u2.vector())])
u.setUp()

# Solve with a PETSc Krylov method:
ksp = PETSc.KSP().create()
# L2 error with default solver settings starts diverging w/ refinement
# after N ~ 100.
ksp.setType(PETSc.KSP.Type.CG)
ksp.setTolerances(rtol=1e-15)
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b,u)
# !!!!!!! Important in parallel !!!!!!!
v2p(u1.vector()).ghostUpdate()
v2p(u2.vector()).ghostUpdate()

# Check error:
e1 = u1-u_ex(x1)
print(sqrt(assemble((e1**2)*dx)))

File("u1.pvd") << u1
File("u2.pvd") << u2

