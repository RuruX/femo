from dolfin import *

n = 2
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'CG', 1)
VF = FunctionSpace(mesh, 'DG', 0)

u = Function(V)
v = TestFunction(V)

# Define the source term
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
alpha = Constant(1e-6)
f_analytic = Expression("1/(1+alpha*4*pow(pi,4))*w", w=w, alpha=alpha, degree=3)
f = interpolate(f_analytic, VF)

# Apply zero boundary condition on the outer boundary
bc = DirichletBC(V, Constant(1.0), "on_boundary")

# Variational form of Poisson's equation
res = (inner(grad(u),grad(v))-f*v)*dx

# Quantity of interest: will be used as the Jacobian in adjoint method
dRdu = derivative(res, u)

# Option 1: assemble A and b seperately
A = assemble(dRdu)
b = assemble(res)
bc.apply(A,b)

# Option 2: assemble the system
A_,b_ = assemble_system(dRdu, res, bcs=bc)

def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()

print(" ------ Matrix A by old dolfin - assemble ------- ")
print(convertToDense(as_backend_type(A).mat()))
print(b.get_local())
print(" ------ Matrix A by old dolfin - assemble_system ------- ")
print(convertToDense(as_backend_type(A_).mat()))
print(b_.get_local())