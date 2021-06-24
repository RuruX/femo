"""
Brute force solution of the inverse problem of determining the optimal
thickness profile for a cantilevered beam to minimize tip deflection.
"""

from dolfin import *
import ufl

import numpy as np 

Nel = 10
k = 1
mesh = UnitIntervalMesh(Nel)
n = FacetNormal(mesh)
h = CellDiameter(mesh)

UE = FiniteElement("CG",mesh.ufl_cell(),k) # Displacement u
VE = FiniteElement("CG",mesh.ufl_cell(),k) # EI*u''
#TE = FiniteElement("CG",mesh.ufl_cell(),k) # Thickness
TE = FiniteElement("DG",mesh.ufl_cell(),0) # Thickness
LE = FiniteElement("R",mesh.ufl_cell(),0) # Global constant multiplier

W = FunctionSpace(mesh,MixedElement([UE,VE, # PDE system solution
                                     UE,VE, # Multiplier for PDE constraint
                                     TE, # Thickness
                                     LE])) # Multiplier for mass constraint

# Characteristic functions for left and right end of the mesh; slightly
# easier than flagging facets (which is the "correct" approach):
x = SpatialCoordinate(mesh)
leftChar = 1.0-x[0]
rightChar = x[0]

# PDE residual; BC enforcement not immediately obvious!
def pdeRes(u,v,du,dv,t):
    EI = t ** 3
    q = rightChar*Constant(-1.0) # Self-weight
    #q = Constant(-1.0) # Uniform
    return inner(grad(u),grad(du))*dx + inner(v/EI,du)*dx \
        + inner(grad(v),grad(dv))*dx + inner(q,dv)*dx \
        + leftChar*u*inner(n,grad(dv))*ds \
        + rightChar*v*inner(n,grad(du))*ds \
        - leftChar*dot(grad(v),n)*dv*ds \
        - rightChar*dot(grad(u),n)*du*ds

# Objective function:
def J(u):
    return Constant(0.1)*rightChar*0.5*u*u*ds

# Constrain integral of thickness over the beam to be equal to a constant:
def constraint(t,lam):
    return lam*(t-Constant(1.0))*dx

# Regularization for thickness; if this is too small the nonlinear solve
# diverges.
alpha = Constant(1e-2)

def regularization(t):
    # Tikhonov regularization: Induces artificial BCs
    #return 0.5*alpha*inner(grad(t),grad(t))*dx
    
    # L2 regularization
    return 0.5*alpha*inner(t,t)*dx

    # L2 + jumps
    #return 0.5*alpha*(inner(t,t)*dx + avg(h)*(jump(t)**2)*dS)

# PDE-constrained Lagrangian for minimizing tip-deflection squared:
def L(u,v,du,dv,t,lam):
    return J(u) + constraint(t,lam) + pdeRes(u,v,du,dv,t) + regularization(t)

w = Function(W)
u,v,du,dv,t,lam = split(w)


# Initialize thickness to uniform positive guess:
w.interpolate(Constant((0,0,0,0,1,0)))

# TE = FunctionSpace(mesh,"DG",0)
# t = Function(TE)
# t.vector().set_local(np.ones(Nel))

# Solve for everything:
res = derivative(L(u,v,du,dv,t,lam),w)

"""
    why have twice derivative here?
    why is res a vector rather than a matrix?
        AttributeError: 'dolfin.cpp.la.PETScVector' object has no attribute 'mat'

"""

## Convert ufl form into numpy array
#dL_dw = assemble(derivative(pdeRes(u,v,du,dv,t), w))
#dL_dw_sparse = as_backend_type(dL_dw).mat()
#
#from scipy.sparse import csr_matrix
#
#dL_dw_csr = csr_matrix(dL_dw_sparse.getValuesCSR()[::-1], shape=dL_dw_sparse.size)
## As a CSR Sparse Matrix
#rows,cols  =  dL_dw_csr.nonzero()
#data =  dL_dw_csr[rows,cols]
#dL_dw_csr.toarray()


Dres = derivative(res,w)

# Convert ufl form into numpy array
dL_dw = assemble(Dres)
dL_dw_sparse = as_backend_type(dL_dw).mat()

from scipy.sparse import csr_matrix

dL_dw_csr = csr_matrix(dL_dw_sparse.getValuesCSR()[::-1], shape=dL_dw_sparse.size)
# As a CSR Sparse Matrix
rows,cols  =  dL_dw_csr.nonzero()
data =  dL_dw_csr[rows,cols]
print(rows)
print(cols)
print(data)



problem = NonlinearVariationalProblem(res,w,bcs=[],J=Dres)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'snes' 
solver.parameters['snes_solver']['line_search'] = 'bt'
solver.parameters['snes_solver']['linear_solver'] = 'lu'
solver.solve()

# Check objective function:
print("Value of objective function = " + str(assemble(J(u))))

# Check mass constraint:
print("Integral of t = " + str(assemble(t*dx)))


# Visual output:
from matplotlib import pyplot as plt

# Plot displacement:
# plt.figure(1)
# plt.title('u')
# plot(u)

# plt.figure(2)
# plt.title('v')
# plot(v)

# print('u values:', project(u,FunctionSpace(mesh,"DG",0)).vector().get_local())
# print('v values:', project(v,FunctionSpace(mesh,"DG",0)).vector().get_local())

# Plot thickness profile:
plt.figure(3)
plot(project(t,FunctionSpace(mesh,"DG",0)))
plt.show()
