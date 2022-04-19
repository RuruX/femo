
# Optimal control of the Poisson equation
# =======================================


from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

import moola

n = 16

mesh = UnitSquareMesh(n, n)

#cf = MeshFunction("bool", mesh, mesh.geometric_dimension())
#subdomain = CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
#subdomain.mark(cf, True)
#mesh = refine(mesh, cf)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

# The optimisation algorithm will use the value of the control function 𝑓 as an initial guess for the optimisation.
f = interpolate(Expression("x[0]+2*x[1]", name='Control', degree=1), W)

u = Function(V, name='State')
v = TestFunction(V)

F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)



x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1/(2*pi**2)
d = Expression("d*w", d=d, w=w, degree=3)

alpha = Constant(1e-6)
J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)


rf = ReducedFunctional(J, control)


problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(f)
solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
                        'maxiter': 20,
                        'display': 3,
                        'ncg_hesstol': 0})

sol = solver.solve()
f_opt = sol['control'].data

#plot(f_opt, title="f_opt")
#from matplotlib import pyplot as plt
#plt.show()

# Define the expressions of the analytical solution
f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)

f.assign(f_opt)
print(f_opt.vector().get_local())

solve(F == 0, u, bc)

File('u.pvd') << u
File('f.pvd') << f

control_error = errornorm(f_analytic, f_opt)
state_error = errornorm(u_analytic, u)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)
