"""
The "nonmatching" module:
-------------------------
This module provides functionality for solving coupling of nonmatcing 
problem in spline space using the "tIGAr" library with multiple 2D 
spline patches in 3D space. B-Spline patches are created using "igakit"
or "tIGAr".
"""
from dolfin import *
from mshr import *
from petsc4py import PETSc
import matplotlib.pyplot as plt


def compute_rate(x,y):
	return (y[1]-y[0])/(x[1]-x[0])

def m2p(A):
	return as_backend_type(A).mat()

def v2p(v):
	return as_backend_type(v).vec()

def AT_R_B(A,R,B):
	"""
	Compute A^T*R*B.
	"""
	return (m2p(A).transposeMatMult(m2p(R)).matMult(m2p(B)))

def RT_AT_B(A,R,B):
	"""
	Compute (R^T*A)^T*B.
	"""
	return (R.transposeMatMult(m2p(A)).transposeMatMult(m2p(B)))


def zero_PETSc_v(num_el):
	"""
	Create zeros PETSc vector with size num_el.
	"""
	v = PETSc.Vec().create()
	v.setSizes(num_el)
	v.setType('seq')
	v.setUp()
	v.assemble()
	return v

def zero_PETSc_M(row, col):
	"""
	Create zeros PETSc matrix with shape (row, col).
	"""
	A = PETSc.Mat().create()
	A.setSizes([row, col])
	A.setType('seqaij')
	A.setUp()
	A.assemble()
	return A

class nonmatching_FE(object):
	"""
	This class is used to solve the coupling of nonmatching problem by pure finite element method.
	"""
	def __init__(self, pts_list, num_el_list, p_list):

		self.dx = dx(metadata={"quadrature_degree":2})
		self.num_field = 1
		self.EPS = 1e-10
		self.num_spline = num_spline # The number of B-Spline patches, int
		self.pts_list = pts_list # Points for each patches, list, each element has 4 points
		self.num_el_list = num_el_list # Number of elements for each patch, list

		self.mortar_mesh = [] # The list contains mortar meshes, it has "num_mortar_mesh" elements

		self.V1 = [] # Function spaces for B-Spline patches
		self.Vm = [] # Function spaces for mortar mesh

		self.u = [] # Function for B-Spline patches
		self.v = [] # Test function for B-Spline patches
		self.um = [] # Function for mortar mesh, includes 'num_mortar_mesh'*2 elements


	def createMortarMesh(self, num_mortar_mesh, mortar_pts, mortar_nels):
		"""
		Create mortar meshes that used to build transfer matrix and corresponding function. 
		"mortar_pts" is a list that contains the points of location for each mortar mesh, it has
		"num_mortar_mesh" elements, each elements contains 4 numbers which are the lication of 
		the mortar mesh.
		"""
		self.num_mortar_mesh = num_mortar_mesh # The number of mortar meshes
		for i in range(self.num_mortar_mesh):
			mesh_m = BoundaryMesh(RectangleMesh(Point(mortar_pts[i][0], mortar_pts[i][1]),\
					 Point(mortar_pts[i][2], mortar_pts[i][3]),mortar_nels[i][0],\
					 mortar_nels[i][1]),"exterior")
			self.mortar_mesh.append(mesh_m)
			self.Vm.append(FunctionSpace(mesh_m, FiniteElement('DG', mesh_m.ufl_cell(), 0)))
			self.um.append([Function(self.Vm[i]), Function(self.Vm[i])])

	def transferMatrix(self, mortar_mesh_ind, move_mortar_mesh):
		"""
		Build transfer matrices that transfer data from B-Spline patches
		to mortar mesh. "move_mortar_mesh" is a list that includes "num_mortar_mesh"
		elements (list), each element contains the 2 sets (dictionary) of movements 
		that can produe 2 transfer matrices.
		"""
		# Indeices that marked which 2 B-Spline patches the mortar connected 
		self.mortar_mesh_view = []
		self.mortar_mesh_ind = mortar_mesh_ind
		self.transfer_matrix = [] # Transfer matrices for all mortar meshes
		for i in range(self.num_mortar_mesh):
			ind = self.mortar_mesh_ind[i]
			transfer_matrix_ = [] # 2 transfer matrices for each mortar mesh
			mortar_mesh_view_ = []
			mortar_mesh_view_.append(Mesh(self.mortar_mesh[i]))

			for j in range(len(move_mortar_mesh[i])): # j is the matrix index, 1 or 2
				for k in range(len(move_mortar_mesh[i][j])): # k is the step number
					move_type = list(move_mortar_mesh[i][j][k].keys())[0]
					vals = move_mortar_mesh[i][j][k][move_type]

					if move_type == 'translate':
						ALE.move(self.mortar_mesh[i], Constant(vals[0]))
					elif move_type == 'rotate':
						self.mortar_mesh[i].rotate(vals[0], vals[1],Point(vals[2]))
					elif move_type == 'stay':
						pass
					else:
						print("Undefined movement \"{}\" for mortar mesh {}!".format(move_type, i))

				mortar_mesh_view_.append(Mesh(self.mortar_mesh[i]))			
				A_m = PETScDMCollection.create_transfer_matrix(self.V[ind[j]], self.Vm[i])
				transfer_matrix_.append(A_m)
	
			self.mortar_mesh_view.append(mortar_mesh_view_)
			self.transfer_matrix.append(transfer_matrix_)

	def penaltyEnergy(self):
		"""
		Penalize energy: "(u1m - u2m)**2*dx".
		"""
		# The list includes penaly energy for each mortar mesh. It has 
		self.PE = []
		k = self.num_el_list[0]*1e1
		for i in range(self.num_mortar_mesh):
			self.PE.append(0.5*k*((self.um[i][0] - self.um[i][1])**2)*self.dx)

	def problemSetUp(self, pdeRes, f_control):
		"""
		Set up the nonmatching problem and pose it in spline space. "pdeRes" is function
		that computes the residual of the weak form.
		"""
		self.x = []
		self.Ri = []
		self.Aii = []
		num_rows = []
		num_cols = []
		for i in range(self.num_spline):
			# x_.append(self.spline_list[ind[j]].parametricCoordinates())
			self.x.append(self.spline_list[i].spatialCoordinates())
			pde_residual = pdeRes(self.u[i],self.v[i],self.x[i],\
				self.spline_list[i],f_control[i])
			self.Ri.append(v2p(self.spline_list[i].assembleVector(-pde_residual)))
			self.Aii.append(m2p(self.spline_list[i].assembleMatrix(\
				derivative(pde_residual, self.u[i]))))
			num_rows.append(self.Aii[i].size[0])
			num_cols.append(self.Aii[i].size[1])

		self.A_nest = []
		self.R_nest = []
		self.u_iga = []
		self.u_nest = []
		for i in range(self.num_spline):
			A_nest_ = []
			self.R_nest.append(zero_PETSc_v(num_rows[i]))
			self.u_iga.append(multTranspose(self.spline_list[i].M, self.u[i].vector()))
			self.u_nest.append(v2p(self.u_iga[i]))
			for j in range(self.num_spline):
				A_nest_.append(zero_PETSc_M(num_rows[i], num_cols[j]))
			self.A_nest.append(A_nest_)

		self.penaltyEnergy()
		self.Rm = []
		self.dRm_dum = []
		self.dR_du = []
		self.dR_du_iga = []
		for i in range(self.num_mortar_mesh):
			ind = self.mortar_mesh_ind[i]
			Rm_ = [] # Derivatives of panelty energy for mortar mesh i
			for j in range(len(ind)):
				Rm_.append(derivative(self.PE[i], self.um[i][j]))
			self.Rm.append(Rm_)

			dRm_dum_ = []
			dR_du_ = []
			dR_du_iga_ = []
			for j in range(len(ind)):
				dRm_dum__ = []
				dR_du__ = []
				dR_du_iga__ = []
				for k in range(len(ind)):
					dRm_dum__.append(derivative(self.Rm[i][j], self.um[i][k]))
					dR_du__.append(AT_R_B(self.transfer_matrix[i][j], \
						assemble(dRm_dum__[k]), self.transfer_matrix[i][k]))
					dR_du_iga__.append(RT_AT_B(self.spline_list[ind[j]].M, \
						dR_du__[k], self.spline_list[ind[k]].M))

					self.A_nest[ind[j]][ind[k]] += dR_du_iga__[k]
				self.A_nest[ind[j]][ind[j]] += self.Aii[ind[j]]
				self.R_nest[ind[j]] += self.Ri[ind[j]]

				dRm_dum_.append(dRm_dum__)
				dR_du_.append(dR_du__)
				dR_du_iga_.append(dR_du_iga__)

			self.dRm_dum.append(dRm_dum_)
			self.dR_du.append(dR_du_)
			self.dR_du_iga.append(dR_du_iga_)

	def solveNonmatching(self,rtol=1e-15):
		"""
		Solve the coupling of nonmatching problem in spline space.
		Only works for 2 patches now.
		"""
		# Create PETSc nest
		self.A_ksp = PETSc.Mat()
		self.A_ksp.createNest(self.A_nest)
		self.A_ksp.setUp()

		self.b_ksp = PETSc.Vec()
		self.b_ksp.createNest(self.R_nest)
		self.b_ksp.setUp()

		self.u_ksp = PETSc.Vec()
		self.u_ksp.createNest(self.u_nest)
		self.u_ksp.setUp()

		# Slove the system
		ksp = PETSc.KSP().create()
		ksp.setType(PETSc.KSP.Type.CG)
		ksp.setTolerances(rtol=rtol)
		ksp.setOperators(self.A_ksp)
		ksp.setFromOptions()
		ksp.solve(self.b_ksp,self.u_ksp)

		# Convert u in IGA DoFs to FE DoFs
		for i in range(self.num_spline):
			IGAtoFE(self.u_iga[i], self.u[i], self.spline_list[i].M)

	def L2Error(self, u_ex):
		"""
		Compute L2 error for each spline
		"""
		self.L2e = []
		for i in range(self.num_spline):
			x = self.spline_list[i].spatialCoordinates()
			L2e_ = sqrt(assemble(((self.u[i]-u_ex(x))**2)*self.spline_list[i].dx))
			self.L2e.append(L2e_)

	def saveResults(self):
		"""
		Save computed data and geometric mapping of B-Spline patches
		"""
		for i in range(self.num_spline):
			self.u[i].rename("u{}".format(i),"u{}".format(i))
			File("results/u{}.pvd".format(i)) << self.u[i]
			for j in range(4):
				self.spline_list[i].cpFuncs[j].rename("s{}F{}".\
					format(i,j),"s{}F{}".format(i,j))
				File("results/s{}F{}.pvd".format(i,j)) << \
					self.spline_list[i].cpFuncs[j]

	def viewMortarMesh(self):
		"""
		View the movement of mortar meshes when building transfer matrices.
		"""
		for i in range(self.num_mortar_mesh):
			plt.figure()
			for j in range(len(self.mortar_mesh_view[i])):
				plot(self.mortar_mesh_view[i][j], label='Step {}'.format(j))
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('Movement of mortar mesh {}'.format(i))
			plt.legend()
		plt.show()

	def viewSpline(self):
		"""
		View B-Spline patches using igakit plot tools.
		"""
		igaplt.plt.figure()
		for i in range(self.num_spline):
			igaplt.plt.plot(self.srf_list[i])
		igaplt.plt.xlabel('x')
		igaplt.plt.ylabel('y')
		igaplt.plt.zlabel('z')
		igaplt.plt.title('B-Spline patches in physical domain')
		plt.show()


	###### To view results in physical domain using ParaView ######
	# ((s0F0)/s0F3-coordsX)*iHat + ((s0F1)/s0F3-coordsY)*jHat + ((s0F2)/s0F3-coordsZ)*kHat
	# ((s1F0)/s1F3-coordsX)*iHat + ((s1F1)/s1F3-coordsY)*jHat + ((s1F2)/s1F3-coordsZ)*kHat
	# ((s2F0)/s2F3-coordsX)*iHat + ((s2F1)/s2F3-coordsY)*jHat + ((s2F2)/s2F3-coordsZ)*kHat
	# ((s3F0)/s3F3-coordsX)*iHat + ((s3F1)/s3F3-coordsY)*jHat + ((s3F2)/s3F3-coordsZ)*kHat
