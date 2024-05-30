from shell_analysis_fenicsx import *

import basix
import scipy.sparse as sp
import numpy as np
import ufl


class RMShellPDE:
    def __init__(self, mesh):
        self.mesh = mesh
        element_type = "CG2CG1"
        #element_type = "CG2CR1"

        self.element = element = ShellElement(
                        mesh,
                        element_type)
        self.dx_inplane, self.dx_shear = element.dx_inplane, element.dx_shear

        self.W  = self.element.W
        self.VT = FunctionSpace(mesh, ("CG", 1))
        self.VF = VectorFunctionSpace(mesh, ("CG", 1))
        self.bf_sup_sizes = assemble_vector(
                form(TestFunction(self.VF.sub(0).collapse()[0])*dx)).getArray()
        # self.bf_sup_sizes = np.ones_like(self.bf_sup_sizes)

    def compute_alpha(self):
        h_mesh = ufl.CellDiameter(self.mesh)
        V1 = FunctionSpace(self.mesh, ('CG', 1))
        h_mesh_func = Function(V1)
        project(h_mesh, h_mesh_func, lump_mass=False)
        # alpha is a parameter based on the cell area
        alpha = np.average(h_mesh_func.vector.getArray())**2/2
        return alpha

    def pdeRes(self,h,w,f,E,nu,penalty=False, dss=ufl.ds, dSS=ufl.dS, g=None):
        material_model = MaterialModel(E=E,nu=nu,h=h)
        self.elastic_model = elastic_model = ElasticModel(self.mesh,
                                                w,material_model.CLT)
        elastic_energy = elastic_model.elasticEnergy(E, h,
                                    self.dx_inplane,self.dx_shear)
        return elastic_model.weakFormResidual(elastic_energy, f,
                                            penalty=penalty, dss=dss, dSS=dSS, g=g)

    def kinetic_residual(self,rho,h):
        return self.elastic_model.inertialResidual(rho,h)

    def elastic_residual(self,h,w,f,E,nu,penalty=False, dss=ufl.ds, dSS=ufl.dS, g=None):
        return self.pdeRes(h,w,f,E,nu,penalty=False, dss=ufl.ds, dSS=ufl.dS, g=None) \
                + inner(f,self.elastic_model.du_mid)*dx

    def regularization(self, h, type=None):
        alpha1 = Constant(self.mesh, 1e3)
        alpha2 = Constant(self.mesh, 1e0)
        h_mesh = CellDiameter(self.mesh)

        regularization = 0.
        if type=='H1':
            # H1 regularization
            regularization = 0.5*alpha1*dot(grad(h),grad(h))*dx
        elif type=='L2H1':
            # L2 + H1 regularization
            regularization = 0.5*alpha1*inner(h,h)*dx + \
                               0.5*alpha2*h_mesh**2*inner(grad(h),grad(h))*dx
        elif type=='L2':
            # L2 regularization
            regularization = 0.5*alpha1*inner(h,h)*dx
        # No regularization
        return regularization

    def compliance(self,u_mid,f):
        return inner(u_mid,f)*ufl.dx
    
    def tip_disp(self,u_mid,h,dxx):
        return Constant(self.mesh, 0.5)*inner(u_mid,u_mid)*dxx + self.regularization(h)

    def volume(self,h):
        return h*dx

    def mass(self,h,rho):
        return rho*h*dx

    def elastic_energy(self,w,h,E):
        elastic_energy = self.elastic_model.elasticEnergy(E, h,
                                        self.dx_inplane, self.dx_shear)
        return elastic_energy

    def pnorm_stress(self,w,h,E,nu,dx,m=1e-6,rho=100,alpha=None,regularization=False):
        """
        Compute the p-norm of the stress
        `rho` is the Constraint aggregation factor
        """
        shell_stress_RM = ShellStressRM(self.mesh, w, h, E, nu)
        # stress on the top surface
        vm_stress = shell_stress_RM.vonMisesStress(h/2)
        pnorm = (m*vm_stress)**rho*dx
        if regularization:
            regularization = 0.5*Constant(self.mesh, 1e3)*h**rho*dx
            pnorm += regularization
        if alpha == None:
            ##### alpha is a parameter based on the surface area
            alpha_form = Constant(self.mesh,1.0)*dx
            alpha = assemble_scalar(form(alpha_form))
        return 1/alpha*pnorm

    def von_Mises_stress(self,w,h,E,nu,surface='Top'):
        shell_stress_RM = ShellStressRM(self.mesh, w, h, E, nu)
        if surface == 'Top':
            # stress on the top surface
            vm_stress = shell_stress_RM.vonMisesStress(h/2)
        elif surface == 'Mid':
            # stress on the mid surface
            vm_stress = shell_stress_RM.vonMisesStress(0.)
        elif surface == 'Bot':
            # stress on the bottom surface
            vm_stress = shell_stress_RM.vonMisesStress(-h/2)
        else:
            TypeError("Unsupported surface type for stress computation.")
        return vm_stress

    def projected_von_Mises_stress(self, vm_stress):
        von_Mises_func = Function(self.VT)
        project(vm_stress, von_Mises_func, lump_mass=False)
        return von_Mises_func

    def compute_nodal_disp(self,func):
        return computeNodalDisp(func)

    def construct_nodal_disp_map(self):
        deriv_us_to_ua_coord_list = []
        Q_map = self.construct_CG2_CG1_interpolation_map()
        disp_extraction_mats = self.construct_disp_extraction_mats()
        for i in range(3):
            deriv_us_to_ua_coord_list += [sp.csr_matrix(
                                            Q_map@disp_extraction_mats[i])]
        disp_extraction_mats = sp.vstack(deriv_us_to_ua_coord_list)
        # print(disp_extraction_mats.shape)
        return disp_extraction_mats

    def compute_sparse_mass_matrix(self):
        # functions used to assemble FEA mass matrix
        f_trial = TrialFunction(self.VT)
        f_test = TestFunction(self.VT)

        # assemble PETSc mass matrix
        Mat_f = assemble_matrix(form(inner(f_test, f_trial)*dx))
        Mat_f.assemble()

        # convert mass matrix to sparse Python array
        Mat_f_csr = Mat_f.getValuesCSR()
        Mat_f_sp = sp.csr_matrix((Mat_f_csr[2], Mat_f_csr[1], Mat_f_csr[0]))

        # eliminate zeros that are present in mass matrix
        Mat_f_sp.eliminate_zeros()
        return Mat_f_sp

    def construct_disp_extraction_mats(self):
        # first we construct the extraction matrix for all displacements
        disp_space, solid_disp_idxs = self.W.sub(0).collapse()
        num_disp_dofs = len(solid_disp_idxs)
        solid_rot_idxs = self.W.sub(1).collapse()[1]
        num_rot_dofs = len(solid_rot_idxs)
        # __init__ sparse mapping matrix
        disp_mat = sp.lil_matrix((num_disp_dofs, num_disp_dofs+num_rot_dofs))
        # set relevant entries to 1
        disp_mat[list(range(num_disp_dofs)), solid_disp_idxs] = 1
        # convert sparse matrix to CSR format (for faster matrix-vector products)
        disp_mat.tocsr()

        # afterwards we generate the extraction matrices for the 3 displacement components
        disp_component_extraction_mats = []
        for i in range(3):
            solid_disp_coord_idxs = disp_space.sub(i).collapse()[1]
            num_disp_coord_dofs = len(solid_disp_coord_idxs)
            # __init__ sparse mapping matrix
            disp_coord_mat = sp.lil_matrix((num_disp_coord_dofs, disp_mat.shape[0]))
            # set relevant entries to 1
            disp_coord_mat[list(range(num_disp_coord_dofs)), solid_disp_coord_idxs] = 1
            # convert sparse matrix to CSR format (for faster matrix-vector products)
            disp_coord_mat.tocsr()

            # we multiply each coordinate extraction matrix with the displacement extraction matrix
            disp_coord_mat = disp_coord_mat@disp_mat

            disp_component_extraction_mats += [disp_coord_mat]

        return disp_component_extraction_mats

    def construct_CG2_CG1_interpolation_map(self):
        CG2_space = self.W.sub(0).collapse()[0]
        phys_coord_array = self.mesh.geometry.x
        mesh_bbt = dolfinx.geometry.BoundingBoxTree(self.mesh,
                                                    self.mesh.topology.dim)

        # create basix element
        ct = basix.cell.string_to_type(self.mesh.topology.cell_type.name)
        c_element = basix.create_element(basix.ElementFamily.P, ct, 2,
                                        basix.LagrangeVariant.equispaced)
        num_cg2_dofs = CG2_space.tabulate_dof_coordinates().shape[0]

        for i in range(self.mesh.topology.index_map(0).size_local):
            x_point = np.array(phys_coord_array[i, :])
            x_point_eval = self.eval_fe_basis_all_dolfinx(x_point,
                                CG2_space.dofmap.list, mesh_bbt, c_element, i,
                                mat_shape=(self.mesh.geometry.x.shape[0], num_cg2_dofs))
            if i == 0:
                sample_mat = x_point_eval
            else:
                sample_mat += x_point_eval

        return sample_mat

    def eval_fe_basis_all_dolfinx(self, x, dofmap_adjacencylist, mesh_bbt, basix_element, x_idx, mat_shape):
        mesh_cur = self.mesh
        cell_candidates = geometry.compute_collisions(mesh_bbt, x)
        cell_ids = geometry.compute_colliding_cells(mesh_cur, cell_candidates, x)
        geom_dofs = dofmap_adjacencylist.links(cell_ids[0])
        cell_vertex_ids = mesh_cur.geometry.dofmap.links(cell_ids[0])
        x_ref = mesh_cur.geometry.cmap.pull_back(x[None, :], mesh_cur.geometry.x[cell_vertex_ids])

        c_tab = basix_element.tabulate(0, x_ref)[0, 0, :, 0]
        # NOTE: c_tab contains the DoF values on the cell cell_ids[0]; geom_dofs contains their global DoF numbers

        # basis_vec = np.zeros((mesh_cur.geometry.dofmap.num_nodes,))
        basis_vec = sp.csr_array((c_tab, (c_tab.shape[0]*[int(x_idx)], geom_dofs)), shape=mat_shape)

        return basis_vec

class RadialBasisFunctions:
    def Gaussian(x_dist, eps=1.):
        return np.exp(-(eps*x_dist)**2)

    def BumpFunction(x_dist, eps=1.):
        # filter x_dist to get rid of x_dist >= 1/eps, this prevents overflow warnings
        x_dist_filt = np.where(x_dist < 1/eps, x_dist, 0.)
        f_mat = np.where(x_dist < 1/eps, np.exp(-1./(1.-(eps*x_dist_filt)**2)), 0.)
        return f_mat/np.exp(-1)  # normalize output so x_dist = 0 corresponds to f = 1

    def ThinPlateSpline(x_dist):
        return np.multiply(np.power(x_dist, 2), np.log(x_dist))

class NodalMap:
    def __init__(self, solid_nodal_mesh, fluid_nodal_mesh, RBF_width_par=np.inf, RBF_func=RadialBasisFunctions.Gaussian, column_scaling_vec=None):
        self.solid_nodal_mesh = solid_nodal_mesh
        self.fluid_nodal_mesh = fluid_nodal_mesh
        self.RBF_width_par = RBF_width_par
        self.RBF_func = RBF_func
        if column_scaling_vec is not None:
            self.column_scaling_vec = column_scaling_vec
        else:
            self.column_scaling_vec = np.ones((solid_nodal_mesh.shape[0],))

        self.map_shape = [self.fluid_nodal_mesh.shape[0], self.solid_nodal_mesh.shape[0]]
        self.distance_matrix = self.compute_distance_matrix()
        self.map = self.construct_map()

    def compute_distance_matrix(self):
        coord_dist_mat = np.zeros((self.map_shape + [3]))
        for i in range(3):
            coord_dist_mat[:, :, i] = NodalMap.coord_diff(self.fluid_nodal_mesh[:, i], self.solid_nodal_mesh[:, i])

        coord_dist_mat = NodalMap.compute_pairwise_Euclidean_distance(coord_dist_mat)
        return coord_dist_mat

    def construct_map(self):
        influence_coefficients = self.RBF_func(self.distance_matrix, eps=self.RBF_width_par)

        # influence_coefficients = np.multiply(influence_coefficients, influence_dist_below_max_mask)

        # include influence of column scaling
        influence_coefficients = np.einsum('ij,j->ij', influence_coefficients, self.column_scaling_vec)
        # print("min influence:", np.min(influence_coefficients))
        # print("max influence:", np.max(influence_coefficients))
        # exit()
        influence_coefficients[influence_coefficients < 1e-16] = 0.
        # TODO: Make influence_coefficients matrix sparse before summation below?
        #       -> seems like the matrix sparsity depends heavily on the value of self.RBF_par_width,
        #           probably not worthwhile to make the matrix sparse

        # sum influence coefficients in each row and normalize the coefficients with those row sums
        inf_coeffs_per_row = np.sum(influence_coefficients, axis=1)
        normalized_inf_coeff_map = np.divide(influence_coefficients, inf_coeffs_per_row[:, None])
        return normalized_inf_coeff_map

    @staticmethod
    def coord_diff(arr_1, arr_2):
        # subtracts arr_1 and arr_2 of different sizes in such a way that the result is a matrix of size (arr_1.shape[0], arr_2.shape[0])
        return np.subtract(arr_1[:, None], arr_2[None, :])

    @staticmethod
    def compute_pairwise_Euclidean_distance(coord_dist_mat):
        coord_dist_mat_sqrd = np.power(coord_dist_mat, 2)
        coord_dist_mat_summed = np.sum(coord_dist_mat_sqrd, axis=2)
        return np.sqrt(coord_dist_mat_summed)