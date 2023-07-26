from femo.fea.fea_dolfinx import FEA
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel, OutputFieldModel
from shell_analysis_fenicsx import *
from shell_analysis_fenicsx.read_properties import readCLT, sortIndex
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import basix
import scipy.sparse as sp
import csdl
import numpy as np

class ShellResidualModule(ModuleCSDL):
    '''
    Dynamic shell model

    Output:
    - residual
    '''
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties

    def define(self):
        pde = self.parameters['pde']
        shell_mesh = pde.mesh
        shells = self.parameters['shells']
        shell_name = list(shells.keys())[0]   # this is only taking the first mesh added to the solver.

        E = shells[shell_name]['E']
        nu = shells[shell_name]['nu']
        rho = shells[shell_name]['rho']
        dss = shells[shell_name]['dss']
        dSS = shells[shell_name]['dSS']
        dxx = shells[shell_name]['dxx']
        g = shells[shell_name]['g']

        PENALTY_BC = True


        fea = FEA(shell_mesh)
        fea.PDE_SOLVER = "Newton"
        fea.initialize = True
        fea.linear_problem = True
        # Add input to the PDE problem:
        input_name_1 = shell_name+'_thicknesses'
        input_function_space_1 = pde.VT
        # input_function_space_1 = FunctionSpace(shell_mesh, ("DG", 0))
        input_function_1 = Function(input_function_space_1)
        # Add input to the PDE problem:
        input_name_2 = 'F_solid'
        input_function_space_2 = pde.VF
        input_function_2 = Function(input_function_space_2)

        # Add state to the PDE problem:
        state_name = 'disp_solid'
        state_function_space = pde.W
        state_function = Function(state_function_space)


        # Simple isotropic material
        residual_form = pde.pdeRes(input_function_1,state_function,
                                input_function_2,E,nu,
                                penalty=PENALTY_BC, dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        output_name_1 = 'compliance'
        u_mid, theta = ufl.split(state_function)
        output_form_1 = pde.compliance(u_mid,input_function_1, dxx)
        output_name_2 = 'mass'
        output_form_2 = pde.mass(input_function_1, rho)
        output_name_3 = 'elastic_energy'
        output_form_3 = pde.elastic_energy(state_function,input_function_1,E)
        output_name_4 = 'pnorm_stress'
        m, rho = 1e-6, 100
        dx_reduced = ufl.Measure("dx", domain=shell_mesh, metadata={"quadrature_degree":4})
        output_form_4 = pde.pnorm_stress(state_function,input_function_1,E,nu,
                                dx_reduced,m=m,rho=rho,alpha=None,regularization=False)
        output_name_5 = 'von_Mises_stress'
        output_form_5 = pde.von_Mises_stress(state_function,input_function_1,E,nu,surface='Top')
        fea.add_input(input_name_1, input_function_1, init_val=0.001, record=True)
        fea.add_input(input_name_2, input_function_2, record=False)
        fea.add_state(name=state_name,
                        function=state_function,
                        residual_form=residual_form,
                        arguments=[input_name_1, input_name_2])
        fea.add_output(name=output_name_1,
                        type='scalar',
                        form=output_form_1,
                        arguments=[state_name,input_name_1])
        fea.add_output(name=output_name_2,
                        type='scalar',
                        form=output_form_2,
                        arguments=[input_name_1])
        fea.add_output(name=output_name_3,
                        type='scalar',
                        form=output_form_3,
                        arguments=[input_name_1,state_name])
        fea.add_output(name=output_name_4,
                        type='scalar',
                        form=output_form_4,
                        arguments=[input_name_1,state_name])
        fea.add_field_output(name=output_name_5,
                        form=output_form_5,
                        arguments=[input_name_1,state_name],
                        record=True)
        force_reshaping_model = ForceReshapingModel(pde=pde,
                                    input_name=shell_name+'_forces',
                                    output_name=input_name_2)
        solid_model = StateModel(fea=fea,
                                debug_mode=False,
                                state_name=state_name,
                                arg_name_list=fea.states_dict[state_name]['arguments'])
        compliance_model = OutputModel(fea=fea,
                                    output_name=output_name_1,
                                    arg_name_list=fea.outputs_dict[output_name_1]['arguments'])
        mass_model = OutputModel(fea=fea,
                                    output_name=output_name_2,
                                    arg_name_list=fea.outputs_dict[output_name_2]['arguments'])
        elastic_energy_model = OutputModel(fea=fea,
                                    output_name=output_name_3,
                                    arg_name_list=fea.outputs_dict[output_name_3]['arguments'])
        pnorm_stress_model = OutputModel(fea=fea,
                                    output_name=output_name_4,
                                    arg_name_list=fea.outputs_dict[output_name_4]['arguments'])
        von_Mises_stress_model = OutputFieldModel(fea=fea,
                                    output_name=output_name_5,
                                    arg_name_list=fea.outputs_field_dict[output_name_5]['arguments'])

        disp_extraction_model = DisplacementExtractionModel(pde=pde,
                                    input_name=state_name,
                                    output_name=shell_name+'_displacement')
        aggregated_stress_model = AggregatedStressModel(m=m, rho=rho,
                                    input_name=output_name_4,
                                    output_name=shell_name+'_aggregated_stress')

        self.add(force_reshaping_model, name='force_reshaping_model')
        self.add(solid_model, name='solid_model')
        self.add(disp_extraction_model, name='disp_extraction_model')
        self.add(compliance_model, name='compliance_model')
        self.add(von_Mises_stress_model, name='von_Mises_stress_model')
        self.add(mass_model, name='mass_model')
        self.add(elastic_energy_model, name='elastic_energy_model')
        self.add(pnorm_stress_model, name='von_mises_stress_model')
        self.add(aggregated_stress_model, name='aggregated_stress_model')





class ShellModule(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties

    def define(self):
        pde = self.parameters['pde']
        shell_mesh = pde.mesh
        shells = self.parameters['shells']
        shell_name = list(shells.keys())[0]   # this is only taking the first mesh added to the solver.

        E = shells[shell_name]['E']
        nu = shells[shell_name]['nu']
        rho = shells[shell_name]['rho']
        dss = shells[shell_name]['dss']
        dSS = shells[shell_name]['dSS']
        dxx = shells[shell_name]['dxx']
        g = shells[shell_name]['g']

        PENALTY_BC = True


        fea = FEA(shell_mesh)
        fea.PDE_SOLVER = "Newton"
        fea.REPORT = False
        fea.initialize = True
        fea.linear_problem = True
        # Add input to the PDE problem:
        input_name_1 = shell_name+'_thicknesses'
        input_function_space_1 = pde.VT
        # input_function_space_1 = FunctionSpace(shell_mesh, ("DG", 0))
        input_function_1 = Function(input_function_space_1)
        # Add input to the PDE problem:
        input_name_2 = 'F_solid'
        input_function_space_2 = pde.VF
        input_function_2 = Function(input_function_space_2)

        # Add state to the PDE problem:
        state_name = 'disp_solid'
        state_function_space = pde.W
        state_function = Function(state_function_space)


        # Simple isotropic material
        residual_form = pde.pdeRes(input_function_1,state_function,
                                input_function_2,E,nu,
                                penalty=PENALTY_BC, dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        output_name_1 = 'compliance'
        u_mid, theta = ufl.split(state_function)
        output_form_1 = pde.compliance(u_mid,input_function_1, dxx)
        output_name_2 = 'mass'
        output_form_2 = pde.mass(input_function_1, rho)
        output_name_3 = 'elastic_energy'
        output_form_3 = pde.elastic_energy(state_function,input_function_1,E)
        output_name_4 = 'pnorm_stress'
        m, rho = 1e-6, 100
        dx_reduced = ufl.Measure("dx", domain=shell_mesh, metadata={"quadrature_degree":4})
        output_form_4 = pde.pnorm_stress(state_function,input_function_1,E,nu,
                                dx_reduced,m=m,rho=rho,alpha=None,regularization=False)
        output_name_5 = 'von_Mises_stress'
        output_form_5 = pde.von_Mises_stress(state_function,input_function_1,E,nu,surface='Top')

        fea.add_input(input_name_1, input_function_1, init_val=0.001, record=True)
        fea.add_input(input_name_2, input_function_2, record=True)
        fea.add_state(name=state_name,
                        function=state_function,
                        residual_form=residual_form,
                        arguments=[input_name_1, input_name_2])
        fea.add_output(name=output_name_1,
                        type='scalar',
                        form=output_form_1,
                        arguments=[state_name,input_name_1])
        fea.add_output(name=output_name_2,
                        type='scalar',
                        form=output_form_2,
                        arguments=[input_name_1])
        fea.add_output(name=output_name_3,
                        type='scalar',
                        form=output_form_3,
                        arguments=[input_name_1,state_name])
        fea.add_output(name=output_name_4,
                        type='scalar',
                        form=output_form_4,
                        arguments=[input_name_1,state_name])
        fea.add_field_output(name=output_name_5,
                        form=output_form_5,
                        arguments=[input_name_1,state_name],
                        record=True)
        force_reshaping_model = ForceReshapingModel(pde=pde,
                                    input_name=shell_name+'_forces',
                                    output_name=input_name_2)
        solid_model = StateModel(fea=fea,
                                debug_mode=False,
                                state_name=state_name,
                                arg_name_list=fea.states_dict[state_name]['arguments'])
        compliance_model = OutputModel(fea=fea,
                                    output_name=output_name_1,
                                    arg_name_list=fea.outputs_dict[output_name_1]['arguments'])
        mass_model = OutputModel(fea=fea,
                                    output_name=output_name_2,
                                    arg_name_list=fea.outputs_dict[output_name_2]['arguments'])
        elastic_energy_model = OutputModel(fea=fea,
                                    output_name=output_name_3,
                                    arg_name_list=fea.outputs_dict[output_name_3]['arguments'])
        pnorm_stress_model = OutputModel(fea=fea,
                                    output_name=output_name_4,
                                    arg_name_list=fea.outputs_dict[output_name_4]['arguments'])
        von_Mises_stress_model = OutputFieldModel(fea=fea,
                                    output_name=output_name_5,
                                    arg_name_list=fea.outputs_field_dict[output_name_5]['arguments'])
        disp_extraction_model = DisplacementExtractionModel(pde=pde,
                                    input_name=state_name,
                                    output_name=shell_name+'_displacement')
        aggregated_stress_model = AggregatedStressModel(m=m, rho=rho,
                                    input_name=output_name_4,
                                    output_name=shell_name+'_aggregated_stress')

        self.add(force_reshaping_model, name='force_reshaping_model')
        self.add(solid_model, name='solid_model')
        self.add(disp_extraction_model, name='disp_extraction_model')
        self.add(compliance_model, name='compliance_model')
        self.add(von_Mises_stress_model, name='von_Mises_stress_model')
        self.add(mass_model, name='mass_model')
        self.add(elastic_energy_model, name='elastic_energy_model')
        self.add(pnorm_stress_model, name='von_mises_stress_model')
        self.add(aggregated_stress_model, name='aggregated_stress_model')

class AggregatedStressModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('m', types=float)
        self.parameters.declare('rho', types=int)
        self.parameters.declare('input_name')
        self.parameters.declare('output_name')

    def define(self):
        m = self.parameters['m']
        rho = self.parameters['rho']
        input_name = self.parameters['input_name']
        output_name = self.parameters['output_name']
        pnorm_stress = self.declare_variable(input_name, val=1.)

        # Expressions with multiple binary operations
        aggregated_stress = 1/m*pnorm_stress**(1/rho)
        self.register_output(output_name, aggregated_stress)

class DisplacementExtractionModel(csdl.Model):
    '''
    Extract and reshape displacement vector into matrix
    '''
    def initialize(self):
        self.parameters.declare('pde')
        self.parameters.declare('input_name')
        self.parameters.declare('output_name')

    def define(self):
        pde = self.parameters['pde']
        input_name = self.parameters['input_name']
        output_name = self.parameters['output_name']
        disp_extraction_mats = pde.construct_nodal_disp_map()
        # Both vector or tensors need to be numpy arrays
        shape = pde.mesh.geometry.x.shape
        dummy_func = Function(pde.W)
        input_size = len(dummy_func.x.array)
        vector = np.arange(input_size)
        # contains all dofs of displacements (CG2) and rotations
        disp_vec = self.declare_variable(input_name, val=vector)
        # contains nodal displacements only (CG1)
        nodal_disp_vec = csdl.matvec(disp_extraction_mats, disp_vec)
        nodal_disp_mat = csdl.reshape(nodal_disp_vec, new_shape=(shape[1],shape[0]))
        # print("nodal_disp_vec shape:", nodal_disp_vec.shape)
        # print("Q_map shape:", disp_extraction_mats.shape)
        self.register_output(output_name,
                             csdl.transpose(nodal_disp_mat))

class ForceReshapingModel(csdl.Model):
    '''
    Reshape force matrix to vector
    '''
    def initialize(self):
        self.parameters.declare('pde')
        self.parameters.declare('input_name')
        self.parameters.declare('output_name')

    def define(self):
        pde = self.parameters['pde']
        input_name = self.parameters['input_name']
        output_name = self.parameters['output_name']
        # Both vector or tensors need to be numpy arrays
        shape = pde.mesh.geometry.x.shape
        dummy_func = Function(pde.VF)
        size = len(dummy_func.x.array)
        vector = np.arange(size)
        tensor = vector.reshape(shape)
        # contains nodal forces (CG1)
        nodal_force_mat = self.declare_variable(input_name, val=tensor)

        self.register_output(output_name,
                             csdl.reshape(nodal_force_mat, new_shape=(size,)))



class ShellPDE(object):
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

    def compliance(self,u_mid,h,dxx):
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
        # initialize sparse mapping matrix
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
            # initialize sparse mapping matrix
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
