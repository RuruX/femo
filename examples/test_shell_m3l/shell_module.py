import m3l
import array_mapper as am
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from shell_pde import ShellPDE, ShellModule, NodalMap
# from shell_pde import ShellPDE, ShellModule, ShellResidual, ShellResidualJacobian, NodalMap
import numpy as np
from scipy.sparse.linalg import spsolve

from typing import Tuple, Dict
from copy import deepcopy
"""
M3L operations for structural optimization:
>> Shell solver
>> Nodal displacement map
>> Nodal force map
>> ...
"""

class RMShellM3LDisplacement(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.pde = self.parameters['pde']
        self.shells = self.parameters['shells']


    def evaluate(self, forces:m3l.Variable=None,
                        moments:m3l.Variable=None,
                        thicknesses:m3l.Variable=None,
                        displacements:m3l.Variable=None) -> m3l.Variable:
        '''
        Evaluates the shell model.

        Parameters
        ----------
        forces : m3l.Variable = None
            The forces on the mesh nodes.
        moments : m3l.Variable = None
            The moments on the mesh nodes.
        thicknesses : m3l.Variable = None
            The thicknesses on the mesh nodes.

        Returns
        -------
        displacements : m3l.Variable
            The displacements of the mesh nodes.
        rotations : m3l.Variable
            The rotations of the mesh nodes.

        '''

        # Gets information for naming/shapes
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        self.component = self.parameters['component']
        # Define operation arguments
        self.name = f'{self.component.name}_rm_shell_model'

        self.inputs = {}
        if thicknesses is not None:
            self.inputs[f'{shell_name}_thicknesses'] = thicknesses
        if forces is not None:
            self.inputs[f'{shell_name}_forces'] = forces
        if moments is not None:
            self.inputs[f'{shell_name}_moments'] = moments

        # Define operation outputs (still an input to the residual) (It may not be necessary to distinguish these two, we'll see)
        self.outputs ={}
        if displacements is not None:
            self.outputs[f'{shell_name}_displacements'] = displacements

        # Declare residual partials - key is the residual csdl name, value is the m3l variable that's being partialed
        self.residual_partials = {}
        self.residual_partials['displacement_jacobian'] = displacements
        # self.residual_partials['force_jacobian'] = forces

        # create residual variable
        self.size = 6*2*mesh.shape[0]
        residual = m3l.Variable(name=f'{shell_name}_displacement', shape=(6*2*mesh.shape[0],), operation=self)
        return

    def compute_residual(self):
        shells = self.parameters['shells']
        pde = self.parameters['pde']
        # Need to define what the residual is called in the csdl model
        shell_name = list(self.parameters['shells'].keys())[0]
        self.residual_name = f'{shell_name}_residual'

        # CSDL model computing residuals - inputs are named as defined in evaluate()
        csdl_model = LinearShellResidualCSDL(
            module=self,
            pde=pde,
            shells=shells)
        return csdl_model

    # optional method
    def solve_residual_equations(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        shells = self.parameters['shells']
        pde = self.parameters['pde']
        # inputs/outputs named as defined in evaluate()
        csdl_model = LinearShellCSDL(
            module=self,
            pde=pde,
            shells=shells,)
        return csdl_model


    # optional-ish for dynamic, not needed for static
    # The idea here is that, if this method is not defined, m3l will finite difference to get the relevant values. 
    # Alternitively, a solver dev could finite difference for themselves and put it here.
    def compute_derivatives(self):
        shells = self.parameters['shells']
        pde = self.parameters['pde']
        # Need to define what the residual is called in the csdl model
        shell_name = list(self.parameters['shells'].keys())[0]

        # CSDL model computing residual jacobian - inputs and partials are named as defined in evaluate()
        csdl_model = LinearShellResidualJacobiansCSDL(
            module=self,
            pde=pde,
            shells=shells)
        
        return csdl_model
    

class RMShell(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.pde = self.parameters['pde']
        self.shells = self.parameters['shells']

    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        shells = self.parameters['shells']
        pde = self.parameters['pde']

        csdl_model = LinearShellCSDL(
            module=self,
            pde=pde,
            shells=shells)

        return csdl_model

    def evaluate(self, forces:m3l.Variable=None,
                        moments:m3l.Variable=None,
                        thicknesses:m3l.Variable=None) -> m3l.Variable:
        '''
        Evaluates the shell model.

        Parameters
        ----------
        forces : m3l.Variable = None
            The forces on the mesh nodes.
        moments : m3l.Variable = None
            The moments on the mesh nodes.

        Returns
        -------
        displacements : m3l.Variable
            The displacements of the mesh nodes.
        rotations : m3l.Variable
            The rotations of the mesh nodes.

        '''

        # Gets information for naming/shapes
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        self.component = self.parameters['component']
        # Define operation arguments
        self.name = f'{self.component.name}_rm_shell_model'

        self.arguments = {}
        # TODO: add input for thicknesses
        if thicknesses is not None:
            self.arguments[f'{shell_name}_thicknesses'] = thicknesses
        if forces is not None:
            self.arguments[f'{shell_name}_forces'] = forces
        if moments is not None:
            self.arguments[f'{shell_name}_moments'] = moments

        # Create the M3L variables that are being output
        displacements = m3l.Variable(name=f'{shell_name}_displacement',
                            shape=mesh.shape, operation=self)
        stresses = m3l.Variable(name=f'{shell_name}_stress',
                            shape=mesh.shape[0], operation=self)
        rotations = m3l.Variable(name=f'{shell_name}_rotation',
                            shape=mesh.shape, operation=self)
        mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3,3), operation=self)
        return displacements, rotations, stresses, mass


class RMShellForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.pde = self.parameters['pde']
        self.shells = self.parameters['shells']

    def compute(self) -> csdl.Model:
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        self.pde = pde = self.parameters['pde']
        nodal_forces = self.arguments['nodal_forces']

        csdl_model = ModuleCSDL()

        force_map = self.fmap(mesh.value.reshape((-1,3)),
                                oml=self.nodal_forces_mesh.value.reshape((-1,3)))

        flattened_nodal_forces_shape = (np.prod(nodal_forces.shape[:-1]),
                                        nodal_forces.shape[-1])
        nodal_forces_csdl = csdl_model.register_module_input(
                                                name='nodal_forces',
                                                shape=nodal_forces.shape)
        flattened_nodal_forces = csdl.reshape(nodal_forces_csdl,
                                                new_shape=flattened_nodal_forces_shape)
        force_map_csdl = csdl_model.create_input(f'nodal_to_{shell_name}_forces_map', val=force_map)
        flatenned_shell_mesh_forces = csdl.matmat(force_map_csdl, flattened_nodal_forces)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        shell_mesh_forces = csdl.reshape(flatenned_shell_mesh_forces, new_shape=output_shape)

        # define matrix that scales the coordinates in the defined way to align them with other tools
        scaling_mat = np.diag(np.array([-1., 1., 1.]))
        scaling_mat_csdl = csdl_model.create_input(f'nodal_to_{shell_name}_scaling_mat', val=scaling_mat)

        reoriented_shell_mesh_forces = csdl.matmat(shell_mesh_forces, scaling_mat_csdl)
        csdl_model.register_module_output(f'{shell_name}_forces', reoriented_shell_mesh_forces)
        # NOTE: Why the factor `-1.0` in the line above?
        return csdl_model


    def evaluate(self, nodal_forces:m3l.Variable,
                    nodal_forces_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces from arbitrary locations to the mesh nodes.

        Parameters
        ----------
        nodal_forces : m3l.Variable
            The forces to be mapped to the mesh nodes.
        nodal_forces_mesh : m3l.Variable
            The mesh that the nodal forces are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''
        self.component = self.parameters['component']
        self.name = f'{self.component.name}_rm_shell_force_mapping'

        self.nodal_forces_mesh = nodal_forces_mesh
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.

        self.arguments = {'nodal_forces': nodal_forces}
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        shell_forces = m3l.Variable(name=f'{shell_name}_forces',
                            shape=output_shape, operation=self)
        return shell_forces


    def fmap(self, mesh, oml):
        G_mat = NodalMap(mesh, oml, RBF_width_par=4.,
                            column_scaling_vec=self.pde.bf_sup_sizes)
        rhs_mats = G_mat.map.T
        mat_f_sp = self.pde.compute_sparse_mass_matrix()
        weights = spsolve(mat_f_sp, rhs_mats)
        return weights


class RMShellNodalDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.pde = self.parameters['pde']
        self.shells = self.parameters['shells']

    def compute(self)->csdl.Model:

        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        self.pde = pde = self.parameters['pde']

        nodal_displacements_mesh = self.nodal_displacements_mesh
        shell_displacements = self.arguments[f'{shell_name}_displacement']

        csdl_model = ModuleCSDL()

        # create mirrored mesh coordinates
        mesh_original = mesh.value.reshape((-1,3))
        # mesh_mirrored = deepcopy(mesh_original)
        # mesh_mirrored[:, 1] *= -1.  # mirror coordinates along the y-axis
        # we concatenate both meshes (original and mirrored) and compute the displacement map
        # mesh_concat = np.vstack([mesh_original, mesh_mirrored])
        displacement_map = self.umap(mesh_original,
                        oml=nodal_displacements_mesh.value.reshape((-1,3)),
                        repeat_bf_sup_size_vector=False)

        shell_displacements_csdl = csdl_model.register_module_input(
                                            name=f'{shell_name}_displacement',
                                            shape=shell_displacements.shape)

        # we create a matrix as csdl operator that repeats the shell displacement variables twice
        # rep_mat = np.vstack([np.eye(shell_displacements.shape[0])]*2)
        # we manually set the fifth entry of rep_mat to `-1`, since the y-displacement is mirrored
        # rep_mat[4, 4] = -1.

        # rep_mat_csdl = csdl_model.create_input(f'{shell_name}_displacement_repeater_mat', val=rep_mat)
        # compute repeated shell_displacements_csdl
        # shell_displacements_csdl_rep = csdl.matmat(rep_mat_csdl,
                                            # shell_displacements_csdl)

        displacement_map_csdl = csdl_model.create_input(
                            f'{shell_name}_displacements_to_nodal_displacements',
                            val=displacement_map)
        # displacement_map_rightwing_csdl = csdl_model.create_input(
        #                     f'{shell_name}_rightwing_displacements_to_nodal_displacements',
        #                     val=displacement_map)
        nodal_displacements = csdl.matmat(displacement_map_csdl,
                                            shell_displacements_csdl)
        csdl_model.register_module_output(f'{shell_name}_nodal_displacement',
                                            nodal_displacements)        
        csdl_model.register_module_output(f'{shell_name}_tip_displacement',
                                            csdl.max(nodal_displacements, rho=1000))

        return csdl_model

    def evaluate(self, shell_displacements:m3l.Variable,
                nodal_displacements_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces and moments from arbitrary locations to the mesh nodes.

        Parameters
        ----------
        shell_displacements : m3l.Variable
            The displacements to be mapped from the shell mesh to the desired mesh.
        nodal_displacements_mesh : m3l.Variable
            The mesh to evaluate the displacements over.

        Returns
        -------
        nodal_displacements : m3l.Variable
            The displacements on the given nodal displacements mesh.
        '''
        self.component = self.parameters['component']
        shell_name = list(self.parameters['shells'].keys())[0]   # this is only taking the first mesh added to the solver.
        self.name = f'{self.component.name}_rm_shell_displacement_map'
        self.arguments = {f'{shell_name}_displacement': shell_displacements}
        self.nodal_displacements_mesh = nodal_displacements_mesh

        nodal_displacements = m3l.Variable(name=f'{shell_name}_nodal_displacement',
                                            shape=nodal_displacements_mesh.shape,
                                            operation=self)
        tip_displacement = m3l.Variable(name=f'{shell_name}_tip_displacement',
                                    shape=(1,),
                                    operation=self)
        return nodal_displacements, tip_displacement

    def umap(self, mesh, oml, repeat_bf_sup_size_vector=False):
        # Up = W*Us
        if repeat_bf_sup_size_vector:
            col_scaling_vec = np.concatenate([self.pde.bf_sup_sizes]*2)
        else:
            col_scaling_vec = self.pde.bf_sup_sizes

        G_mat = NodalMap(mesh, oml, RBF_width_par=5.,
                            column_scaling_vec=col_scaling_vec)
        weights = G_mat.map
        return weights


class RMShellNodalStress(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.pde = self.parameters['pde']
        self.shells = self.parameters['shells']

    def compute(self)->csdl.Model:

        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        self.pde = pde = self.parameters['pde']

        nodal_stress_mesh = self.nodal_stress_mesh
        shell_stress = self.arguments[f'{shell_name}_stress']

        csdl_model = ModuleCSDL()

        stress_map = self.stressmap(mesh.value.reshape((-1,3)),
                        oml=nodal_stress_mesh.value.reshape((-1,3)))
        shell_stress_csdl = csdl_model.register_module_input(
                                            name=f'{shell_name}_stress',
                                            shape=shell_stress.shape)
        stress_map_csdl = csdl_model.create_input(
                            f'{shell_name}_stress_to_nodal_stress',
                            val=stress_map)
        nodal_stress = csdl.matvec(stress_map_csdl,
                                            shell_stress_csdl)
        csdl_model.register_module_output(f'{shell_name}_nodal_stress',
                                            nodal_stress)

        return csdl_model

    def evaluate(self, shell_stress:m3l.Variable,
                nodal_stress_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces and moments from arbitrary locations to the mesh nodes.

        Parameters
        ----------
        shell_stress : m3l.Variable
            The stress to be mapped from the shell mesh to the desired mesh.
        nodal_stress_mesh : m3l.Variable
            The mesh to evaluate the stress over.

        Returns
        -------
        nodal_stress : m3l.Variable
            The stress on the given nodal stress mesh.
        '''
        self.component = self.parameters['component']
        shell_name = list(self.parameters['shells'].keys())[0]   # this is only taking the first mesh added to the solver.
        self.name = f'{self.component.name}_rm_shell_stress_map'
        self.arguments = {f'{shell_name}_stress': shell_stress}
        self.nodal_stress_mesh = nodal_stress_mesh

        nodal_stress = m3l.Variable(name=f'{shell_name}_nodal_stress',
                                            shape=nodal_stress_mesh.shape[0],
                                            operation=self)
        return nodal_stress

    def stressmap(self, mesh, oml):
        G_mat = NodalMap(mesh, oml, RBF_width_par=5.0,
                            column_scaling_vec=self.pde.bf_sup_sizes)
        weights = G_mat.map
        return weights
    
    
class LinearShellMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')


class LinearShellCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties


    def define(self):
        pde = self.parameters['pde']
        shells = self.parameters['shells']
        # solve the shell group:
        self.add_module(ShellModule(pde=pde,shells=shells), name='rm_shell')


class LinearShellResidualCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties


    def define(self):
        pde = self.parameters['pde']
        shells = self.parameters['shells']
        # solve the shell group:
        self.add_module(ShellResidual(pde=pde,shells=shells), name='rm_shell_residual')


class LinearBeamResidualJacobiansCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties

    def define(self):
        pde = self.parameters['pde']
        shells = self.parameters['shells']

        # solve the beam group:
        self.add_module(ShellResidualJacobian(pde=pde,shells=shells), name='rm_shell_residual_jacobian')

class RMShellStrain(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass

class RMShellStress(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass