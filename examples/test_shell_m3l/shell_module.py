import m3l
import array_mapper as am
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from shell_pde import ShellPDE, ShellModule, NodalMap
import numpy as np
from scipy.sparse.linalg import spsolve

from typing import Tuple, Dict
"""
M3L operations for structural optimization:
>> Shell solver
>> Nodal displacement map
>> Nodal force map
>> ...
"""

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
        rotations = m3l.Variable(name=f'{shell_name}_rotation',
                            shape=mesh.shape, operation=self)
        mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3,3), operation=self)
        return displacements, rotations, mass


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
        csdl_model.register_module_output(f'{shell_name}_forces', -1.0*shell_mesh_forces)

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

        displacement_map = self.umap(mesh.value.reshape((-1,3)),
                        oml=nodal_displacements_mesh.value.reshape((-1,3)))
        shell_displacements_csdl = csdl_model.register_module_input(
                                            name=f'{shell_name}_displacement',
                                            shape=shell_displacements.shape)
        displacement_map_csdl = csdl_model.create_input(
                            f'{shell_name}_displacements_to_nodal_displacements',
                            val=displacement_map)
        nodal_displacements = csdl.matmat(displacement_map_csdl,
                                            shell_displacements_csdl)
        csdl_model.register_module_output(f'{shell_name}_nodal_displacement',
                                            nodal_displacements)

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
        return nodal_displacements

    def umap(self, mesh, oml):
        # Up = W*Us
        G_mat = NodalMap(mesh, oml, RBF_width_par=2.0,
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
