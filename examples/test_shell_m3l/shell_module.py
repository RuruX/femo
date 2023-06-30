import m3l
import array_mapper as am
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from shell_pde import ShellPDE, ShellModule, NodalMap
import numpy as np

"""
M3L operations for structural optimizatiom:
>> Shell solver
>> Nodal displacement map
>> Nodal force map
>> ...
"""

class RMShell(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None) # mesh information
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={}) # material properties


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
                        moments:m3l.Variable=None) -> m3l.Variable:
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

        # Assembles the CSDL model
        operation_csdl = self.compute()

        # Gets information for naming/shapes
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.

        arguments = {}
        if forces is not None:
            arguments[f'{shell_name}_forces'] = forces
        if moments is not None:
            arguments[f'{shell_name}_moments'] = moments

        # Create the M3L graph operation
        shell_operation = m3l.CSDLOperation(name='rm_shell_model',
                            arguments=arguments, operation_csdl=operation_csdl)
        # Create the M3L variables that are being output
        displacements = m3l.Variable(name=f'{shell_name}_displacement',
                            shape=mesh.shape, operation=shell_operation)
        rotations = m3l.Variable(name=f'{shell_name}_rotation',
                            shape=mesh.shape, operation=shell_operation)

        return displacements, rotations


class RMShellForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def compute(self, nodal_forces:m3l.Variable,
                nodal_forces_mesh:am.MappedArray) -> csdl.Model:

        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.

        csdl_model = ModuleCSDL()

        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_forces_mesh.value.reshape((-1,3)))

        flattened_nodal_forces_shape = (np.prod(nodal_forces.shape[:-1]), nodal_forces.shape[-1])
        nodal_forces = csdl_model.register_module_input(name='nodal_forces', shape=nodal_forces.shape)
        flattened_nodal_forces = csdl.reshape(nodal_forces, new_shape=flattened_nodal_forces_shape)
        force_map_csdl = csdl_model.create_input(f'nodal_to_{shell_name}_forces_map', val=force_map)
        flatenned_shell_mesh_forces = csdl.matmat(force_map_csdl, flattened_nodal_forces)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        shell_mesh_forces = csdl.reshape(flatenned_shell_mesh_forces, new_shape=output_shape)
        csdl_model.register_module_output(f'{shell_name}_forces', shell_mesh_forces)

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
        operation_csdl = self.compute(nodal_forces=nodal_forces,
                                        nodal_forces_mesh=nodal_forces_mesh)

        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.

        arguments = {'nodal_forces': nodal_forces}
        force_map_operation = m3l.CSDLOperation(name='rmshell_force_map',
                            arguments=arguments, operation_csdl=operation_csdl)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        shell_forces = m3l.Variable(name=f'{shell_name}_forces',
                            shape=output_shape, operation=force_map_operation)
        return shell_forces

    def fmap(self, mesh, oml):
        # Fs = W*Fp

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((n, m))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)

            weights[ia, i] = (l - ac)/length
            weights[ib, i] = (l - bc)/length

        return weights

class RMShellNodalDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('pde', default=None)
        self.parameters.declare('shells', default={})

    def compute(self, shell_displacements:m3l.Variable,
                nodal_displacements_mesh:am.MappedArray)->csdl.Model:
        shell_name = list(self.parameters['shells'].keys())[0]
        # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]
        # this is only taking the first mesh added to the solver.
        pde = self.parameters['pde']

        csdl_model = ModuleCSDL()

        # nodal_disp_map = pde.construct_nodal_disp_map().todense()
        umap = self.umap(mesh.value.reshape((-1,3)),
                        oml=nodal_displacements_mesh.value.reshape((-1,3)))

        displacement_map = umap

        shell_displacements = csdl_model.register_module_input(
                                            name=f'{shell_name}_displacement',
                                            shape=shell_displacements.shape)
        displacement_map_csdl = csdl_model.create_input(
                            f'{shell_name}_displacements_to_nodal_displacements',
                            val=displacement_map)
        nodal_displacements = csdl.matmat(displacement_map_csdl,
                                            shell_displacements)
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
        operation_csdl = self.compute(shell_displacements, nodal_displacements_mesh)

        shell_name = list(self.parameters['shells'].keys())[0]   # this is only taking the first mesh added to the solver.

        arguments = {f'{shell_name}_displacement': shell_displacements}
        displacement_map_operation = m3l.CSDLOperation(
                                            name='rm_shell_displacement_map',
                                            arguments=arguments,
                                            operation_csdl=operation_csdl)
        nodal_displacements = m3l.Variable(name=f'{shell_name}_nodal_displacement',
                                            shape=nodal_displacements_mesh.shape,
                                            operation=displacement_map_operation)
        return nodal_displacements

    def umap(self, mesh, oml):
        # Up = W*Us

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((m,n))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)

            weights[i, ia] = (l - ac)/length
            weights[i, ib] = (l - bc)/length

        ## Compute weights with Sebastiaan's mapping matrix
        # G_mat = NodalMap(mesh, oml, RBF_width_par=0.5)
        # weights = G_mat.map
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
        self.add_module(ShellModule(pde=pde,shells=shells), name='RM_shell')


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
