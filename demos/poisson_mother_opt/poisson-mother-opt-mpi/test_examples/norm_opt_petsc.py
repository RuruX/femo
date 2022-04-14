from petsc4py import PETSc
import numpy as np
import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs
from mpi4py import MPI as pympi
from openmdao.utils.mpi import MPI as ommpi


def zero_PETSc_v(comm, size):
    """
    Create zero PETSc vector with size.
    """
    v = PETSc.Vec(comm).create(comm)
    v.setSizes(size)
    v.setUp()
    v.assemble()
    return v
    
def PETSc_v(comm, size, val, ind):
    """
    Create PETSc vector with size, values and indices.
    """
    v = PETSc.Vec(comm).create(comm)
    v.setSizes(size)
    v.setUp()
    v.setValues(ind.astype('int32'), val[ind])
    v.assemble()
    return v



class DistributedIndepVarComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('name', types=str)
        self.options.declare('val', types=np.ndarray)
        self.options.declare('size', types=int)
        self.options['distributed'] = True

    def setup(self):
        name = self.options['name']
        size = self.options['size']
        val = self.options['val']        
        
        comm = self.comm
        self.rank = rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        self.ind = ind = np.arange(Istart, Iend)
        
        self.add_output(name, shape=mysize)

    def compute(self, inputs, outputs):
        name = self.options['name']
        val = self.options['val']
        ind = self.ind
        outputs[name] = val[ind]

    def compute_partials(self, inputs, partials):
        pass




class NormComp(om.ExplicitComponent):

    def initialize(self):
        # pass the petsc vectors as parallel containers
        self.options.declare('size', types=int)
        self.options.declare('val', types=np.ndarray)
        self.options['distributed'] = True
        
    def setup(self):
        size = self.options['size']
        val = self.options['val']
        comm = self.comm
        self.rank = rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        self.ind = ind = np.arange(Istart, Iend)
        
        self.v1_petsc = zero_PETSc_v(comm, size)
        self.v2_petsc = PETSc_v(comm, size, val, ind)
        
        self.add_input('v1', shape=size)
        self.add_output('L2norm')
        self.declare_partials('L2norm', 'v1')

    def compute(self, inputs, outputs):
        self.v1_petsc.setValues(self.ind.astype('int32'), inputs['v1'][self.ind])
        self.v1_petsc.assemble()
        self.err = self.v1_petsc-self.v2_petsc
        self.err.assemble()
        outputs['L2norm'] = np.dot(self.err.getArray(),self.err.getArray())

    def compute_partials(self, inputs, partials):
        size = self.options['size']
        dLdv = np.zeros(size)
        dLdv[self.ind] = 2 * self.err.getArray()
        partials['L2norm', 'v1'] = dLdv



class SummerComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('comm_size', types=int)
        
    def setup(self):
        comm_size = self.options['comm_size']
        self.add_input('vec',shape=comm_size)
        self.add_output('sum')
        self.declare_partials('sum','vec',val=np.ones(comm_size))
    
    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['vec'])
        
        


class NormGroup(om.Group):

    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        size = self.options['size']
        comm_size = pympi.COMM_WORLD.Get_size()

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('v1', shape=size)

        """
        If we use distributed design variable component, there will be an error.
        RuntimeError: Distributed design variables are not supported by this driver (pyOptSparseDriver), but the following variables are distributed: [inputs_comp.f]
        """
#        
#        inputs_comp = DistributedIndepVarComp(
#                        name='v1', size=size, val=np.ones(size))

        self.add_subsystem('inputs_comp', inputs_comp)

        norm_comp = NormComp(val=np.arange(size), size=size)
        self.add_subsystem('norm_comp', norm_comp)
        
        summer_comp = SummerComp(comm_size=comm_size)
        self.add_subsystem('summer_comp', summer_comp)
        
        self.connect('inputs_comp.v1', 'norm_comp.v1')
        self.connect('norm_comp.L2norm', 'summer_comp.vec')
        self.add_design_var('inputs_comp.v1')
        self.add_objective('summer_comp.sum')


if __name__ == '__main__':

    size = 19
    
    group = NormGroup(size=size)
    prob = om.Problem(model=group)
#    prob.driver = driver = om.pyOptSparseDriver()
#    driver.options['optimizer']='SNOPT'
#    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
#    prob.driver.opt_settings['Major optimality tolerance'] = 1e-12
#    prob.driver.options['print_results'] = False
##    prob.driver.opt_settings['Major opti limit'] = 100

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    prob.setup()
    prob.run_model()
    print('*'*20)
    print('Processor #', pympi.COMM_WORLD.Get_rank(), 'Run model:')
    print('The L2 norm: ', prob.get_val('norm_comp.L2norm', get_remote=True))
    print('*'*20)
    
    prob.run_driver()
    print('*'*20)
    print('Processor ', pympi.COMM_WORLD.Get_rank(), 'Run optimization:')
    print('The L2 norm: ', prob.get_val('norm_comp.L2norm', get_remote=True))
    print('*'*20)
    
    
    
