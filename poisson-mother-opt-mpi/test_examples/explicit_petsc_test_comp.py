from petsc4py import PETSc
import openmdao.api as om
import numpy as np
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI

def zero_petsc_vec(comm, localSize, globalSize):
    petsc_vec = PETSc.Vec(comm).create(comm)
    petsc_vec.setSizes([localSize,globalSize])
    petsc_vec.setUp()
    petsc_vec.assemble()
    return petsc_vec

    

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
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        self.ind = np.arange(Istart, Iend)
        self.add_output(name, shape=sizes[rank])

    def compute(self, inputs, outputs):
        name = self.options['name']
        val = self.options['val']
        ind = self.ind
        outputs[name] = val[ind]

    def compute_partials(self, inputs, partials):
        pass



class DotComp(om.ExplicitComponent):

    def initialize(self):
        '''
        Ru: it is possible to pass PETSc vectors to the components, but not sure how exactly this would help.
        '''
#        self.options.declare('petsc_vec_1', types=PETSc.Vec)
#        self.options.declare('petsc_vec_2', types=PETSc.Vec)
        self.options.declare('size', types=int)
        self.options['distributed'] = True

    def setup(self):
        size = self.options['size']
        comm = self.comm
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        ind = np.arange(Istart, Iend)
        self.v1_p = zero_petsc_vec(comm, mysize, size)
        self.v2_p = zero_petsc_vec(comm, mysize, size)
        self.ind = ind
        self.add_input('v1', src_indices=ind)
        self.add_input('v2', src_indices=ind)
        self.add_output('vDot')
        self.declare_partials('vDot','v1')
        self.declare_partials('vDot','v2')

    def compute(self, inputs, outputs):
        print("---- Calling compute()... ----")

        self.v1_p.setValues(self.ind.astype('int32'), inputs['v1'])
        self.v2_p.setValues(self.ind.astype('int32'), inputs['v2'])
        outputs['vDot'] = self.v1_p.dot(self.v2_p)

    def compute_partials(self, inputs, partials):
        print("---- Calling compute_partials()... ----")

        self.v1_p.setValues(self.ind.astype('int32'), inputs['v1'])
        self.v2_p.setValues(self.ind.astype('int32'), inputs['v2'])
        partials['vDot','v1'] = self.v2_p.getArray()
        partials['vDot','v2'] = self.v1_p.getArray()

if __name__ == '__main__':
    size = 16
    prob = om.Problem()
    comp_1 = DistributedIndepVarComp(
        name='v1',
        size=size,
        val=np.ones(size),
    )
    prob.model.add_subsystem('v1_comp', comp_1, promotes=['*'])

    comp_2 = DistributedIndepVarComp(
        name='v2',
        size=size,
        val=np.ones(size),
    )
    prob.model.add_subsystem('v2_comp', comp_2, promotes=['*'])

    comp = DotComp(size=size)
    prob.model.add_subsystem('dot_comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    print('*'*20)
    print('Processor #', comp.comm.Get_rank(), '-- check_partials:')
    print(prob['vDot'])
    print('*'*20)
    prob.check_partials(compact_print=False)
