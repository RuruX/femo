import openmdao.api as om
import numpy as np
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI

#import warnings
#warnings.filterwarnings("ignore")

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
        
        
        
class StatesComp(om.ImplicitComponent):
    
    def initialize(self):
        self.options.declare('size')
        self.options['distributed'] = True
        
    def setup(self):
        size = self.options['size']
        comm = self.comm
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        ind = np.arange(Istart, Iend)
        
        self.A = np.eye(mysize)
        self.add_input('f', np.ones(mysize, float), src_indices = ind)
        self.add_output('displacements', np.ones(mysize, float))
        self.declare_partials('displacements','displacements')
        self.declare_partials('displacements','f')
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        print("---- Calling apply nonlinear... ----")
        residuals['displacements'] = outputs['displacements'] - inputs['f']
        
    def solve_nonlinear(self, inputs, outputs):
        print("---- Calling solve nonlinear... ----")
        outputs['displacements'] = inputs['f']
        
    def linearize(self, inputs, outputs, partials):
        print("---- Calling linearize... ----")
        B = self.A
        self.dRdu = B
        self.dRdf = -B
        self.iter = 0
    
    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        print("---- Calling apply linear... ----")
        self.iter += 1
        self.comm.Barrier()
        print(self.comm.rank, self.iter, mode,
        	"--------------------------------------------", flush=True)
        self.comm.Barrier()
        if mode == 'fwd':
            if 'displacements' in d_residuals:
                if 'displacements' in d_outputs:
                    d_residuals['displacements'] += np.dot(self.dRdu, d_outputs['displacements'])
                if 'f' in d_inputs:
                    d_residuals['displacements'] += np.dot(self.dRdf, d_inputs['f'])
                    
        if mode == 'rev':
            if 'displacements' in d_residuals:
                if 'displacements' in d_outputs:
                    d_outputs['displacements'] += np.dot(self.dRdu.T, d_residuals['displacements'])
                if 'f' in d_inputs:
                    d_inputs['f'] += np.dot(self.dRdf.T, d_residuals['displacements'])


            
    def solve_linear(self, d_outputs, d_residuals, mode):
        print("---- Calling solve linear... ----")
        if mode == 'fwd':
            d_outputs['displacements'] = d_residuals['displacements']
        else:
            d_residuals['displacements'] = d_outputs['displacements']
            


if __name__ == '__main__':
    size = 19
    comp = StatesComp(size=size)
    prob = om.Problem()
    comp_1 = DistributedIndepVarComp(
        name='f',
        size=size,
        val=np.ones(size),
    )
    prob.model.add_subsystem('f_comp', comp_1, promotes=['*'])

    comp = StatesComp(size=size)
    prob.model.add_subsystem('states_comp', comp, promotes=['*'])
    
    prob.setup()
    prob.run_model()
    print('*'*20)
    print('Processor #', comp.comm.Get_rank(), '-- check_partials:')
    print('*'*20)
    prob.model.list_outputs()
#    prob.check_partials(compact_print=False)


