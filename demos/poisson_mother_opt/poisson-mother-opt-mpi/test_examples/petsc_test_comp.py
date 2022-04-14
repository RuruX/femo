from petsc4py import PETSc
import openmdao.api as om
import numpy as np
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI


def zero_PETSc_v(comm,size):
    """
    Create zeros PETSc vector with size.
    """
    v = PETSc.Vec(comm).create(comm)
    v.setSizes(size)
    v.setUp()
    v.assemble()
    return v

def multTranspose(M,b):
    """
    Returns M^T*b, where M and b are "PETSc.Mat" and "PETSc.Vec" objects.
    """
    totalDofs = M.getSizes()[1][1]
    comm = M.getComm()
    MTbv = PETSc.Vec(comm)
    MTbv.create(comm)
    MTbv.setSizes(totalDofs)
    MTbv.setUp()
    M.multTranspose(b,MTbv)
    return MTbv


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
        
        # if comm.size is 2 and size is 15, this results in
        # 8 entries for proc 0 and 7 entries for proc 1
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        
        self.mysize = mysize = sizes[rank]
        Istart, Iend = offsets[rank], offsets[rank]+mysize
        ind = np.arange(Istart, Iend)
        
        rows = ind.astype('int32')
        cols = ind.astype('int32')
        values = np.eye(mysize)
        A = PETSc.Mat(comm).create(PETSc.COMM_WORLD)
        A.setSizes([[mysize,None],[None,size]])
        A.setType("aij")
        A.setUp()
        A.setValues(rows, cols, values)
        A.assemblyBegin()
        A.assemblyEnd()

        self.A = A
        self.du = zero_PETSc_v(A.getComm(),size)
        self.dR = zero_PETSc_v(A.getComm(),size)
        self.df = zero_PETSc_v(A.getComm(),size)
        self.ind = ind
        self.add_input('f', np.ones(mysize, float), src_indices = ind)
        self.add_output('displacements', np.ones(mysize, float))
        self.declare_partials('displacements','displacements')
        self.declare_partials('displacements','f')
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        print("---- Calling apply nonlinear... ----")
        residuals['displacements'] = outputs['displacements'] - 2*inputs['f']
        
    def solve_nonlinear(self, inputs, outputs):
        print("---- Calling solve nonlinear... ----")
        outputs['displacements'] = 2*inputs['f']
        
    def linearize(self, inputs, outputs, partials):
        print("---- Calling linearize... ----")
        self.dRdu = self.A
        self.dRdf = -2*self.A
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
                if 'f' in d_inputs:
                    self.df.setValues(self.ind.astype('int32'), d_inputs['f'])
                    self.df.assemble()
                    dR_PETSc = self.dRdf * self.df
                    dR_PETSc.assemble()
                    d_residuals['displacements'] += dR_PETSc.getArray()

                if 'displacements' in d_outputs:
                    self.du.setValues(self.ind.astype('int32'),
                                    d_outputs['displacements'])
                    self.du.assemble()
                    dR_PETSc = self.dRdu * self.du
                    dR_PETSc.assemble()
                    d_residuals['displacements'] += dR_PETSc.getArray()

        if mode == 'rev':
            if 'displacements' in d_residuals:
                self.dR.setValues(self.ind.astype('int32'),
                                d_residuals['displacements'])
                self.dR.assemble()
                if 'f' in d_inputs:
                    df_PETSc = multTranspose(self.dRdf, self.dR)
                    df_PETSc.assemble()
                    d_inputs['f'] += df_PETSc.getArray()
                if 'displacements' in d_outputs:
                    du_PETSc = multTranspose(self.dRdu, self.dR)
                    du_PETSc.assemble()
                    d_outputs['displacements'] += du_PETSc.getArray()
        else:
            print("     *** Alternative mode ***")
            pass

    
    def solve_linear(self, d_outputs, d_residuals, mode):
        print("---- Calling solve linear... ----")
        if mode == 'fwd':
            d_outputs['displacements'] = d_residuals['displacements']
        else:
            d_residuals['displacements'] = 2*d_outputs['displacements']
            


if __name__ == '__main__':
    size = 19

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
    prob.check_partials(compact_print=False)
    

                                         
