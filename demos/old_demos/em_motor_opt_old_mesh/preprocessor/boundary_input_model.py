from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator
from scipy.sparse import csr_matrix

class BoundaryInputModel(Model):
    """
    input: edge_deltas
    output: uhat_bc
    """
    def initialize(self):

        self.parameters.declare('edge_indices')
        self.parameters.declare('output_size', types=int)
    def define(self):
        edge_indices = self.parameters['edge_indices']
        output_size = self.parameters['output_size']
        input_size = len(edge_indices)
        edge_deltas = self.declare_variable('edge_deltas',
                        shape=(input_size,),
                        val=np.zeros(input_size).reshape(input_size,))

        e = LinearMapping(
                        input_name='edge_deltas',
                        output_name='uhat_bc',
                        input_size=input_size,
                        output_size=output_size,
                        indices=edge_indices)
        output = csdl.custom(edge_deltas, op=e)
        self.register_output('uhat_bc', output)


class LinearMapping(CustomExplicitOperation):
    """
    input: array_1
    output: array_2
    """
    def initialize(self):
        self.parameters.declare('input_size')
        self.parameters.declare('output_size')
        self.parameters.declare('indices')
        self.parameters.declare('input_name')
        self.parameters.declare('output_name')

    def define(self):
        self.indices = self.parameters['indices']
        self.input_size = self.parameters['input_size']
        self.output_size = self.parameters['output_size']
        self.input_name = self.parameters['input_name']
        self.output_name = self.parameters['output_name']
        self.add_input(self.input_name,
                        shape=(self.input_size,),
                        val=0.0)
        self.add_output(self.output_name,
                        shape=(self.output_size,),)
        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        array_2 = np.zeros(self.output_size)
        for i in range(self.input_size):
            array_2[self.indices[i]] = inputs[self.input_name][i]
        outputs[self.output_name] = array_2

    def compute_derivatives(self, inputs, derivatives):
        row_ind = self.indices
        col_ind = np.arange(self.input_size)
        data = np.ones(self.input_size)
        M = csr_matrix((data, (row_ind, col_ind)),
                        shape=(self.output_size, self.input_size))
        derivatives[self.output_name, self.input_name] = M.toarray()

if __name__ == "__main__":
    edge_deltas = np.array([1., 0., 0., 0.1])
    edge_indices = np.array([1,2,5,7])
    output_size = 10
    model = BoundaryInputModel(edge_indices=edge_indices,
                                output_size=output_size)
    sim = py_simulator(model)
    sim['edge_deltas'] = edge_deltas
    sim.run()
    print(sim['uhat_bc'])
    sim.check_partials(compact_print=True)
