from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
import scipy.sparse 
from scipy import spatial


class GeneralFilterModel(Model):

    def initialize(self):
        self.parameters.declare('nel')
        self.parameters.declare('beta', default=2.)
        self.parameters.declare('coordinates')
        self.parameters.declare('h_avg')

    def define(self):
        nel = self.parameters['nel']
        beta = self.parameters['beta']
        coordinates = self.parameters['coordinates']
        h_avg = self.parameters['h_avg']
        density_unfiltered = self.declare_variable('density_unfiltered',
                                    shape=(nel,),
                                    val=1.0)

        e = GeneralFilterOperation(nel=nel, 
                                    beta=beta,
                                    coordinates=coordinates,
                                    h_avg=h_avg)
        output = csdl.custom(density_unfiltered, op=e)
        self.register_output('density', output)


class GeneralFilterOperation(CustomExplicitOperation):
    """
    input: unfiltered density
    output: filtered density
    """
    def initialize(self):
        self.parameters.declare('nel')
        self.parameters.declare('beta', default=2.)
        self.parameters.declare('coordinates')
        self.parameters.declare('h_avg')

    def define(self):
        nel = self.parameters['nel']
        beta = self.parameters['beta']
        coords = self.parameters['coordinates']
        h_avg = self.parameters['h_avg']

        self.add_input('density_unfiltered',
                        shape=(nel,),
                        val=0.0)
        self.add_output('density',
                        shape=(nel,))
        weight_ij, rows, cols = self.compute_weight_mat(coords, h_avg, beta, nel)
        self.weight_mtx = scipy.sparse.csr_matrix((weight_ij, 
                                                    (rows, cols)), 
                                                    shape=(nel, nel))
        self.declare_derivatives('density', 'density_unfiltered',
                                rows=rows, 
                                cols=cols,
                                val=weight_ij)

    def compute(self, inputs, outputs):
        outputs['density'] = self.weight_mtx.dot(inputs['density_unfiltered'])

    def compute_weight_mat(self, coords, h_avg, beta, nel):
        radius = beta * h_avg

        weight_ij = []
        col = []
        row = []

        for i in range(nel):
            current_point = coords[i,:]
            points_selection = coords
            tree = spatial.cKDTree(points_selection)
            idx = tree.query_ball_point(list(current_point), radius)
            nearest_points = points_selection[idx]
            di = np.linalg.norm(current_point - nearest_points,axis = 1)
            weight_sum = sum(radius - di)

            for j in idx:
                dj = np.linalg.norm(current_point - points_selection[j])
                weight = (radius - dj)/weight_sum
                row.append(i)
                col.append(j)
                weight_ij.append(weight)       
       
        return weight_ij, row, col
