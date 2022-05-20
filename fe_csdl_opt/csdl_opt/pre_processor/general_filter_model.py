from fe_csdl_opt.fea.fea_dolfinx import *
from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
import scipy.sparse 
from scipy import spatial


class GeneralFilterModel(Model):

    def initialize(self):
        self.parameters.declare('num_element_unfiltered')
        self.parameters.declare('num_element_filtered', default=2.)
        self.parameters.declare('coordinates')
        self.parameters.declare('h_avg')

    def define(self):
        num_element_unfiltered = self.parameters['num_element_unfiltered']
        num_element_filtered = self.parameters['num_element_filtered']
        coordinates = self.parameters['coordinates']
        h_avg = self.parameters['h_avg']
        density_unfiltered = self.declare_variable('density_unfiltered',
                                    shape=(num_element_unfiltered,),
                                    val=1.0)

        e = GeneralFilterOperation(num_element_unfiltered, 
                                    num_element_filtered,
                                    coordinates,
                                    h_avg)
        output = csdl.custom(density_unfiltered, op=e)
        self.register_output('density', output)


class GeneralFilterOperation(CustomExplicitOperation):
    """
    input: unfiltered density
    output: filtered density
    """
    def initialize(self):
        self.parameters.declare('num_element_unfiltered')
        self.parameters.declare('num_element_filtered', default=2.)
        self.parameters.declare('coordinates')
        self.parameters.declare('h_avg')

    def define(self):
        num_element_unfiltered = self.parameters['num_element_unfiltered']
        num_element_filtered = self.parameters['num_element_filtered']
        coordinates = self.parameters['coordinates']
        h_avg = self.parameters['h_avg']

        self.input_size = num_element_unfiltered
        self.output_size = num_element_filtered
        self.add_input('density_unfiltered',
                        shape=(self.input_size,),
                        val=0.0)
        self.add_output('density',
                        shape=(self.output_size,))
        # coords = density_function_space.tabulate_dof_coordinates()
        # h_avg = (density_function_space.mesh.hmax() 
        #         + density_function_space.mesh.hmin())/2
        self.weightMat, row, col = self.compute_weight_mat(coordinates, h_avg,
                                                            num_element_filtered,
                                                            num_element_unfiltered)
                                                            
        self.declare_derivatives('density', 'density_unfiltered',
                                rows=np.array(row), 
                                cols=np.array(col),
                                val=np.array(self.weightMat))

    def compute(self, inputs, outputs):
        outputs['density'] = self.weightMat.dot(inputs['density_unfiltered'])

    def compute_weight_mat(self, coords, h_avg, nel_filtered, nel):
        radius = nel_filtered * h_avg

        weight_ij = []
        col = []
        row = []

        for i in range(nel):
            current_point = coords[i,:]
            points_selection = coords
            tree = spatial.cKDTree(points_selection)
            idx = tree.query_ball_point(list(current_point), radius)
            nearest_points = points_selection[idx]
            
            weight_sum = sum(radius - np.linalg.norm(current_point - nearest_points,axis = 1))

            for j in idx:
                weight = ( radius - np.linalg.norm(current_point - points_selection[j]))/weight_sum
                row.append(i)
                col.append(j)
                weight_ij.append(weight)       
       
        weight_mtx = scipy.sparse.csr_matrix((weight_ij, (row, col)), shape=(nel, nel))
        return weight_mtx, row, col
