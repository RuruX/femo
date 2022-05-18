import dolfin as df
import numpy as np
import scipy.sparse 
from scipy import spatial


from openmdao.api import ExplicitComponent


class GeneralFilterComp(ExplicitComponent):
    """
    GeneralFilterComp calculates the filtered densities
    with its derivatives.
    The filter radius is N times the average element size.
    Parameters
    ----------
    density_function_space   
       The FEniCS function space of the density variables
    num_element_filtered : float
        the filter radius 
        (the number multiplied by the average element size)
    Returns
    -------
    outputs[density] numpy array
        filtered densities
    """

    def initialize(self):
        self.options.declare('density_function_space')
        self.options.declare('num_element_filtered', default=2.)

   
    def setup(self):
        density_function_space = self.options['density_function_space']
        num_element_filtered = self.options['num_element_filtered']
        NUM_ELEMENTS = density_function_space.dim()


        self.add_input('density_unfiltered', shape=NUM_ELEMENTS)
        self.add_output('density', shape=NUM_ELEMENTS)

        scalar_output_center_coord = density_function_space.tabulate_dof_coordinates()

        ''' Todo: use average element size to define the radius '''

        # filter radius defined as two times the average size

        mesh_size_max = density_function_space.mesh().hmax()
        mesh_size_min = density_function_space.mesh().hmin()

        radius = num_element_filtered * ((mesh_size_max + mesh_size_min) /2)

        weight_ij = []
        col = []
        row = []

        for i in range(NUM_ELEMENTS):
            current_point = scalar_output_center_coord[i,:]
            points_selection = scalar_output_center_coord
            tree = spatial.cKDTree(points_selection)
            idx = tree.query_ball_point(list(current_point), radius)
            nearest_points = points_selection[idx]
            
            weight_sum = sum(radius - np.linalg.norm(current_point - nearest_points,axis = 1))

            for j in idx:
                weight = ( radius - np.linalg.norm(current_point - points_selection[j]))/weight_sum
                row.append(i)
                col.append(j)
                weight_ij.append(weight)       
       
        self.weight_mtx = scipy.sparse.csr_matrix((weight_ij, (row, col)), shape=(NUM_ELEMENTS, NUM_ELEMENTS))

        self.declare_partials('density', 'density_unfiltered',rows=np.array(row), cols=np.array(col),val=np.array(weight_ij))

    def compute(self, inputs, outputs):
        outputs['density'] = self.weight_mtx.dot(inputs['density_unfiltered'])