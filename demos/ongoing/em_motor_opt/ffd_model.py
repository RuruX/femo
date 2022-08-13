import numpy as np
import os
import csdl
from csdl_om import Simulator
from motor_project.geometry.motor_mesh_class import MotorMesh
from lsdo_mesh.csdl_mesh_models import ShapeParameterModel, EdgeUpdateModel

class ShapeParameterUpdateModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('unique_shape_parameter_list')

    def define(self):
        unique_shape_parameter_list = self.parameters['unique_shape_parameter_list']
        print(unique_shape_parameter_list)
        '''
        COMPUTATION OF MAP BETWEEN DESIGN VARIABLES AND SHAPE PARAMETERS

        LIST OF SHAPE PARAMETERS:
            - inner_stator_radius_sp
            - magnet_thickness_sp
            - magnet_width_sp
            - outer_stator_radius_sp
            - rotor_radius_sp
            - shaft_radius_sp
            - stator_tooth_shoe_thickness_sp
            - winding_top_radius_sp
            - winding_width_sp
        '''

        # THE IDEA HERE IS TO REGISTER ALL OF THE SHAPE PARAMETERS WITHIN 
        # unique_shape_parameter_list AS OUTPUTS TO FEED INTO THE FFD MODELS
        # OR WE USE THE SHAPE PARAMETER AS DESIGN VARIABLES 
        # shaft_radius_dv = self.create_input('shaft_radius_dv')
        # shaft_radius_sp = self.register_output(
        #     'shaft_radius_sp',
        #     1*shaft_radius_dv
        # )

        magnet_thickness_dv = self.create_input('magnet_thickness_dv', val=0.)
        magnet_thickness_sp = self.register_output(
            'magnet_thickness_sp',
            1*magnet_thickness_dv
        )

        '''
        THE FINAL OUTPUTS HERE ARE THE SHAPE PARAMETERS THAT FEED INTO THE 
        INDIVIDUAL MESH MODELS WITHIN INSTANCE MODELS
        LIST OF SHAPE PARAMETERS:
            - inner_stator_radius_sp
            - magnet_thickness_sp
            - magnet_width_sp
            - outer_stator_radius_sp
            - rotor_radius_sp
            - shaft_radius_sp
            - stator_tooth_shoe_thickness_sp
            - winding_top_radius_sp
            - winding_width_sp
        '''

class FFDModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('parametrization_dict')

    def define(self):

        param_dict = self.parameters['parametrization_dict']
        unique_sp_list = sorted(set(param_dict['shape_parameter_list_input']))

        self.add(
            ShapeParameterUpdateModel(
                unique_shape_parameter_list=unique_sp_list
            ),
            'shape_parameter_update_model'
        )

        self.add(
            ShapeParameterModel(
                shape_parameter_list_input=param_dict['shape_parameter_list_input'],
                shape_parameter_index_input=param_dict['shape_parameter_index_input'],
                shape_parametrization=param_dict['shape_parametrization'],
            ),
            'shape_parameter_model'
        )

        self.add(
            EdgeUpdateModel(
                ffd_parametrization=param_dict['ffd_parametrization'][0],
                edge_parametrization=param_dict['edge_parametrization'][0],
                mesh_points=param_dict['mesh_points'][0],
                ffd_cps=param_dict['ffd_cps'][0],
                initial_edge_coords=param_dict['initial_edge_coordinates'][0],
            ),
            'edge_update_model'
        )
