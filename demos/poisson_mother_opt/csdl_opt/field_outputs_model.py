# Import `dolfin` first to avoid segmentation fault
from dolfin import *

from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from fea import *

class FieldOutputsModel(Model):

    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size_u = self.fea.total_dofs_u
        u = self.declare_variable('u',
                        shape=(self.input_size_u,),
                        val=0.0)
        e = AzAirGap(fea=self.fea)
        u_air_gap = csdl.custom(u, op=e)
        self.register_output('u_air_gap', u_air_gap)


class FieldOutputsOperation(CustomExplicitOperation):
    """
    input: u
    output: field output(s)
    """
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = self.parameters['fea']
        self.input_size_u = self.fea.total_dofs_u
        self.output_size = len(self.fea.u_air_gap_indices)
        self.add_input('u',
                        shape=(self.input_size_u,),
                        val=0.0)
        self.add_output('u_air_gap',
                        shape=(self.output_size,),
                        val=0.0)
        self.declare_derivatives('u_air_gap', 'u')
        
    def compute(self, inputs, outputs):
        update(self.fea.u, inputs['u'])
        outputs['u_air_gap'] = self.fea.extractAzAirGap()

    def compute_derivatives(self, inputs, derivatives):
        update(self.fea.u, inputs['u'])
        dA_ag_du = self.fea.extractAzAirGapDerivatives()
        derivatives['u_air_gap', 'u'] = dA_ag_du.todense()

                    
if __name__ == "__main__":
    iq                  = 282.2 / 3
    f = open('edge_deformation_data/init_edge_coords.txt', 'r+')
    old_edge_coords = np.fromstring(f.read(), dtype=float, sep=' ')
    f.close()

    f = open('edge_deformation_data/edge_coord_deltas.txt', 'r+')
    edge_deltas = np.fromstring(f.read(), dtype=float, sep=' ')
    f.close()
    

    fea = MotorFEA(mesh_file="mesh_files/motor_mesh_1", 
                            old_edge_coords=old_edge_coords)
    fea.edge_deltas = 0.1*edge_deltas
    fea.iq.assign(Constant(float(iq)))
    f = open('u_air_gap_coords_1.txt', 'r+')
    u_air_gap_coords = np.fromstring(f.read(), dtype=float, sep=' ')
    f.close()
    
    fea.u_air_gap_indices = fea.locateAzIndices(u_air_gap_coords)
   
    sim = Simulator(AzAirGapModel(fea=fea))
    from matplotlib import pyplot as plt
    print("CSDL: Running the model...")
#    fea.solveMeshMotion()   
    fea.solveMagnetostatic()
    sim['u'] = fea.u.vector().get_local()
    sim.run()
    print("- "*30)
    print("Nodal evaluations of u in the air gap:")
    print("- "*30)
    print(sim['u_air_gap'])
    print('length of u_air_gap: ',len(sim['u_air_gap']))

    print("CSDL: Running check_partials()...")
    # sim.check_partials()


        
        
        
        
        
        
        
        
        
        
        
        
        
