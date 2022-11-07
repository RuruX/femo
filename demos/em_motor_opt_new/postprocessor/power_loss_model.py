import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator

class LossSumModel(Model):
    def define(self):

        eddy_current_loss = self.declare_variable('eddy_current_loss')
        hysteresis_loss = self.declare_variable('hysteresis_loss')

        loss_sum = eddy_current_loss + hysteresis_loss

        loss_sum = self.register_output(
            name='loss_sum',
            var=loss_sum
        )

        self.print_var(loss_sum)


class PowerLossModel(Model):
    def define(self):

        # # COPPER LOSS
        # winding_area        = self.declare_variable('winding_area')
        # num_windings        = self.declare_variable('num_windings')
        # slot_fill_factor    = self.declare_variable('slot_fill_factor')

        # wire_radius         = ((winding_area * slot_fill_factor / num_windings) / np.pi) ** 0.5
        # wire_radius         = self.register_output(
        #     name='wire_radius',
        #     var=wire_radius
        # )

        # copper_resistivity      = self.declare_variable('copper_resistivity')
        # copper_permeability     = self.declare_variable('copper_permeability', val=1.256629e-6)
        frequency               = self.declare_variable('frequency', val=1.)
        # wire_resistance         = self.declare_variable('wire_resistance')
        # current_amplitude       = self.declare_variable('current_amplitude')

        # wire_skin_depth     = (copper_resistivity / (np.pi * frequency * copper_permeability))**(0.5)
        # wire_skin_depth     = self.register_output('wire_skin_depth', wire_skin_depth)

        # wire_resistance_AC  = wire_resistance / ((2*wire_skin_depth/wire_radius) - (wire_skin_depth/wire_radius)**(0.5))
        # wire_resistance_AC  = self.register_output('wire_resistance_AC', wire_resistance_AC)

        # # copper_loss     = 3 * (current_amplitude / np.sqrt(2))**2 * wire_resistance_AC

        # copper_loss     = 3 * (current_amplitude / np.sqrt(2))**2 * wire_resistance

        # copper_loss     = self.register_output(
        #     name='copper_loss',
        #     var=copper_loss
        # )

        # EDDY CURRENT LOSS
        lamination_thickness    = self.declare_variable('lamination_thickness', val=0.2e-3)
        avg_flux_influence_ec   = self.declare_variable('B_influence_eddy_current')
        steel_conductivity      = self.declare_variable('steel_conductivity', val=2.17e6) # grain oriented electrical steel
        steel_area              = self.declare_variable('steel_area')
        motor_length            = self.declare_variable('motor_length')
        # eddy_current_loss = np.pi**2 / 6 * motor_length * avg_flux_influence_ec * frequency**2 * \
        #     lamination_thickness**2 * steel_conductivity
        eddy_current_loss = 2 * np.pi**2 * frequency**2 * motor_length * avg_flux_influence_ec * 0.07

        eddy_current_loss = self.register_output(
            name='eddy_current_loss',
            var=eddy_current_loss
        )

        # HYSTERESIS LOSS
        steel_hysteresis_coeff  = self.declare_variable('steel_hysteresis_coeff', val=1.91)
        avg_flux_influence_h    = self.declare_variable('B_influence_hysteresis')
        hysteresis_coeff        = self.declare_variable('hysteresis_coeff', val=55.)

        # hysteresis_loss = steel_hysteresis_coeff * motor_length * avg_flux_influence_h * frequency
        hysteresis_loss = 2*np.pi*frequency*hysteresis_coeff*motor_length*avg_flux_influence_h

        hysteresis_loss = self.register_output(
            name='hysteresis_loss',
            var=hysteresis_loss
        )

        # # WINDAGE LOSSES
        # air_density         = self.declare_variable('air_density', val=1.204)
        # air_viscosity       = self.declare_variable('air_viscosity', val=1.825e-5)
        # rotor_radius_o      = self.declare_variable('rotor_radius_outer', val = 80.e-3)
        # stator_radius_i     = self.declare_variable('stator_radius_inner', val = 81.e-3)
        # omega               = self.declare_variable('omega')

        # air_gap_depth       = self.register_output(
        #     name='air_gap_depth',
        #     var=stator_radius_i - rotor_radius_o
        # )

        # azimuthal_vel       = self.create_output(
        #     name='azimuthal_vel',
        #     shape=(1,)
        # )

        # azimuthal_vel[0]    = rotor_radius_o**2 * omega / air_gap_depth / (stator_radius_i**2 - rotor_radius_o**2) * \
        #     (stator_radius_i**2 * csdl.log(stator_radius_i/rotor_radius_o) - 0.5*(stator_radius_i**2 - rotor_radius_o**2))

        # air_gap_Re          = air_density * azimuthal_vel * air_gap_depth / air_viscosity
        # air_gap_Re          = self.register_output(
        #     name='air_gap_Re',
        #     var=air_gap_Re
        # )

        # air_gap_Ta  = air_gap_Re**2 * (air_gap_depth/rotor_radius_o)
        # air_gap_Ta  = self.register_output(
        #     name='air_gap_Ta',
        #     var=air_gap_Ta
        # )

        # friction_coeff      = 0.0152 /  (air_gap_Re)**(0.24)
        # friction_coeff = self.register_output(
        #     name='friction_coeff',
        #     var=friction_coeff
        # )

        # # windage_loss        = np.pi * air_density * azimuthal_vel**2 * rotor_radius_o**2 * friction_coeff * motor_length * omega # UNFINISHED
        # windage_loss        = 2 * np.pi * friction_coeff * air_density * omega**3 * rotor_radius_o**4 * motor_length

        # windage_loss = self.register_output(
        #     name='windage_loss',
        #     var=windage_loss
        # )

        # # BEARING LOSSES


        # # STRAY LOSSES
        # avg_input_power     = self.declare_variable('avg_input_power')
        # stray_loss          = avg_input_power * 0.01

        # stray_loss          = self.register_output(
        #     name='stray_loss',
        #     var=stray_loss
        # )



if __name__ == '__main__':
    aaa = PowerLossModel()
    sim = py_simulator(aaa)

    sim.run()

    print(sim['eddy_current_loss'])
    sim.check_partials(compact_print=True)
