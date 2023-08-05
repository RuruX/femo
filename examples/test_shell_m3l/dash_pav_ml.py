import lsdo_dash.api as ld

# from tc2_main_script import caddee, system_m3l_model, rep
# from run_pav_new_aeroelastic_opt_viz import caddee_viz, index_functions_map, rep, wing

from run_pav_shell_ml import caddee_viz, index_functions_map, index_functions_surfaces, rep, wing_component, pav_geom_mesh, system_model_name

class TC2DB(ld.DashBuilder):

    def define(self):
        max_thickness = 0.0006
        min_thickness = 0.0001

        max_stress = 2.3e8
        min_stress = -3.6e7

        # max_force = 0.082407
        # min_force = -10.034668 
        # max_force = None
        # min_force = None
        max_force = 1.0
        min_force = -4.0
        # Optimization frame
        geo_frame = self.add_frame(
            'optimization',
            height_in=10,
            width_in=13,
            ncols=2,
            nrows = 2,
            wspace=0.4,
            hspace=0.4,
        )
        
        # geo_frame[0,0] = ld.default_plotters.build_historic_plotter(
        #     [system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress'],
        #     title = 'Maximum stress [Pa]',
        #     legend = False,
        # )
        # Try SNOPT here
        geo_frame[0, 0] = ld.default_plotters.build_SNOPT_plotter(show_merit = False)

        geo_frame[1,0] = ld.default_plotters.build_historic_plotter(
            [system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass'],
            title = 'Mass [kg]',
            legend = False,
        )
        # geo_frame[0,1] = ld.default_plotters.build_historic_plotter(
        #     [system_model_name+'Wing_rm_shell_model.rm_shell.compliance_model.compliance'],
        #     title = 'Compliance',
        #     legend = False,
        # )
        geo_frame[0,1] = ld.default_plotters.build_historic_plotter(
            [system_model_name+'Wing_rm_shell_displacement_map.wing_shell_tip_displacement'],
            title = 'Tip displacement [m]',
            legend = False,
        )
        geo_frame[1,1] = ld.default_plotters.build_historic_plotter(
            [system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress'],
            title = 'Stress [Pa]',
            legend = False,
        )

        # Main Geometry Frame
        geo_frame = self.add_frame(
            'geometry_frame',
            height_in=14.,
            width_in=22.,
            ncols=220,
            nrows = 140,
            wspace=0.4,
            hspace=0.4,
        )

        # geometry settings
        center_x = 4  # eyeballing center x coordinate of geometry
        center_z = 1  # eyeballing center z coordinate of geometry
        center_y = 0
        camera_settings = {
            'pos': (-6.4, -6.4, 8.8),
            # 'pos': (-26, -26, 30),
            'viewup': (0, 0, 1),
            # 'focalPoint': (center_x+8, 0+6, center_z-15)
            'focalPoint': (center_x, center_y+1, center_z-3.0)
        }
        # geo_frame[30:60,0:30] = caddee_viz.build_geometry_plotter(show = False, camera_settings = camera_settings)

        # Delete upper and lower panel surfaces from thickness plot
        upper_lower_panel_surfaces = []
        upper_panel_surfaces = []
        lower_panel_surfaces = []
        other_surfaces = []
        for primitive_name in index_functions_map['wing_thickness'].coefficients:
            if 'panel' in primitive_name:
                upper_lower_panel_surfaces.append(primitive_name)

                if 't_panel' in primitive_name:
                    upper_panel_surfaces.append(primitive_name)
                elif 'b_panel' in primitive_name:
                    lower_panel_surfaces.append(primitive_name)
            else:
                other_surfaces.append(primitive_name)

        wing_surfaces = []
        for primitive_name in wing_component.get_primitives():
            wing_surfaces.append(primitive_name)
        for name in caddee_viz.caddee.system_representation.spatial_representation.primitives.keys():
            if "spar" in name:
                wing_surfaces.append(name)
            elif "panel" in name:
                wing_surfaces.append(name)
            elif "rib" in name:
                wing_surfaces.append(name)

        # Exploded view settings
        def location_callback(surface_name, plot_points):
            # plot_points[:,:,1] = plot_points[:,:,1]

            if surface_name in upper_panel_surfaces:
                plot_points[:,:,2] += 0.7
            elif surface_name in lower_panel_surfaces:
                plot_points[:,:,2] -= 0.8

        # Add in all the geometry stuff
        geo_elements = []
        geo_elements.append(caddee_viz.build_geometry_plotter(show = 0, remove_primitives=wing_surfaces))
        # geo_elements.append(caddee_viz.build_state_plotter(index_functions_map['wing_displacement'], rep=rep, displacements=20))
        
        geo_elements.append(caddee_viz.build_state_plotter(
            index_functions_map['wing_cp'], 
            rep=rep,
            remove_primitives=index_functions_surfaces['valid_surfaces_ml_left_wing'],
            grid_num = 15,
            vmin = min_force, 
            vmax = max_force,
            ))
        
        geo_elements.append(build_combined_plotter(
            caddee_viz,
            index_functions_map['wing_thickness'], 
            index_functions_map['wing_displacement'],
            rep=rep,
            remove_primitives=[],#upper_lower_panel_surfaces,
            displacement_factor=10,
            vmin = min_thickness,
            vmax = max_thickness,
            location_callback = location_callback
            ))

        geo_frame[0:140,0:140] = caddee_viz.build_vedo_renderer(
            geo_elements,
            camera_settings = camera_settings,
            show = 0,
            x_size=1.0,
            y_size=1.0,
        )


        # Three smaller plots
        def add_axes(ax_subplot, data_dict, data_dict_history):
            import matplotlib
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 14})
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_force, vmax=max_force), cmap='coolwarm'), cax = ax_subplot)
            ax_subplot.set_title('C_P')
        geo_frame[10:38, 214:220] = ld.BaseAxesPlotter(
            plot_function = add_axes,
        )
        camera_settings = {
            'pos': (-2.3,-2.25,4.5),
            'viewup': (0, 0, 1),
            'focalPoint': (3, -2.25, 0.8)
        }
        geo_elements = []
        geo_elements.append(caddee_viz.build_state_plotter(
            index_functions_map['wing_cp'],
            rep=rep,
            remove_primitives=index_functions_surfaces['valid_surfaces_ml_right_wing'],
            grid_num = 15,
            vmin = min_force,
            vmax = max_force,
            ))

        geo_frame[0:48, 135:220] = caddee_viz.build_vedo_renderer(
            geo_elements,
            camera_settings = camera_settings,
            show = 0,
            x_size=1.0,
            y_size=1.0,
        )

        # =-=-=-=-=-=-=-=-=- Thickness =-=-=-=-=-=-=-=-=-
        def add_axes(ax_subplot, data_dict, data_dict_history):
            import matplotlib
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 14})
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_thickness, vmax=max_thickness), cmap='coolwarm'), cax = ax_subplot)
            ax_subplot.set_title('Thickness')
        
        geo_frame[58:85, 214:220] = ld.BaseAxesPlotter(
            plot_function = add_axes,
        )

        geo_elements = []
        geo_elements.append(build_combined_plotter(
            caddee_viz,
            index_functions_map['wing_thickness'], 
            index_functions_map['wing_displacement'],
            rep=rep,
            remove_primitives=[],#upper_lower_panel_surfaces,
            displacement_factor=10,
            vmin = min_thickness,
            vmax = max_thickness,
            ))
        geo_frame[48:95, 135:220] = caddee_viz.build_vedo_renderer(
            geo_elements,
            camera_settings = camera_settings,
            show = 0,
            x_size=1.0,
            y_size=1.0,
        )
        
        # =-=-=-=-=-=-=-=-=- STRESS =-=-=-=-=-=-=-=-=-
        def add_axes(ax_subplot, data_dict, data_dict_history):
            import matplotlib
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 14})
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_stress, vmax=max_stress), cmap='coolwarm'), cax = ax_subplot)
            ax_subplot.set_title('Stress [Pa]')
        
        geo_frame[105:130, 214:220] = ld.BaseAxesPlotter(
            plot_function = add_axes,
        )

        geo_elements = []
        geo_elements.append(build_combined_plotter(
            caddee_viz,
            index_functions_map['wing_stress'], 
            index_functions_map['wing_displacement'],
            rep=rep,
            remove_primitives=[],#upper_lower_panel_surfaces,
            displacement_factor=10,
            vmin = min_stress,
            vmax = max_stress,
            ))

        geo_frame[95:140, 135:220] = caddee_viz.build_vedo_renderer(
            geo_elements,
            camera_settings = camera_settings,
            show = 0,
            x_size=1.0,
            y_size=1.0,
        )



base_color = (115, 147, 179)
import numpy as np
import vedo
def build_combined_plotter(
        self,
        index_function,
        displacement_index_function,
        grid_num = 10,
        show = False,
        displacement_factor = 100,
        color = base_color,
        opacity = 1.0,
        rep = None,
        remove_primitives = [],
        vmin = None,
        vmax = None,
        location_callback = None,
    ):
    surface_names  = list(index_function.coefficients.keys())

    for name in displacement_index_function.coefficients.keys():
        if name not in surface_names:
            raise ValueError(f'{name} not in {surface_names}')
    color  = color
    index = 0
    from csdl import GraphRepresentation
    if isinstance(rep, GraphRepresentation):
        names_to_save = []
        surface_names_temp = set(surface_names)
        surface_names = []
        rep = rep
        for unpromoted_name in rep.unpromoted_to_promoted:
            remove_surf_names = set()
            for name in surface_names_temp:
                if index_function.coefficients[name].name in unpromoted_name:
                    remove_surf_names.add(name)
                    names_to_save.append(unpromoted_name)
                    surface_names.append(name)
                    # print(unpromoted_name)
            
            for remove_name in remove_surf_names:
                surface_names_temp.remove(remove_name)

        names_to_save_displacements = []
        surface_names_temp = set(surface_names)
        surface_names_displacements = []
        for unpromoted_name in rep.unpromoted_to_promoted:
            remove_surf_names = set()
            for name in surface_names_temp:
                if displacement_index_function.coefficients[name].name in unpromoted_name:
                    remove_surf_names.add(name)
                    names_to_save_displacements.append(unpromoted_name)
                    surface_names_displacements.append(name)
                    # print(unpromoted_name)
            
            for remove_name in remove_surf_names:
                surface_names_temp.remove(remove_name)
    # exit()
    transfer_para_mesh = []
    surface_names_to_indices_dict = {}
    end_index = 0
    for name in surface_names:
        start_index = end_index
        for u in np.linspace(0,1,grid_num):
            for v in np.linspace(0,1,grid_num):
                transfer_para_mesh.append((name, np.array([u,v]).reshape((1,2))))
                end_index = end_index + 1
        surface_names_to_indices_dict[name] = (start_index,end_index)
    
    def plot_func(data_dict):
        coefficients = {}
        coefficients_displacements = {}
        for i, name in enumerate(surface_names):
            sim_name = names_to_save[i]
            coefficients[name] = data_dict[sim_name]
        for i, name in enumerate(surface_names_displacements):
            sim_name = names_to_save_displacements[i]
            coefficients_displacements[name] = data_dict[sim_name]
        evaluated_states = index_function.compute(transfer_para_mesh, coefficients)
        evaluated_states_displacements = displacement_index_function.compute(transfer_para_mesh, coefficients_displacements)
        num_states_per_point = evaluated_states.shape[-1]

        system_representation = self.caddee.system_representation
        spatial_rep = system_representation.spatial_representation
        locations = spatial_rep.evaluate_parametric(transfer_para_mesh).value
        locations += evaluated_states_displacements*displacement_factor

        # if location_callback is not None:
        # locations[:,1] = -locations[:,1]
        # x = locations[:,0]
        # y = locations[:,1]
        # z = locations[:,2]

        plotting_elements = []
        # v = np.linalg.norm(evaluated_states, axis=1)
        v = (evaluated_states)

        if num_states_per_point != 1:
            print('states per locoation is not a scalar. taking norm...')
            v = np.linalg.norm(evaluated_states, axis=1)

        # print(v.shape, index_function.name)

        if vmin is None:
            min_v = np.min(v)
        else:
            min_v = vmin
        if vmax is None:
            max_v = np.max(v)
        else:
            max_v = vmax

        # print(np.min(v), np.max(v))

        for surf_num, (name, (start_index, end_index) )in enumerate(surface_names_to_indices_dict.items()):
            
            if name in remove_primitives:
                continue
            # if surf_num > 0:
            #     break
            color_map = []
            v_reshaped = v[start_index:end_index].reshape((grid_num,grid_num))
            vertices = []
            faces = []
            reshaped_plot_points = locations[start_index:end_index].reshape((grid_num,grid_num,3))
            if location_callback is not None:
                location_callback(name, reshaped_plot_points)

            for i in range(grid_num):
                for ii in range(grid_num):
                    vertex = tuple(reshaped_plot_points[i,ii,:])
                    vertices.append(vertex)

                    if i != 0 and ii != 0:
                        num_pts = grid_num
                        face = tuple((
                            (i-1)*num_pts+(ii-1),
                            (i-1)*num_pts+(ii),
                            (i)*num_pts+(ii),
                            (i)*num_pts+(ii-1),
                        ))
                        faces.append(face)
                        # print(face, vertex)
            # print(len(vertices), len(faces))
                    color_map.append(v_reshaped[i,ii])

            mesh = vedo.Mesh([vertices, faces], c = color).opacity(opacity)
            mesh.cmap('coolwarm', color_map, vmin = min_v, vmax = max_v)
            plotting_elements.append(mesh)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Geometry', axes=4, viewup="z", interactive=True)
        else:
            return plotting_elements
    return (names_to_save+names_to_save_displacements, plot_func)


if __name__ == '__main__':
    # Build a standard dashbaord object
    dash_object = TC2DB().assemble_basedash()

    # uncomment to produces images for all frames
    # dash_object.visualize()

    # uncomment to produces images for n_th frame
    # n = 10
    # dash_object.visualize(frame_ind = n, show = True)

    # uncomment to produces image for last frame
    # dash_object.visualize_most_recent(show = True)

    # Visualize during optimization
    # dash_object.visualize_auto_refresh()

    # uncomment to make movie
    dash_object.visualize_all()
    dash_object.make_mov()

    # uncomment to run gui
    # dash_object.run_GUI()
