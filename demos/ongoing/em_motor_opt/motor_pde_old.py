"""
Definition of the variational form of the motor problem
"""
from fe_csdl_opt.fea.fea_dolfinx import *
from permeability.piecewise_permeability_Luca import *


exp_coeff = extractexpDecayCoeff()
cubic_bounds = extractCubicBounds()
# START NEW PERMEABILITY
def RelativePermeability(subdomain, u, uhat):
    gradu = gradx(u,uhat)
    # if subdomain == 1: # Electrical/Silicon/Laminated Steel
    if subdomain == 1 or subdomain == 2: # Electrical/Silicon/Laminated Steel
        B = as_vector((gradu[1], -gradu[0]))
        norm_B = sqrt(dot(B, B) + DOLFIN_EPS)

        mu = conditional(
            lt(norm_B, cubic_bounds[0]),
            linearPortion(norm_B),
            conditional(
                lt(norm_B, cubic_bounds[1]),
                cubicPortion(norm_B),
                (exp_coeff[0] * exp(exp_coeff[1]*norm_B + exp_coeff[2]) + 1)
            )
        )
    elif subdomain == 3:
        mu = 1.00 # insert value for titanium or shaft material
    elif subdomain >= 4 and subdomain <= 28: # AIR
        mu = 1.0
    elif subdomain >= 29 and subdomain <= 40: # NEODYMIUM
        mu = 1.05
    elif subdomain >= 41: # COPPER
        mu = 1.00

    return mu
# END NEW PERMEABILITY

def compute_i_abc(iq, angle=0.0):
    i_abc = as_vector([
        iq * np.sin(angle),
        iq * np.sin(angle - 2*np.pi/3),
        iq * np.sin(angle + 2*np.pi/3),
    ])
    return i_abc
    
def JS(v,uhat,iq,p,s,Hc,angle):
    """
    The variational form for the source term (current) of the
    Maxwell equation
    """
    Jm = 0.
    gradv = gradx(v,uhat)
    base_magnet_dir = 2 * np.pi / p / 2
    magnet_sweep    = 2 * np.pi / p
    for i in range(p):
        flux_angle = base_magnet_dir + i * magnet_sweep
        Hx = (-1)**(i) * Hc * np.cos(flux_angle + angle*2/p)
        Hy = (-1)**(i) * Hc * np.sin(flux_angle + angle*2/p)

        H = as_vector([Hx, Hy])

        curl_v = as_vector([gradv[1],-gradv[0]])
        Jm += inner(H,curl_v)*dx(i + 4 + p*2 + 1)

    num_phases = 3
    num_windings = s
    coil_per_phase = 2
    stator_winding_index_start  = 4 + 3 * p + 1
    stator_winding_index_end    = stator_winding_index_start + num_windings
    Jw = 0.
    i_abc = compute_i_abc(iq, angle)
    JA, JB, JC = i_abc[0] + DOLFIN_EPS, i_abc[1] + DOLFIN_EPS, i_abc[2] + DOLFIN_EPS

    # NEW METHOD
    # for i in range(int((num_windings) / (num_phases * coil_per_phase))):
    #     coil_start_ind = i * num_phases * coil_per_phase
        
    #     J_list = [
    #         JB * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind),
    #         JA * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 1),
    #         JC * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind + 2),
    #         JB * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 3),
    #         JA * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind + 4),
    #         JC * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 5)
    #     ]
    #     Jw += sum(J_list)

    coils_per_pole  = 3
    for i in range(p): # assigning current densities for each set of poles
        coil_start_ind  = stator_winding_index_start + i * coils_per_pole
        coil_end_ind    = coil_start_ind + coils_per_pole

        J_list = [
            JB * (-1)**(i+1) * v * dx(coil_start_ind),
            JA * (-1)**(i) * v * dx(coil_start_ind + 1),
            JC * (-1)**(i+1) * v * dx(coil_start_ind + 2),
        ]

        Jw += sum(J_list)

    return Jm + Jw

def pdeResEM(u,v,uhat,iq,dx,p,s,Hc,vacuum_perm,angle):
    """
    The variational form of the PDE residual for the magnetostatic problem
    """
    res = 0.
    gradu = gradx(u,uhat)
    gradv = gradx(v,uhat)
    num_components = 4 * 3 * p + 2 * s
    for i in range(num_components):
        res += 1./vacuum_perm*(1/RelativePermeability(i + 1, u, uhat))\
                *dot(gradu,gradv)*J(uhat)*dx(i + 1)
    res -= JS(v,uhat,iq,p,s,Hc,angle)
    return res