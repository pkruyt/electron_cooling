import RF_Track as RFT
import sys
import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
sys.path.append(RFTrackPath)


def XSUITE_TO_RF_converter(particles):
    """The desired variables that are needed for a beam in RF Track are:

        1. X
        2. XP
        3. Y
        4. YP
        5. T
        6. P
        7. M
        8. Q             
        """
    ###########################################################################
    """These parameters are needed to compute to corresponding variables in RF Track"""
    p0c = particles.p0c[0]
    m_ion = particles.mass0
    q0 = particles.q0
    #beta=particles.beta0[0]
    n_particles = len(particles.particle_id)


    p_tot = (particles.delta*p0c+p0c)
    Px = particles.px*p0c
    Py = particles.py*p0c
    Pz2 = (p_tot)**2-(Px)**2-(Py)**2
    Pz = np.sqrt(Pz2) 
    
    gamma_particles = np.sqrt(1 + (Pz/m_ion)**2)  # ion relativistic factor
    # gamma_particles=np.sqrt( 1 + (p_tot/m_ion)**2 ) # ion relativistic factor
    beta_particles = np.sqrt(1-1/(gamma_particles*gamma_particles))  # ion beta
        
    gamma_ref = np.sqrt(1 + (p0c/m_ion)**2)  # ion relativistic factor
    beta_ref = np.sqrt(1-1/(gamma_ref*gamma_ref))  # ion beta
           
    accumulated_length = particles.s
    ###########################################################################
    """Direct calculation of the corresponding variables in RF Track:"""

    # X
    X = particles.x * 1e3 #mm
    # XP
    ratio_x = Px/Pz
    angle_x = np.arctan(ratio_x)*1e3
    # Y
    Y = particles.y * 1e3 #mm
    # YP
    ratio_y = Py/Pz
    angle_y = np.arctan(ratio_y)*1e3
    # T
    t_ref=accumulated_length*1e3/beta_ref
    t_tot = (0-particles.zeta)/(beta_ref) #arrival time in m/c
    t = (t_tot)*1e3 + t_ref #mm/c
    # P
    P = p_tot*1e-6 #Mev/c
    # M
    mass = m_ion*np.ones(n_particles)*1e-6 #Mev/c^2
    # Q
    q = q0*np.ones(n_particles) #e
    ###########################################################################
    """Combine into one array and build beam in RF Track"""
    
    arr_ref = np.column_stack(((0, 0, 0, 0, t_ref[0], p0c*1e-6, mass[0], q[0])))
    arr = np.column_stack(((X, angle_x, Y, angle_y, t, P, mass, q)))
    
    arr=np.vstack([arr_ref,arr])
    beam = RFT.Bunch6d(arr)
    

    return beam
