import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)
import RF_Track as RFT


def RF_TO_XSUITE_converter(B0):
    """The desired variables that are needed for a beam in Xsuite are:
        
        1. x
        2. px
        3. y
        4. py
        5. zeta
        6. delta         
        """
    ########################################################################### 
    """These parameters are needed to compute to corresponding variables in RF Track"""
    

   
    context = xo.ContextCpu()
    #buf = context.new_buffer()
    
        
    beam=B0.get_phase_space("%x %Px  %y %Py %t %P")
    beam2=B0.get_phase_space("%m %Q  %y %Py %t %P")
            
    p0c=beam[:,5][0]*1e6
    q0=beam2[:,1][0]
    mass0=beam2[:,0][0]*1e6

    gamma = np.sqrt( 1 + (p0c/mass0)**2 ) # ion relativistic factor
    beta0 = np.sqrt(1-1/(gamma*gamma)) # ion beta
    
    ###########################################################################
    #x
    x = beam[:,0]*1e-3
    #px
    Px = beam[:,1]
    px = Px*1e6/p0c
    #y
    y = beam[:,2]*1e-3
    #py
    Py = beam[:,3]
    py = Py*1e6/p0c
    #z
    t = beam[:,4] 
    S=beam[0,4]*1e-3
    accumulated_length = [S]*len(x) 
    zeta=accumulated_length-(beta0*(t)*1e-3) # according to definition from https://github.com/xsuite/xsuite/issues/8
    #delta        
    P = beam[:,5]*1e6
    delta = (P-p0c)/p0c
    ###########################################################################
    """Build into one beam in Xsuite"""

    particles = xp.Particles(_context=context,
            mass0=mass0, q0=q0, p0c=p0c, 
            x=x, px=px, y=y, py=py,
            zeta=zeta, delta=delta)
    
    particles.s=S
       
    particles=particles.filter(particles.x!=0)
    
    return particles



