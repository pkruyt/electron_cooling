def SPS_Lattice():
    #RFTrackPath = '/Users/dgamba/CERN/COOLING/Software/RF-Track2.0'
    #RFTrackPath = '/home/agson/cernbox2/Agnieszka/inter-rf-track'
    RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'

    import sys
    import numpy as np
    sys.path.append(RFTrackPath)
    import RF_Track as RFT
    #from SPS_Pb_beam import SPS_Pb_beam
    import pickle

    #Beta_cooler = Setup.Cooler_beta;# % m, desired beta function at the cooler
    #Beta_cooler_y = Setup.Cooler_beta_y;# % m, desired beta function at the cooler
    #L_EC = Setup.Cooler_L; #% m
    with open('twiss.pkl', 'rb') as f:
        twiss = pickle.load(f)   

    #%% SPS params
    # Lcirc = 6911.5038; #% m, SPS circumference length
    # L = Lcirc  #% m
    # p0c=18644000000000.0*1e-6
    # Dx=0
    
    
    # SPS_momentum_compaction = 0.0030769483220676706; #% SPS momentum compaction
    
    # Qx = 20.150000258547372; #% 2pi, tune x
    # Qy = 20.250000001309765; #% 2pi, tune y
    # DQx = -1.3180678214064212; #% 2pi, chromaticity x
    # DQy = -0.7261591831761449; #% 2pi, chromaticity y

    # #% Twiss parameters at the cooler start
    # alpha_x = -1.60081 #% converging beam
    # alpha_y = 0.78249 #%
    # beta_x = 86.5759 #% m
    # beta_y = 38.6715 #% m
    
    # #% Crate one matching transfer matrix for the rest of the ring
    # TWISS = np.zeros((2, 11))
    # #TWISS[0,:] = [ 0, beta_x, -alpha_x, 0,  beta_y, -alpha_y, 0 , Setup.Dx, 0, 0, 0]; #% cooler end
    # TWISS[0,:] = [ 0, beta_x, alpha_x, 0,  beta_y, alpha_y, 0 , Dx, 0, 0, 0]; #% cooler end
    # TWISS[1,:] = [ L, beta_x,  alpha_x, Qx, beta_y,  alpha_y, Qy, Dx, 0, 0, 0]; #% cooler start
    
    Lcirc = twiss['circumference']; #% m, SPS circumference length
    #L = Lcirc  #% m
    p0c=18644000000000.0*1e-6
    
    n_points = len(twiss['alfx'])
    
    
    
    
              
             
    
    
    SPS_momentum_compaction = twiss['momentum_compaction_factor']; #% SPS momentum compaction
    
    Qx = twiss['qx']; #% 2pi, tune x
    Qy = twiss['qy']; #% 2pi, tune y
    DQx = twiss['dqx']; #% 2pi, chromaticity x
    DQy = twiss['dqy']; #% 2pi, chromaticity y

    #% Twiss parameters at the cooler start
    alpha_x = twiss['alfx'] #% converging beam
    alpha_y = twiss['alfy'] #%
    beta_x = twiss['betx']#% m
    beta_y = twiss['bety'] #% m
    
    s = twiss['s']
    mux=twiss['mux']
    muy=twiss['muy']
    
    Dx=twiss['dx']
    Dy=twiss['dy']
    dpx=twiss['dpx']
    dpy=twiss['dpy']
    
    #dx_sum=sum(Dx)
    #dpx_sum=sum(dpx)
    # Dx=(Dx)
    # Dy=(Dy)
    # dpx=[0.141447614,0.141447614]
    #dpy=0
    
    # T= RFT.Bunch6d_twiss()
    
    # T.emitt_x 
    # T.emitt_y
    # T.emitt_z 
    # T.alpha_x
    # T.alpha_y
    # T.alpha_z
    # T.beta_x 
    # T.beta_y 
    # T.beta_z 
    # T.disp_x 
    # T.disp_xp
    # T.disp_y 
    # T.disp_yp
    # T.disp_z 
        
    # # #% Crate one matching transfer matrix for the rest of the ring
    # TWISS = np.zeros((n_points, 11))
    
    
    # for i in range(n_points):
    
    #     TWISS[i,:] = [ s[i], beta_x[i], alpha_x[i], mux[i],  beta_y[i], alpha_y[i], muy[i] , Dx[i], dpx[i], Dy[i], dpy[i]]; #% cooler end
        
        
       #TWISS[1,:] = [ L, beta_x,  alpha_x, Qx, beta_y,  alpha_y, Qy, Dx, 0, 0, 0]; #% cooler start
       
    TWISS = np.zeros((2, 11))    
    TWISS[0,:] = [ 0,     beta_x[0],  alpha_x[0],  0,  beta_y[0],   alpha_y[0], 0 ,  Dx[0],  dpx[0],  Dy[0], dpy[0]]; #% cooler end
    TWISS[1,:] = [ Lcirc, beta_x[-1], alpha_x[-1], Qx, beta_y[-1],  alpha_y[-1], Qy, Dx[-1], dpx[-1], Dy[-1], dpy[-1]]; #% cooler start
    
    # TWISS[0,:] = [ 0, beta_x[0], alpha_x[0], 0,  beta_y[0], alpha_y[0], 0 , 0, 0, 0, 0]; #% cooler end
    # TWISS[1,:] = [ Lcirc, beta_x[-1],  alpha_x[-1], Qx, beta_y[-1],  alpha_y[-1], Qy, dx_sum, dpx_sum, Dy[-1], dpy[-1]]; #% cooler start
    
    # TWISS[0,:] = [ 0, beta_x[0], alpha_x[0], 0,  beta_y[0], alpha_y[0], 0 , 0, 0, 0, 0]; #% cooler end
    # TWISS[1,:] = [ Lcirc, beta_x[-1],  alpha_x[-1], Qx, beta_y[-1],  alpha_y[-1], Qy, 0, 0, 0, 0]; #% cooler start
    
    #TWISS[0,:] = [ 0, beta_x[0],  beta_y[0], alpha_x[0], alpha_y[0],  mux[0], muy[0], Dx[0], Dy[0], dpx[0], dpy[0]]; #% cooler end
    #TWISS[1,:] = [ Lcirc, beta_x[-1],  beta_y[-1], alpha_x[-1], alpha_y[-1],  mux[-1], muy[-1], Dx[-1], Dy[-1], dpx[-1], dpy[-1]]; #% cooler start
    
    
    
    
    RING = RFT.TransferLine(TWISS, DQx, DQy, SPS_momentum_compaction, p0c);

    #%% Create the lattice
    L = RFT.Lattice()
    L.append(RING)

    return L,TWISS
