def SPS_Pb_beam(Setup):
    
    #RFTrackPath = '/Users/dgamba/CERN/COOLING/Software/RF-Track2.0'
    #RFTrackPath = '/home/agson/cernbox2/Agnieszka/inter-rf-track'
    RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
    
    import numpy as np
    import sys
    sys.path.append(RFTrackPath)
    import RF_Track as RFT
    import pickle

    with open('twiss.pkl', 'rb') as f:
        twiss = pickle.load(f)   

    Nions = 5e8; #% total number of ions circulating in the ring (not important if
                 #% we ignore other collective effects)

    N_particles = 10000; #% number of simulated macroparticles
    
    Lcirc = twiss['circumference']; #% m, SPS circumference length


    #%% Twiss parameters
    Beta = Setup.Cooler_beta; #% m, desired beta function at the cooler
    Beta_y = Setup.Cooler_beta_y; #% m, desired beta function at the cooler
    L_EC = Setup.Cooler_L; #% m

    BunchT = RFT.Bunch6d_twiss();
    BunchT.emitt_x = Setup.Ions_emitt; #% mm.mrad
    BunchT.emitt_y = Setup.Ions_emitt; #% mm.mrad
    BunchT.alpha_x = (L_EC/2) / Beta; #% converging beam
    BunchT.alpha_y = (L_EC/2) / Beta_y; #%
    BunchT.beta_x = Beta + (L_EC/2)**2 / Beta; #% m
    BunchT.beta_y = Beta_y + (L_EC/2)**2 / Beta_y; #% m


    #%% Create bunch
    B0 = RFT.Bunch6d(Setup.Ions_mass, Nions, Setup.Ions_Q, Setup.Ions_P, BunchT, N_particles);

    X_XP_Y_YP = B0.get_phase_space('%x %xp %y %yp');# % transverse phase space

    #Setting starting parametters for chosen ions
    #X_XP_Y_YP[1][1]=5
    #X_XP_Y_YP[1][0]=4
    #
    # X_XP_Y_YP[0][0]=0
    # X_XP_Y_YP[0][1]=0
    # X_XP_Y_YP[0][2]=1
    # X_XP_Y_YP[0][3]=0
    #
    #
    # X_XP_Y_YP[1][0]=0
    # X_XP_Y_YP[1][1]=0
    # X_XP_Y_YP[1][2]=1
    # X_XP_Y_YP[1][3]=0

    #%% add Gaussian energy spread and uniform longitidinal distribution + dispersion
    np.random.seed(seed=12345)
    D=np.random.randn(N_particles,1)* Setup.Ions_Pspread/100

    spread = 1 + D ; #% momentum spread factor
    X_XP_Y_YP[:,0]=(D*Setup.Dx*1000)[:,0] +X_XP_Y_YP[:,0] #Dispersion
    Vz = B0.get_phase_space('%Vz') * spread; #% c, apply the momentum spread to Vz

    T = np.random.rand(N_particles, 1) * Lcirc * 1e3 / Vz; #% mm/c, uniform distribution
    P = Setup.Ions_P * spread; #% MeV/c, apply the momentum spread to P

    #%% final bunch
    B0 = RFT.Bunch6d(Setup.Ions_mass, Nions, Setup.Ions_Q, np.hstack((X_XP_Y_YP, T, P )));

    return B0
