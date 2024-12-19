def LEIR_Lattice(Setup):
    #RFTrackPath = '/Users/dgamba/CERN/COOLING/Software/RF-Track2.0'
    #RFTrackPath = '/home/agson/cernbox2/Agnieszka/inter-rf-track'
    RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'

    import sys
    import numpy as np
    sys.path.append(RFTrackPath)
    import RF_Track as RFT
    from LEIR_Pb_beam import LEIR_Pb_beam
    from LEIR_Cooler_uniform import LEIR_Cooler_uniform

    Beta_cooler = Setup.Cooler_beta;# % m, desired beta function at the cooler
    Beta_cooler_y = Setup.Cooler_beta_y;# % m, desired beta function at the cooler
    L_EC = Setup.Cooler_L; #% m

    #%% Define cooler
    EC = LEIR_Cooler_uniform(Setup);

    #%% LEIR params
    Lcirc = 25*np.pi; #% m, LEIR circumference length
    L = Lcirc - L_EC; #% m

    LEIR_momentum_compaction = 0.124; #% LEIR momentum compaction
    Qx = 1.82; #% 2pi, tune x
    Qy = 2.72; #% 2pi, tune y
    DQx = -0.5; #% 2pi, chromaticity x
    DQy = -1.0; #% 2pi, chromaticity y

    #% Twiss parameters at the cooler start
    alpha_x = (L_EC/2) / Beta_cooler; #% converging beam
    alpha_y = (L_EC/2) / Beta_cooler_y; #%
    beta_x = Beta_cooler + (L_EC/2)**2 / Beta_cooler; #% m
    beta_y = Beta_cooler_y + (L_EC/2)**2 / Beta_cooler_y; #% m

    #% Crate one matching transfer matrix for the rest of the ring
    TWISS = np.zeros((2, 11))
    TWISS[0,:] = [ 0, beta_x, -alpha_x, 0,  beta_y, -alpha_y, 0 , Setup.Dx, 0, 0, 0]; #% cooler end
    TWISS[1,:] = [ L, beta_x,  alpha_x, Qx, beta_y,  alpha_y, Qy, Setup.Dx, 0, 0, 0]; #% cooler start

    RING = RFT.TransferLine(TWISS, DQx, DQy, LEIR_momentum_compaction, Setup.Ions_P);

    #%% Create the lattice
    L = RFT.Lattice()
    L.append(EC)
    L.append(RING)

    return L
