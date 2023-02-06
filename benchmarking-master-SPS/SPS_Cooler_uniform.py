import math
import numpy as np

#RFTrackPath = '/Users/dgamba/CERN/COOLING/Software/RF-Track2.0'
#RFTrackPath = '/home/agson/cernbox2/Agnieszka/inter-rf-track'
RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'

def deek(r, r_0, I, beta):
  '''kinetic energy variation as a function of radius
  '''
  return 1.2*10**-4 * I/beta**3 * (r/r_0)**2

def dpp(r, r_0, I, beta):
  '''momentum variation as a function of radius
  '''
  return deek(r, r_0, I, beta)*(1-np.sqrt(1-beta**2))/beta**2

def SPS_Cooler_uniform(Setup):
    import sys
    sys.path.append(RFTrackPath)
    import RF_Track as RFT

    Nx = 20;
    Ny = 20;
    Nz = 2;

    E = RFT.ElectronCooler(Setup.Cooler_L, Setup.Cooler_r0/1e3, Setup.Cooler_r0/1e3);
    E.set_Q(-1); #% electron charge
    E.set_static_Bfield(0.0, 0.0, Setup.Cooler_B);
    E.set_temperature(Setup.Cooler_Tr, Setup.Cooler_Tl);


    if (Setup.Neutralisation):
        #For constant electron velocity
        E.set_electron_mesh(Nx, Ny, Nz, Setup.Cooler_Ne, 0, 0, Setup.Cooler_Vele);

    else:
        density = Setup.Cooler_Ne * np.ones([Nx, Ny]);
        Vx = np.zeros([Nx, Ny]);
        Vy = np.zeros([Nx, Ny]);


        Beta=Setup.Cooler_Vele
        Gamma=1/np.sqrt(1-Beta**2)
        BetaGamma=Beta*Gamma


        NX=np.linspace(-Setup.Cooler_r0/1e3,Setup.Cooler_r0/1e3,Nx)
        NY=np.linspace(-Setup.Cooler_r0/1e3,Setup.Cooler_r0/1e3,Ny)
        NX,NY = np.meshgrid(NX, NY);
        R = np.sqrt(NX**2 + NY**2)
        Vz=np.zeros((Nx,Ny))

        for i in range(0,Nx):
            for j in range(0,Ny):
                r=R[i][j]
                Vz[i][j]=((dpp(r, Setup.Cooler_r0/1e3, Setup.I, Beta))*BetaGamma+BetaGamma)/Gamma

        E.set_electron_mesh(Nz, density ,Vx ,Vy,Vz )

    return E
