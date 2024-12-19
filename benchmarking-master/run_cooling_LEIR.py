#python3.8 run_cooling_LEIR.py dir 1(disp_val)
#RFTrackPath = '/Users/dgamba/CERN/COOLING/Software/RF-Track2.0'
#RFTrackPath = '/home/agson/cernbox2/Agnieszka/inter-rf-track'
RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
from LEIR_Pb_beam import LEIR_Pb_beam
from LEIR_Cooler_uniform import LEIR_Cooler_uniform
from LEIR_Lattice import LEIR_Lattice
import sys
import os
import math
import numpy as np
import datetime
sys.path.append(RFTrackPath)
import RF_Track as RFT

# Specify where results should be stored:
outdir = './results/'
if not os.path.exists(outdir):
    os.system('mkdir -p '+outdir)

# os.environ["DEBUG"] = "1"
begin_time = datetime.datetime.now()
print(datetime.datetime.now())

Ions_A = 207.2; #% atomic number
Ions_Q = 54;# % ion charge
Ions_mass = Ions_A * RFT.protonmass; #% MeV/c^2
#print('Ion mass:',Ions_mass)
Ions_K = 4.2 * Ions_A; #% MeV, initial kinetic energy
Ions_E = Ions_mass + Ions_K;# % MeV
Ions_P = np.sqrt(Ions_E**2 - Ions_mass**2); #% MeV/c
Ions_Pspread =0.025 #% percent
Ions_beta_gamma = Ions_P / Ions_mass;
print(Ions_beta_gamma,"Ions_beta_gamma")
Ions_emitt = Ions_beta_gamma *10;# mm.mrad, initial emittance
Ions_x0 = 0; #% mm, closed orbit (orbit at cooler entrance)
Ions_y0 = 0; #% mm, closed orbit (orbit at cooler entrance)
Ions_xp0 = 0; #% mrad, closed orbit (orbit at cooler entrance)
Ions_yp0 = 0;# % mrad, closed orbit (orbit at cooler entrance)

#%% Cooler params
Cooler_Vele = 9.43015738e-02 #% c
Cooler_L = 1.5;# % m, cooler length
Cooler_B = 0.06; #% T 0.07
Cooler_r0 = 24.8;# % mm, electron beam radius
#Cooler_Ne = 8.27e13;# % electron number density I=0.6A
Cooler_Ne = 3.945e+13# 6.8e+13;# % electron number density #/m^3   I= 350mA
I=0.35
Cooler_beta = 1.9; #% m, beta function in the middle of the cooler
Cooler_beta_y = 6.4; #% m, beta function in the middle of the cooler
Cooler_Tr = 0.1; #% eV
Cooler_Tl = 0.01;# % eV
Neutralisation=0 # 0/1 -- 0==OFF, 1==ON
Dx=0 #dispersion m
Cooler_Gamma=(1/(Cooler_beta*1e3))


class Empty:
    pass
Setup=Empty()
#%% Pack all parameters into one structure to carry around
Setup.Ions_mass = Ions_mass; #% MeV/c^2
Setup.Ions_Q = Ions_Q; #% e
Setup.Ions_P = Ions_P; #% MeV/c
Setup.Ions_Pspread = Ions_Pspread; #% percent
Setup.Ions_emitt = Ions_emitt; #% mm.mrad
Setup.Cooler_Ne = Cooler_Ne;# % electron number density #/m^3
Setup.Cooler_Vele = Cooler_Vele; #% c
Setup.Cooler_L = Cooler_L; #% m
Setup.Cooler_B = Cooler_B; #% T
Setup.Cooler_r0 = Cooler_r0; #% mm, electron beam radius
Setup.Cooler_beta = Cooler_beta; #% m
Setup.Cooler_beta_y = Cooler_beta_y; #% m
Setup.Cooler_Tr = Cooler_Tr; #% eV
Setup.Cooler_Tl = Cooler_Tl; #% eV
Setup.Ions_x0 = Ions_x0; #% mm
Setup.Ions_y0 = Ions_y0; #% mm
Setup.Ions_xp0 = Ions_xp0;# % mrad
Setup.Ions_yp0 = Ions_yp0; #% mrad
Setup.Dx=Dx
Setup.I=I
Setup.Neutralisation=Neutralisation# 0/1
#%% useful function to tabulate the relevant quantities
def get_table_row(B, Setup):
    M = B.get_phase_space("%x %Px %y %Py %t %Pz %d");
    P = np.hypot(M[:,1], np.hypot(M[:,3], M[:,5])); #% MeV/c, total momentum
    E = np.hypot(Setup.Ions_mass, np.hypot(M[:,1], np.hypot(M[:,3], M[:,5]))); #% MeV, total energy
    V = P / E; #% c, velocity
    M[:, 1 ] *= 1e3 / Setup.Ions_P;# % mrad, (Px, Py), to (px, py) (canonical coordinates)
    M[:,3 ] *= 1e3 / Setup.Ions_P;# % mrad, (Px, Py), to (px, py) (canonical coordinates)
    TMP_M=M[:,0]
    M[:,0]=-(M[:,6]*Setup.Dx) +M[:,0]


    emitt_x = max(np.linalg.det(np.cov(M[:,0:2].T)),0)**(1./2); #% mm.mrad
    emitt_y = max(np.linalg.det(np.cov(M[:,2:4].T)),0)**(1./2); #% mm.mrad
    emitt4d = max(np.linalg.det(np.cov(M[:,0:4].T)),0)**(1./4); #% mm.mrad
    ABS_x=np.sort(np.abs(M[:,0]))
    #(np.nonzero(ABS_x>10)[0][0]+1)/ABS_x.size -- fraction of particles with x<10
    R = [ emitt_x, emitt_y, emitt4d, np.mean(V), np.std(V), np.mean(P), np.std(P),np.mean(TMP_M),np.std(TMP_M)]#, (np.nonzero(ABS_x>10)[0][0]+1)/ABS_x.size,(np.nonzero(ABS_x>5)[0][0]+1)/ABS_x.size,(np.nonzero(ABS_x>2.5)[0][0]+1)/ABS_x.size ];
    return R


#%%%%%%%%%%%%%% Main part
B0   = LEIR_Pb_beam(Setup);
LEIR = LEIR_Lattice(Setup);

TMP = B0.get_phase_space("%x %Px %y %Py %t %Pz %d")
M = B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E %P")


#Histograms for Schottky plots 
histX, bin_edgesX = np.histogram(M[:,0], range=(-100, 100), bins=400)
histX=np.vstack((bin_edgesX[:-1]+(bin_edgesX[1]-bin_edgesX[0])/2,histX))


histY, bin_edgesY = np.histogram(M[:,2], range=(-100, 100), bins=400)
histY=np.vstack((bin_edgesY[:-1]+(bin_edgesY[1]-bin_edgesY[0])/2,histY))


histP, bin_edgesP = np.histogram(M[:,9], range=(18000, 19000), bins=4000)
histP=np.vstack((bin_edgesP[:-1]+(bin_edgesP[1]-bin_edgesP[0])/2,histP))
TIME_np=np.zeros(0)
TIME_np=np.hstack((TIME_np,0))

#To "track" parameters of chosen ion
Part=B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[2]
Part10=B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[10]
M2 = np.hstack((0,B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[2],Cooler_Gamma*(Part[0]**2)+Cooler_beta*1e3*((Part[1]*1e3 / Setup.Ions_P)**2)))
M10 = np.hstack((0,B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[10],Cooler_Gamma*(Part10[0]**2)+Cooler_beta*1e3*((Part10[1]*1e3 / Setup.Ions_P)**2)))


np.savetxt(outdir+'LEIR_beam_in.dat', np.c_[(M)])

t_ms = 0; #% ms

E = np.hstack(( t_ms, get_table_row(B0, Setup) ));

#Force
#TT=0
#TT0=0
from tqdm import tqdm
for i in tqdm(range(1,10000)):
    
#    if 0 :#% should we want to scan the closed orbit
#        M = B0.get_phase_space();
        #%% set closed orbit
#        M[:][0] -= np.mean(M[:][0]) - Setup.Ions_x0;
#        M[:][2] -= np.mean(M[:][2]) - Setup.Ions_y0;
#        M[:][1] -= np.mean(M[:][1]) - Setup.Ions_xp0;
#        M[:][3] -= np.mean(M[:][3]) - Setup.Ions_yp0;
#        B0.set_phase_space(M);
    #tracking, updating the time variable and upadting the table
    B0 = LEIR.track(B0);
    t_ms = np.mean(B0.get_phase_space("%t")) / RFT.ms;
    E=np.vstack((E,np.hstack(( t_ms, get_table_row(B0, Setup) ))))

# To save information about chosen particles
#    Part=B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[2]
#    Part10=B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E")[10]
#    M2=np.vstack((M2,np.hstack(( t_ms,(Part),Cooler_Gamma*(Part[0]**2)+Cooler_beta*1e3*((Part[1]*1e3 / Setup.Ions_P)**2)))))
#    M10=np.vstack((M10,np.hstack(( t_ms,(Part10),Cooler_Gamma*(Part10[0]**2)+Cooler_beta*1e3*((Part10[1]*1e3 / Setup.Ions_P)**2)))))

    #To save the force -- only with  "os.environ["DEBUG"] = "1""
    # F = np.loadtxt('cooling_force_beam.txt')
    # #TT=np.vstack((TT,(np.sqrt(((F[1,0]) * 1e6)**2+((F[1,1]) * 1e6)**2))))
    # #TT0=np.vstack((TT0,-(F[1,2]) * 1e6))
    # TT0=-(F[:,2]) * 1e6


    #%% Save the table every 500 turns
    if i%500 == 0:
        #np.savetxt(outdir+'LEIR_emitt.dat.gz', np.c_[(E)])
        np.savetxt(outdir+'LEIR_emitt.dat', np.c_[(E)])
        TMP = B0.get_phase_space("%x %Px %y %Py %t %Pz %d")
        M = B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E %P")
        np.savetxt(outdir+'LEIR_beam'+str(i)+'.dat', np.c_[(M)])
        #HISTOGRAMS
        # hist, bin_edges = np.histogram(M[:,0], range=(-100, 100), bins=400)
        # histX=np.vstack((histX,hist))
        #
        # hist3, bin_edges3 = np.histogram(M[:,2], range=(-100, 100), bins=400)
        # histY=np.vstack((histY,hist3))
        #
        # hist2, bin_edges2 = np.histogram(M[:,9], range=(18000, 19000), bins=4000)
        # histP=np.vstack((histP,hist2))
        # TIME_np=np.hstack((TIME_np,t_ms))

        print('t = %g msec %g    \n'% (t_ms,i))

    #%% Stop when emmitance drop by 10.5 times or after 200 ms
    if  E[i,1]<float(E[0,1])/10.5 or t_ms>=200:
        break;

np.savetxt(outdir+'LEIR_emitt.dat', np.c_[(E)])

#np.savetxt(outdir+'LEIR_single.dat', np.c_[(M2)])
#np.savetxt(outdir+'LEIR_single2.dat', np.c_[(M10)])

#np.savetxt(outdir+'LEIR_beam_out_force.dat', np.c_[(TT0[1:],TT[1:],M1[:-1,1],M1[:-1,5],M1[:-1,6],M1[:-1,7])])
#np.savetxt(outdir+'LEIR_F_Scharge_M0.dat', np.c_[(TT0[:10000],M[:,5])])

M=B0.get_phase_space("%x %xp %y %yp %t %Vz %Vx %Vy %E %P")
#np.savetxt(outdir+'LEIR_beam_out.dat.gz', np.c_[(M)])
np.savetxt(outdir+'LEIR_beam_out.dat', np.c_[(M)])
#Save histograms:
# np.savetxt(outdir+'LEIR_Xhist.dat.gz', np.c_[(histX)])
# np.savetxt(outdir+'LEIR_Yhist.dat.gz', np.c_[(histY)])
# np.savetxt(outdir+'LEIR_Phist.dat.gz', np.c_[(histP)])
# np.savetxt(outdir+'LEIR_time.dat.gz', np.c_[(TIME_np)])
print(datetime.datetime.now() - begin_time)

#To save computation time
# file1 = open("5e4_time.txt","a+")
# file1.write(str(datetime.datetime.now() - begin_time)+"\n")
# file1.close()
