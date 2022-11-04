
RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)

import RF_Track as RFT
from SPS_lattice import SPS_Lattice


L,Twiss=SPS_Lattice()


mass = 1e-6 * 193733676421.31158
p0c = 1e-6  * 18644000000000.0
charge = 79 

n_part=int(1e2)
num_part=1e2

#%%







#%%

#B = RFT.Bunch6d( mass=mass, population=n_part, charge=charge, Pref=p0c, Twiss=Twiss, nParticles=n_part)

B = RFT.Bunch6d(mass, num_part, charge, p0c, Twiss, n_part,0)


#B0 = RFT.Bunch6d(mass, num_part, charge, np.hstack((X_XP_Y_YP, T, P )));