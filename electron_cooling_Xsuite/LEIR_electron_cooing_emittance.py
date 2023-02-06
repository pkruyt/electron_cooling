# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk




# for ctx in xo.context.get_test_contexts():
#     print(f"Test {ctx.__class__}")

I=0.6
L = 2.5 # m cooler length
r_beam=25*1e-3


#ne = 8.27e13 #nocolo's value
#ne=83435412163766.98
T_perp = 0.01 # <E> [eV] = kb*T
T_l =  0.001 # <E> [eV]
B = 0.075 # T for LEIR
Z=54
beta=0.09405801936973575
gamma = 1.004453024014182
mass0=193729.0248722061*1e6 #eV
c=299792458.0
p0c = mass0*beta*gamma #eV/c


arc=dtk.LinearTransferMatrix(Q_x=1.8201, Q_y=2.72,
                            beta_x_0=5, beta_x_1=5,
                            beta_y_0=5,  beta_y_1=5,
                            alpha_x_0=0,   alpha_x_1=0,
                            alpha_y_0=0,   alpha_y_1=0,
                            disp_x_0=0,disp_x_1=0,
                            disp_y_0=0,    disp_y_1=0,
                            beta_s=1.11111111111*1e40,
                            Q_s=0,
                            chroma_x=0, chroma_y=0
                            )

emittance=14*1e-6
#x=np.random.normal(0,np.sqrt(5*emittance) ,1),
#px=np.random.normal(0, np.sqrt(emittance/5), 1),
dtk_particle = dtk.TestParticles(
        
        mass0=mass0,
        p0c=p0c,
        x=np.random.normal(0,np.sqrt(5*emittance) ,1000),
        px=np.random.normal(0, np.sqrt(emittance/5), 1000),
        y=0,
        py=0,
        delta=0,
        zeta=0)



dtk_particle_copy_old=dtk_particle.copy()







dtk_cooler = dtk.elements.ElectronCooler(I=I,L=L,r_beam=r_beam,#ne=ne,
                                         T_perp=T_perp,T_l=T_l,
                                         B=B,Z=Z,B_ratio=1e-10,Neutralisation=1)

       
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
num_turns=int(25000)
# num_turns=int(100000)

x=[]
px=[]

for i in tqdm(range(num_turns)):
    x.append(dtk_particle.x)
    px.append(dtk_particle.px)
    
    arc.track(dtk_particle)
    dtk_cooler.track(dtk_particle)




# dtk_cooler.track(dtk_particle)

# Fx = dtk_cooler.force(dtk_particle)

# import matplotlib.pyplot as plt

# plt.scatter(dtk_particle.px,(Fx),label='$B_⟂$/$B_∥$')
# plt.xlim([-0.002,0.002])
# plt.title('cooling force vs px')
# plt.ylabel('cooling force [eV/m]')
# plt.xlabel('transverse momentum px [rad]')
# plt.legend()

# print(dtk_particle.x)



# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line = ax.scatter(x[0], px[0],color='orange',label='initial')

fig.suptitle('Energy reduction with dispersion')
ax.set_xlabel('x [m]')
ax.set_ylabel('px [rad]')

ax.legend()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


    
fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Turns',
    valmin=0,
    valmax=num_turns-1,
    valinit=1,
)


def update(val):
    value=freq_slider.val
    ax.clear()
    
    #a1=lower_bound_list[int(value)]
    #a2=max_x_list[int(value)]
    
    #ax.axvline(x=a1,color='red',label='cooling zone')
    #ax.axvline(x=a2,color='red')
    ax.scatter(x[0], px[0],color='orange',label='initial')
    ax.scatter(x[int(value)],px[int(value)]
                ,label='turn evolution')
    ax.legend()
    # Set the x and y limits to the initial values
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #plt.draw()

    
freq_slider.on_changed(update)

#%%


# turns=list(range(num_turns))
# result = [number/1000 for number in turns]

def emittance_2d(x,px):
    
    
  
        
    cov_list=[]
    for i in range(num_turns):
        x0=x[i]
        px0=px[i]
    
    
        cov00=np.cov(x0,px0)
    
        #det00 = (np.sqrt((np.linalg.det(cov00)))*beta*gamma)
        det00 = (np.sqrt((np.linalg.det(cov00))))
        cov_list.append(det00)
        
   
    #print(cov_list)
    
    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    plt.plot(cov_list)
    
    plt.title('Horizontal emmitance vs turns (2d)')
    plt.ylabel('emittance (m)')
    plt.xlabel('number of turns')    
    return 

emittance_2d(x,px)

