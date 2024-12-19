# importing some generic packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import math
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import scipy.optimize

def monoExp(x, t):
    return 14.07920631*np.exp(-1* x/t)




R0 = []
R1 = []

delimiter=" "
#with open('/home/agson/cernbox2/Agnieszka/new_rf_track/inter_tracking/results/Scharge/shift3YD0/LEIR_emitt.dat') as f:
#with open('/home/pkruyt/cernbox/RF_Code/benchmarking-master-SPS/tracking/results/LEIR_emitt.dat') as f:
with open('/home/pkruyt/cernbox/electron_cooling/benchmarking-master-SPS/results/LEIR_emitt.dat') as f:
    data = f.readlines()
    for line in data:
       # print(line)
        R0.append(float(line.strip().split(delimiter)[0]))
        R1.append(float(line.strip().split(delimiter)[1]))
#Some fitting if you need
# p0 = (0.12) # start with values near those we expect
# params, cv = scipy.optimize.curve_fit(monoExp, R0, R1,p0)
# t = params
# print(params)

R0_old = []
R1_old = []
delimiter=" "
#with open('/home/agson/cernbox2/Agnieszka/new_rf_track/inter_tracking/results/Scharge/shift3YD2/LEIR_emitt.dat') as f:
with open('/home/pkruyt/cernbox/electron_cooling/benchmarking-master-SPS/results/LEIR_emitt.dat') as f:
    data = f.readlines()
    for line in data:
       # print(line)
        R0_old.append(float(line.strip().split(delimiter)[0]))
        R1_old.append(float(line.strip().split(delimiter)[1]))


R0_int = []
R1_int = []
delimiter=" "
#with open('/home/agson/cernbox2/Agnieszka/new_rf_track/inter_tracking/results/Scharge/shift3YD5/LEIR_emitt.dat') as f:
with open('/home/pkruyt/cernbox/electron_cooling/benchmarking-master-SPS/results/LEIR_emitt.dat') as f:
    data = f.readlines()
    for line in data:
       # print(line)
        R0_int.append(float(line.strip().split(delimiter)[0]))
        R1_int.append(float(line.strip().split(delimiter)[1]))


#R0PY = []
#R1PY = []
#delimiter="\t"
##with open('/home/agson/cernbox2/Lxplus-betacool/Par/emittance.cur') as f:
#with open('/home/agson/cernbox2/Lxplus-betacool/Par/emittance.cur') as f:
#    data = f.readlines()
#    for line in data:
#       # print(line)
#        R0PY.append(float(line.strip().split(delimiter)[0]))
#        R1PY.append(float(line.strip().split(delimiter)[1]))


fig, ax = plt.subplots()

#Betacool
# R0_B_ms = [element *1e3 for element in R0PY]  #
# ax.plot(R0_B_ms, R1PY,'-',label='%s' % ('Betacool Park'))

ax.plot(R0_old, R1_old,'-',label='%s' % ('RF-Track A'))
#ax.plot(R0, R1,'-',label='%s' % ('RF-Track B'))
#ax.plot(R0_int, R1_int,'-',label='%s' % ('RF-Track C'))

plt.axhline(y=2, color='black', linestyle='-')

# p1 = (0.5) # start with values near those we expect
# params1, cv1 = scipy.optimize.curve_fit(monoExp, R0PY, R1PY,p1)
# t1 = params1
# print(1/t1,1/R0PY[(np.abs(np.array(R1PY)-np.array(R1PY)[0]/math.e)).argmin()],1/R0PY[(np.abs(np.array(R1PY)-np.array(R1PY)[0]/2)).argmin()])

#ax.plot(R0_B_ms, monoExp(np.array(R0_B_ms), t1),'-',label='%s' % ('Betacool'))
ax.set_xlabel('time [ms]')
ax.set_ylabel('Emittance')
plt.legend()
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.xlim(0, 20);
plt.xlim(0, 200);
plt.show()
