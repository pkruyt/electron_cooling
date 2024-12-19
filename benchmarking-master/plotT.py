import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
import matplotlib.colors as colors

with open('LEIR_Phist.dat', 'r') as myFile:  #LEIR_Xhist.dat for x dist
  lines=myFile.readlines()
  BIN=np.fromstring(lines[0], dtype=float, sep=' ')
  VAL=np.fromstring(lines[1], dtype=float, sep=' ')
  for line in lines[2:]:
    tmp=np.fromstring(line, dtype=float, sep=' ')
    VAL=np.vstack((VAL,tmp))

with open('LEIR_time.dat', 'r') as myFile:
  lines=myFile.readlines()
  T=np.fromstring(lines[0], dtype=float, sep=' ')
  for line in lines[1:]:
      T1=np.fromstring(line, dtype=float, sep=' ')
      T=np.vstack((T,T1))
print(type(VAL),VAL.shape[0])

print(VAL.shape,T.shape, BIN.shape)
fig, ax = plt.subplots()

bar=ax.pcolormesh(BIN, T, VAL, norm=colors.LogNorm(), vmax=10.5e2)
fig.colorbar(bar)
#plt.xlim(18250,18500)
#plt.xlim(18380,18480)
plt.xlim(18380,18500)
#plt.xlim(-30,30) - For LEIR_Xhist.dat
ax.set_xlabel("P [MeV/c]")
#ax.set_xlabel("x [mm]")
ax.set_ylabel("t [ms]")


plt.show()
