#### Some notes:
'''
Angles and Offsets (of the H- beam) easy/safe to move. 'ELENABEAM/ECOOL-ANGLE-H-mrad'
E_k electrons: at a first order, only need to change  LNR4.ECVCATHLM/REF.DIRECT.V#VALUE
I electrons: need to act, at least, LNR4.ECVGRIDLM/REF.DIRECT.V#VALUE but will probably need to adjust energy as well
-> E_k and I_e are coupled! so you might need to change both at the same time...

The current of the electron, it is in principle measured by LNR4.ECVCOLLM/MEAS.I#VALUE

The current emitted (but not necessarily going into cooling) by the gun, is should be measurable by LNR4.ECVCATHLM/MEAS.I#VALUE

To change the delay of extraction:
AEX.MW-RF/Delay#delay
mind that with 10 ms (10 ticks) delay, you extract approximately at injection

To "not start" the e-cooling process, just set to False
AX.DK3-EC/OutEnable#outEnabled

To delay the start of cooling one can increase the value of
AX.DK3-EC/Delay#delay
Note: the default value=200 1kHz ticks means start at C time 9400, i.e. about 100 ms before our "second injection"


'''



############### Initial imports
from pyjapcscout import PyJapcScout
from datascout import dict_to_parquet
from datascout import parquet_to_dict
from datascout import unixtime_to_string
import numpy as np
import datascout
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter



############### For nice plotting
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 8,
       }
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300


############### Some help functions from previous works
import sys
sys.path.append('/acc/java/data/lna/python/projects/')
from help_functions import treatSchottkyData
from help_functions import imagesc
from help_functions import convert_f_to_dp
from help_functions import ELENA_C_m
from help_functions import analyseProfileEvolution
from help_functions import treatBPMData

data=np.load('cooling/data/test_Peter/optimize_H_angles/angles.npz')

n_steps=data['n_steps']
delay_fixed=data['delay_fixed']
# num_samples=data['num_samples']
# repeated_angles=data['repeated_angles']
angles_list=data['angles_list']
#horiztonal plane

beta_x = 7.6 #m
beta_y = 1.3 #m
D_x = 0.3  #m


horizontal_list = []
vertical_list = []
vertical_integral_list = []
horizontal_integral_list = []


filter_horizontal = []
filter_horizontal_intensity = []

filter_vertical = []
filter_vertical_intensity = []
filter_angles=[]

for i,angle_value in enumerate(angles_list):

        print('angle_value', angle_value)
        data_to_look_at = parquet_to_dict(f'cooling/data/test_Peter/optimize_H_angles/{angle_value}.parquet')
        # BPM = data_to_look_at['LNE00.BSGW.0008/Acquisition']['value']
        BPM = data_to_look_at['LNE50.BSGW.5020/Acquisition']['value']
        vertical = BPM['sigma'][0]
        horizontal = BPM['sigma'][1]
        emittance_x = horizontal**2/beta_x
        emittance_y = vertical ** 2 / beta_y

        vertical_list.append(emittance_y)
        horizontal_list.append(emittance_x)

        integral_y = BPM['rawIntegrals'][0]
        integral_x = BPM['rawIntegrals'][1]
        vertical_integral_list.append(integral_y)
        horizontal_integral_list.append(integral_x)

        filter_threshold_upper = 10
        filter_threshold_lower = 0
        integral_threshold = 5000


        condition_sigma_x = filter_threshold_lower <emittance_x < filter_threshold_upper
        condition_sigma_y = filter_threshold_lower <emittance_y < filter_threshold_upper
        condition_intensity_x = integral_x > integral_threshold
        condition_intensity_y = integral_y > integral_threshold

        if condition_sigma_x & condition_sigma_y & condition_intensity_x & condition_intensity_y:
                filter_horizontal.append(emittance_x)
                filter_vertical.append(emittance_y)
                filter_horizontal_intensity.append(integral_x)
                filter_vertical_intensity.append(integral_y)
                filter_angles.append(angle_value)

plt.figure()
plt.scatter(filter_angles[1:],filter_horizontal[1:])
plt.ylabel('$\epsilon_x$ [mm mrad]')
plt.xlabel('Angle_x ')
# plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(filter_angles[1:],filter_vertical[1:])
plt.ylabel('$\epsilon_x$ [mm mrad]')
plt.xlabel('Angle_x ')
# plt.legend()
plt.tight_layout()
plt.show()



# plt.figure()
# plt.scatter(angles_list[1:],horizontal_list[1:])
# plt.ylabel('$\epsilon_x$ [mm mrad]')
# plt.xlabel('Angle_x ')
# # plt.legend()
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# plt.scatter(angles_list[1:],vertical_list[1:])
# plt.ylabel('$\epsilon_y$ [mm mrad]')
# plt.xlabel('Angle_x ')
# # plt.legend()
# plt.tight_layout()
# plt.show()


#################################################################################################################################

# data=np.load('cooling/data/test_Peter/optimize_H_angles/angles.npz')
#
# n_steps=data['n_steps']
# delay_fixed=data['delay_fixed']
# num_samples=data['num_samples']
# repeated_angles=data['repeated_angles']
# angles_list=data['angles_list']
#horiztonal plane

# mean_emittance_x_list = []
# mean_emittance_y_list = []
# std_emittance_x_list = []
# std_emittance_y_list = []
#
# horizontal_list = []
# vertical_list = []
# vertical_integral_list = []
# horizontal_integral_list = []
#
# for angle_value in angles_list:
#     # horizontal_list = []
#     # vertical_list = []
#     # vertical_integral_list=[]
#     # horizontal_integral_list=[]
#
#
#     for sample_idx in range(num_samples):
#         print('angle_value', angle_value)
#         data_to_look_at = parquet_to_dict(f'cooling/data/test_Peter/optimize_H_angles/angle{angle_value}/{sample_idx}.parquet')
#         # BPM = data_to_look_at['LNE00.BSGW.0008/Acquisition']['value']
#         BPM = data_to_look_at['LNE50.BSGW.5020/Acquisition']['value']
#         vertical = BPM['sigma'][0]
#         horizontal = BPM['sigma'][1]
#         emittance_x = horizontal**2/beta_x
#         emittance_y = vertical ** 2 / beta_y
#
#         vertical_list.append(emittance_y)
#         horizontal_list.append(emittance_x)
#
#         vertical_int = BPM['rawIntegrals'][0]
#         horizontal_int = BPM['rawIntegrals'][1]
#
#         vertical_integral_list.append(vertical_int)
#         horizontal_integral_list.append(horizontal_int)
#
#     filter_threshold_upper = 10
#     filter_threshold_lower = 1.8
#     integral_threshold = 5000
#
#     filter_horizontal = []
#     filter_horizontal_intensity = []
#
#     filter_vertical = []
#     filter_vertical_intensity = []
#
#     for i in range(num_samples):
#         emittance_x = horizontal_list[i]
#         emittance_y = vertical_list[i]
#         integral_x = horizontal_integral_list[i]
#         integral_y = vertical_integral_list[i]
#         condition_sigma_x = filter_threshold_lower <emittance_x < filter_threshold_upper
#         condition_sigma_y = filter_threshold_lower <emittance_y < filter_threshold_upper
#         condition_intensity_x = integral_x > integral_threshold
#         condition_intensity_y = integral_y > integral_threshold
#
#         #if condition_sigma_x & condition_sigma_y & condition_intensity_x & condition_intensity_y:
#         filter_horizontal.append(emittance_x)
#         filter_vertical.append(emittance_y)
#         filter_horizontal_intensity.append(integral_x)
#         filter_vertical_intensity.append(integral_y)
#
#     mean_emittance_x = np.mean(filter_horizontal)
#     mean_emittance_y = np.mean(filter_vertical)
#     std_emittance_x = np.std(filter_horizontal)
#     std_emittance_y = np.std(filter_vertical)
#
#     mean_emittance_x_list.append(mean_emittance_x)
#     mean_emittance_y_list.append(mean_emittance_y)
#     std_emittance_x_list.append(std_emittance_x)
#     std_emittance_y_list.append(std_emittance_y)
#
#
# # plt.figure()
# # plt.errorbar(angles_list, mean_emittance_x_list, yerr=std_emittance_x_list)
# # plt.ylabel('$\epsilon_x$ [mm mrad]')
# # plt.xlabel('Angle_x ')
# # # plt.legend()
# # plt.tight_layout()
# # plt.show()
# #
# # plt.figure()
# # plt.errorbar(angles_list, mean_emittance_y_list, yerr=std_emittance_y_list)
# # plt.ylabel('$\epsilon_y$ [mm mrad]')
# # plt.xlabel('Angle_x ')
# # # plt.legend()
# # plt.tight_layout()
# # plt.show()
