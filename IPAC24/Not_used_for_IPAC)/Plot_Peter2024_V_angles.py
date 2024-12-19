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


############### For interacting with ELENA machine
# start PyJapcScout and so incaify Python instance and get RBAC
myPyJapc = PyJapcScout(incaAcceleratorName='ELENA')
myPyJapc.rbacLogin()

################################################
# this will be the user we will use
selector = 'LNA.USER.PMDEC'
################################################

parameters = [
    'LNR.BINT/Acquisition',
    # 'LNA.BPM/SchottkyAcquisition',
    # 'LNA.BPM/SchottkySettings',
    'LNA.BPM/OrbitAcquisition',
    'LNR4.ECVCATHLM/MEAS.V#VALUE',
    'LNE00.BSGW.0008/Acquisition',
    'LNE50.BSGW.5020/Acquisition',
    'AEX.MW-RF/Acquisition',
    'AX.DK3-EC/Acquisition'
]


data=np.load('cooling/data/test_Peter/V_angles/angles.npz')

n_steps=data['n_steps']
num_samples=data['num_samples']
delay_list=data['delay_list']
repeated_delay=data['repeated_delay']
angles_list=data['angles_list']
#horiztonal plane


plt.figure()

for angle_values in angles_list:

    vertical_list = []
    horizontal_list = []

    vertical_integral_list = []
    horizontal_integral_list = []

    for i in range(len(repeated_delay)):
        print('index', i)
        data_to_look_at = parquet_to_dict(f'cooling/data/test_Peter/V_angles/angles{angle_values}/{i}.parquet')
        # BPM = data_to_look_at['LNE00.BSGW.0008/Acquisition']['value']
        BPM = data_to_look_at['LNE50.BSGW.5020/Acquisition']['value']
        vertical = BPM['sigma'][0]
        horizontal = BPM['sigma'][1]

        vertical_list.append(vertical)
        horizontal_list.append(horizontal)

        vertical_int = BPM['rawIntegrals'][0]
        horizontal_int = BPM['rawIntegrals'][1]

        vertical_integral_list.append(vertical_int)
        horizontal_integral_list.append(horizontal_int)

    filter_threshold = 10
    integral_threshold = 6000

    # filter_horizontal = [sigma_x for sigma_x in horizontal_list if 0 < sigma_x < filter_threshold]
    # filtered_delay_h = [delay_x for idx, delay_x in enumerate(repeated_delay) if
    #                     0 < horizontal_list[idx] < filter_threshold]

    filter_horizontal=[]
    filter_horizontal_intensity=[]
    filtered_delay_h=[]

    for i in range(len(horizontal_list)):
        sigma_x=horizontal_list[i]
        integral_x=horizontal_integral_list[i]
        condition_sigma=sigma_x < filter_threshold
        condition_intensity=integral_x > integral_threshold

        if condition_sigma & condition_intensity:
            filter_horizontal.append(sigma_x)
            filter_horizontal_intensity.append(integral_x)
            filtered_delay_h.append(repeated_delay[i])

    diff_h=np.diff(filtered_delay_h)
    #diff_v=np.diff(filtered_delay_v)

    indices_h = np.where(diff_h != 0)[0]+1
    #indices_v = np.where(diff_v != 0)[0]+1

    sigma_h_groups = np.split(filter_horizontal,indices_h)
    #sigma_v_groups = np.split(filter_vertical,indices_v)

    h_delay_unique=np.unique(filtered_delay_h)
    #v_delay_unique=np.unique(filtered_delay_v)

    means_h=[np.mean(group) for group in  sigma_h_groups]
    #means_v=[np.mean(group) for group in sigma_v_groups]

    stds_h=[np.std(group) for group in sigma_h_groups]
    #stds_v=[np.std(group) for group in sigma_v_groups]


    plt.errorbar(h_delay_unique,means_h,yerr=stds_h,label=f'angle={angle_values}')
    #plt.errorbar(v_delay_unique,means_v,yerr=stds_v,label='vertical')
    plt.xlabel('Delay [ms]')
    plt.ylabel('$\sigma_x$ [mm]')
    plt.legend()
    plt.tight_layout()
plt.show()

###################################################################################################




#vertical plane

plt.figure()

for angle_values in angles:

    vertical_list = []
    horizontal_list = []



    for i in range(len(repeated_delay)):
        print('index', i)
        data_to_look_at = parquet_to_dict(f'cooling/data/test_Peter/angles/angles{angle_values}/{i}.parquet')
        # BPM = data_to_look_at['LNE00.BSGW.0008/Acquisition']['value']
        BPM = data_to_look_at['LNE50.BSGW.5020/Acquisition']['value']
        vertical = BPM['sigma'][0]
        horizontal = BPM['sigma'][1]
        vertical_list.append(vertical)
        horizontal_list.append(horizontal)

    filter_threshold = 10

    filter_horizontal = [sigma_x for sigma_x in horizontal_list if 0 < sigma_x < filter_threshold ]
    filtered_delay_h = [delay_x for idx,delay_x in enumerate(repeated_delay) if 0 < horizontal_list[idx] <filter_threshold]

    filter_vertical = [sigma_y for sigma_y in vertical_list if 0 < sigma_y < filter_threshold ]
    filtered_delay_v = [delay_v for idx,delay_v in enumerate(repeated_delay) if 0 < vertical_list[idx] <filter_threshold]

    diff_h=np.diff(filtered_delay_h)
    diff_v=np.diff(filtered_delay_v)

    indices_h = np.where(diff_h != 0)[0]+1
    indices_v = np.where(diff_v != 0)[0]+1

    sigma_h_groups = np.split(filter_horizontal,indices_h)
    sigma_v_groups = np.split(filter_vertical,indices_v)

    h_delay_unique=np.unique(filtered_delay_h)
    v_delay_unique=np.unique(filtered_delay_v)

    means_h=[np.mean(group) for group in  sigma_h_groups]
    means_v=[np.mean(group) for group in sigma_v_groups]

    stds_h=[np.std(group) for group in sigma_h_groups]
    stds_v=[np.std(group) for group in sigma_v_groups]


    # plt.errorbar(h_delay_unique,means_h,yerr=stds_h,label=f'angle={angle_values}')
    plt.errorbar(v_delay_unique,means_v,yerr=stds_h,label=f'angle={angle_values}')
    plt.xlabel('Delay [ms]')
    plt.ylabel('$\sigma_y$ [mm]')
    plt.legend()
    plt.tight_layout()
plt.show()



# plt.figure()
# plt.scatter(BPM['wirePositions'][1,:],
#          BPM['profiles'][1,:])
# plt.xlabel('[mm]')
# plt.ylabel('[#]')
# plt.tight_layout()
# plt.show()



EC_voltage = data_to_look_at['LNR4.ECVCATHLM/MEAS.V#VALUE']['value']


# last_BPM_data = treatBPMData(myMonitor.lastData['LNA.BPM/OrbitAcquisition']['value'])
#
# plt.plot(last_BPM_data['t_ms'], last_BPM_data['h_mm'][0,:], 'x-')


##### APPENDIX
return
myPyJapc.getSimpleValue('AEX.MW-RF/Delay#delay', selectorOverride=selector)

# myPyJapc.setSimpleValue('AEX.MW-RF/Delay#delay', 2020, selectorOverride=selector)

default_AEX_delay = 2020

