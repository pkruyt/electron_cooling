# a few functions useful to treat data
import numpy as np
import re
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.optimize as op
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET

# to be able to un-roll ad cycle functions
# from ad_cycle import structure as ad_structure


import sys
sys.path.append('/acc/java/data/lna/python/externalProjects/scraperanalysis')
sys.path.append('/acc/java/data/lna/python/externalProjects/LEIR-OP-scripts/MD')

# Physical constants ########
# Speed of light
clight_m_s = 299792458
# Proton mass 
mp_MeV_c2 = 938.27208816
# Electron mass
me_MeV_c2 = 0.510998950
# H- mass 
mH_MeV_c2 = mp_MeV_c2+2*me_MeV_c2
# ELENA/AD Ring Circumference
ELENA_C_m =  30.4054
AD_C_m    = 182.4328
# elementary charge, in Coulomb
e_C = 1.602176634e-19


def treatBPMData(BPM):
    '''
    BPM is expected to be the simple acquisition of  'LNA.BPM/OrbitAcquisition', i.e. a dictionary with all fields of LNA.BPM class
    returns a new dict with all interesting data
    '''

    output = dict()

    # extract all BPM data in a more interesting way...
    BPM_t_ms = BPM['startTimePerWindow_ms'][0] + (1000 / (BPM['samplingFrequencyPerWindow_hz'][0])) * np.arange(np.shape(BPM['positionWindow1'])[1])
    # BPM_t_ms = BPM['hardwareTimestampsWindow1_10us_ticks'][0,:]/100
    BPM_o_mm = BPM['positionWindow1']

    for i in (np.arange(3)):
        aux_t = BPM['startTimePerWindow_ms'][1 + i] + (
                    1000 / (BPM['samplingFrequencyPerWindow_hz'][i + 1])) * np.arange(
            np.shape(BPM['positionWindow' + str(i + 2)])[1])
        #aux_t = BPM[f'hardwareTimestampsWindow{str(i + 2)}_10us_ticks'][0,:]/100
        BPM_t_ms = np.concatenate([BPM_t_ms, aux_t])
        BPM_o_mm = np.concatenate([BPM_o_mm, BPM['positionWindow' + str(i + 2)]], axis=1)
    # sort it by time
    auxIdx = np.argsort(BPM_t_ms)
    BPM_t_ms = BPM_t_ms[auxIdx]
    BPM_o_mm = BPM_o_mm[:, auxIdx]
    # remove duplicate
    BPM_t_ms, auxIdx = np.unique(BPM_t_ms, return_index=True)
    BPM_o_mm = BPM_o_mm[:, auxIdx]

    # sort data by BPM name
    auxNamesIdx = [int((re.findall('\d+', name) or [0])[0]) for name in BPM['bpmNames']]
    auxIdx   = np.argsort(auxNamesIdx)
    sortedNames = [BPM['bpmNames'][i] for i in auxIdx]
    BPM_o_mm = BPM_o_mm[auxIdx, :]
    BPM_H = [bool(re.search('(H$|^DR.UH|^UEH)', s)) for s in sortedNames]
    BPM_V = [bool(re.search('(V$|^DR.UV|^UEV)', s)) for s in sortedNames]

    output['t_ms'] = BPM_t_ms
    output['h_mm'] = BPM_o_mm[BPM_H, :]
    output['v_mm'] = BPM_o_mm[BPM_V, :]
    output['h_names'] = [sortedNames[i] for i in np.arange(len(BPM_H))[BPM_H]]
    output['v_names'] = [sortedNames[i] for i in np.arange(len(BPM_V))[BPM_V]]

    return output

def TBTtoObitData(BPM_tbt, k_orbit = 70/4, remove_mean=False, mean_portion=1, phase_offset_samples=0, 
                     data_n_splits = 1, harmonic = 1):
    '''
    Computes closed orbit data from BPM trajectory data using frequency analysis similar to what done in BPM DSP code.
    This is useful, for example, for the e-cooling e- orbit data analysis.
    '''
    
    ############ start extracting useful data ##########################
    tracesDelta = np.array(BPM_tbt['tracesDelta'], dtype=float)
    tracesSigma = np.array(BPM_tbt['tracesSigma'], dtype=float)
    n_samples_turn = BPM_tbt['bucketWidth_samples']
    tracesNames = BPM_tbt['bpmNames']
    start_time_ms =  BPM_tbt['startTime_us']/1000
    data_BPM_name = [name for name in tracesNames if name != '']
    N_BPMS = len(data_BPM_name)  # could be read from hardware

    # get number of samples
    n_samples = np.shape(tracesSigma)[1]
    
    # apply some phase offset between sum and delta signals
    if phase_offset_samples >= 0:
        data_delta = tracesDelta[:,phase_offset_samples:]
        data_sigma = tracesSigma[:,0:n_samples - phase_offset_samples]
    elif phase_offset_samples < 0:
        phase_offset_samples = -phase_offset_samples
        data_delta = tracesDelta[:,0:n_samples - phase_offset_samples]
        data_sigma = tracesSigma[:,phase_offset_samples:]
        
    # for e-cooler orbit measurement, expected that signal is symmetric. just remove the mean
    # for other signal types (e.g. injection), remove some other baseline...
    if remove_mean:
        data_sigma = data_sigma - np.nanmean(data_sigma[:,0:int(mean_portion*n_samples)])
        data_delta = data_delta - np.nanmean(data_delta[:,0:int(mean_portion*n_samples)])

    
    # make simpler computation in I/Q-like mode
    LO_frequency = harmonic/n_samples_turn
    LO_zero = np.sin(2*np.pi*LO_frequency * np.arange(n_samples))
    LO_perp = np.sin(2*np.pi*LO_frequency * np.arange(n_samples) + np.pi/2)
    
    # project sigma and delta on zero and perp, then split data in n backets and keep all but last (which might be incomplete)
    data_sigma_zero = np.array_split(data_sigma*LO_zero, data_n_splits)[:-1]
    data_sigma_perp = np.array_split(data_sigma*LO_perp, data_n_splits)[:-1]
    #
    data_delta_zero = np.array_split(data_delta*LO_zero, data_n_splits)[:-1]
    data_delta_perp = np.array_split(data_delta*LO_perp, data_n_splits)[:-1]
    
    # prepare array to host data
    orbit_product = np.nan*np.zeros((N_BPMS, data_n_splits))

    for iBPM in np.arange(N_BPMS):
        # project sigma and delta on zero and perp, then split data in n backets and keep all but last (which might be incomplete)
        data_sigma_zero = np.array_split(data_sigma[iBPM]*LO_zero, data_n_splits)
        data_sigma_perp = np.array_split(data_sigma[iBPM]*LO_perp, data_n_splits)
        #
        data_delta_zero = np.array_split(data_delta[iBPM]*LO_zero, data_n_splits)
        data_delta_perp = np.array_split(data_delta[iBPM]*LO_perp, data_n_splits)

        for (j, _sigma_zero, _sigma_perp, _delta_zero, _delta_perp) in zip(np.arange(data_n_splits), data_sigma_zero, data_sigma_perp, data_delta_zero, data_delta_perp):
            _sigma_zero = np.mean(_sigma_zero)
            _sigma_perp = np.mean(_sigma_perp)
            _delta_zero = np.mean(_delta_zero)
            _delta_perp = np.mean(_delta_perp)

            # compute amplitude, or norm of the vector
            _sigma_norm = np.sqrt(_sigma_zero**2 + _sigma_perp**2)

            # compute amplitude, or norm of the vector
            _delta_norm = np.sqrt(_delta_zero**2 + _delta_perp**2)

            # assume delta is in phase with sigma, so compute projection of delta
            #  signal on the "sum" direction, such to keep the sign as well
            orbit_product[iBPM, j] = k_orbit*(_delta_zero*_sigma_zero + _delta_perp*_sigma_perp)/_sigma_norm**2
    
    # compute mean over splits
    all_orbits = np.mean(orbit_product, 1)
    all_orbits_error = np.std(orbit_product, 1)/np.sqrt(data_n_splits)
    
    # sorting by position
    auxNamesIdx = [int((re.findall('\d+', name) or [0])[0]) for name in data_BPM_name]
    auxIdx   = np.argsort(auxNamesIdx)
    sortedNames = [data_BPM_name[i] for i in auxIdx]
    # re-sort orbit data and error
    BPM_o_mm = all_orbits[auxIdx]
    BPM_err_mm = all_orbits_error[auxIdx]
    # also sort data_sigma and data_delta
    data_sigma = data_sigma[auxIdx]
    data_delta = data_delta[auxIdx]
    # try to identify H/V
    BPM_H = [bool(re.search('(H$|^DR.UH|^UEH)', s)) for s in sortedNames]
    BPM_V = [bool(re.search('(V$|^DR.UV|^UEV)', s)) for s in sortedNames]
    
    # prepare output and return it
    output = dict()
    output['t_ms'] = start_time_ms
    output['mm'] = BPM_o_mm
    output['err_mm'] = BPM_err_mm
    output['names'] = sortedNames
    output['h_mm'] = BPM_o_mm[BPM_H]
    output['v_mm'] = BPM_o_mm[BPM_V]
    output['h_err_mm'] = BPM_err_mm[BPM_H]
    output['v_err_mm'] = BPM_err_mm[BPM_V]
    output['h_data_sigma'] = data_sigma[BPM_H]
    output['v_data_sigma'] = data_sigma[BPM_V]
    output['h_data_delta'] = data_delta[BPM_H]
    output['v_data_delta'] = data_delta[BPM_V]
    output['h_names'] = [sortedNames[i] for i in np.arange(len(BPM_H))[BPM_H]]
    output['v_names'] = [sortedNames[i] for i in np.arange(len(BPM_V))[BPM_V]]
    return output

def extractTBTdata(BPM_tbt, k_orbit = 70/4, positive_sum_signal = True, BASELINE_THRESHOLD = 0.02, ORBIT_THRESHOLD = 0.70):
    '''
    Extracts turn-by-turn orbit data, subtracting turn-by-turn baseline...
    
    NOTE: Consider using Bertrand's code instead!!!! Or simply use data from FESA class itself!
    '''
    
    ############ start extracting useful data ##########################
    data_delta = np.array(BPM_tbt['tracesDelta'], dtype=float)
    data_sigma = np.array(BPM_tbt['tracesSigma'], dtype=float)
    tracesNames = BPM_tbt['bpmNames']
    n_samples_turn = BPM_tbt['bucketWidth_samples']
    data_BPM_name = [name for name in tracesNames if name != '']
    N_BPMS = len(data_BPM_name)  # could be read from hardware
        
    ####################### look for turn by turn data ###############################
    data_sigma_turns = np.copy(np.reshape(data_sigma, (N_BPMS, -1, n_samples_turn)))
    data_delta_turns = np.copy(np.reshape(data_delta, (N_BPMS, -1, n_samples_turn)))
    
    # just a holder... it needs to be better treted
    data_orbit_turns = k_orbit*data_delta_turns/data_sigma_turns
    
    # do actual baseline subtraction turn by turn...
    all_sums = np.sum(data_sigma_turns,1)
    all_min = np.min(all_sums, 1)
    all_max = np.max(all_sums, 1)
    #
    for i_bpm in np.arange(N_BPMS):
        baseline_idx = all_sums[i_bpm, :]  < all_min[i_bpm] + BASELINE_THRESHOLD*(all_max[i_bpm]-all_min[i_bpm])
        not_orbit_idx = all_sums[i_bpm, :] < all_min[i_bpm] +    ORBIT_THRESHOLD*(all_max[i_bpm]-all_min[i_bpm])
        #
        for i_turn in np.arange(np.shape(data_sigma_turns)[1]):
            data_sigma_turns[i_bpm, i_turn, :] -= np.mean(data_sigma_turns[i_bpm, i_turn, baseline_idx])
            data_delta_turns[i_bpm, i_turn, :] -= np.mean(data_delta_turns[i_bpm, i_turn, baseline_idx])
            data_orbit_turns[i_bpm, i_turn, :] = k_orbit*data_delta_turns[i_bpm, i_turn, :]/data_sigma_turns[i_bpm, i_turn, :]
            data_orbit_turns[i_bpm, i_turn, not_orbit_idx] = np.nan
    
    return (data_sigma_turns, data_delta_turns, data_orbit_turns, data_BPM_name)

def compute_energy_error(treatedBPMData, DX_m):
    '''
    Given BPM data treated and disperion (as pandas) in m, it computes the expected momentum error

    :param treatedBPMData:
    :param DX_m:
    :return: dp/p
    '''
    # factor 1000 to get oribt in m
    allOrbitHclean = np.copy(np.transpose(treatedBPMData['h_mm'])/1000)
    auxDX = DX_m.loc[treatedBPMData['h_names']].to_numpy()
    DXnorm = np.linalg.norm(auxDX)
    DXnormlised = auxDX/DXnorm
    dp_p = np.dot(allOrbitHclean, DXnormlised)/DXnorm
    return dp_p


def compute_orbit_correction(treatedBPMData, momentum, initValueH, initValueV, RM_H, RM_V, DX,
                             startTime = 0, endTime = 1000000, disabledBPM_patterns = None, frozenBPM_patterns = None,
                             start_end_smooth = True, orbit_filter_average=101):
    ''' compute orbit correction using response matrixes and removing dispersion contribution

    :param treatedBPMData:
    :param momentum:
    :param initValueH:
    :param initValueV:
    :param RM_H:
    :param RM_V:
    :param DX:
    :param startTime:
    :param endTime:
    :param disabledBPM_patterns:
    :param frozenBPM_patterns:
    :param start_end_smooth:  If to correct from start to end in a smooth way, oterwise, just correct to zero.
    :param orbit_filter_average: Some value used to filter orbit value over nearby points...
    :return:
    '''

    # disabled some BPMs readings
    if disabledBPM_patterns is not None:
        enabledBPM_H = [ not any([re.search(pattern, name) for pattern in disabledBPM_patterns]) for name in treatedBPMData['h_names']]
        enabledBPM_V = [ not any([re.search(pattern, name) for pattern in disabledBPM_patterns]) for name in treatedBPMData['v_names']]
    else:
        enabledBPM_H = [True for element in treatedBPMData['h_names']]
        enabledBPM_V = [True for element in treatedBPMData['v_names']]

    ## now let's use the machine data....
    # cleanup from energy error, assuming dispersion is correct
    allOrbitH      = np.copy(np.transpose(treatedBPMData['h_mm']))
    allOrbitV      = np.copy(np.transpose(treatedBPMData['v_mm']))
    allOrbitHclean = np.copy(np.transpose(treatedBPMData['h_mm']))
    auxDX = DX.loc[treatedBPMData['h_names']].to_numpy()
    DXnorm = auxDX/np.linalg.norm(auxDX)
    for i in np.arange(np.shape(allOrbitHclean)[0]):
        allOrbitHclean[i] = allOrbitHclean[i] - np.dot(allOrbitHclean[i], DXnorm)*DXnorm

    # filter in time
    orbit_time_filter = (treatedBPMData['t_ms'] >= startTime) & (treatedBPMData['t_ms'] <= endTime)
    # extract the interesting portion of data after some filtering
    portionOrbitH = np.transpose(savgol_filter(np.transpose(allOrbitHclean), orbit_filter_average, 3))[orbit_time_filter]
    portionOrbitV = np.transpose(savgol_filter(np.transpose(allOrbitV), orbit_filter_average, 3))[orbit_time_filter]
    portionOrbitTime = treatedBPMData['t_ms'][orbit_time_filter]

    # See if you want to correct from init to end in a smooth way or if to just go to zero everywhere
    if start_end_smooth:
        allOrbitHi    = np.arange(np.shape(portionOrbitH)[1])
        f=interp2d(portionOrbitTime[[0, -1]], allOrbitHi, np.transpose(portionOrbitH[[0, -1]]), kind='linear')
        zeroOrbitH = np.transpose(f(portionOrbitTime, allOrbitHi))

        allOrbitVi    = np.arange(np.shape(portionOrbitV)[1])
        f=interp2d(portionOrbitTime[[0, -1]], allOrbitVi, np.transpose(portionOrbitV[[0,-1]]), kind='linear')
        zeroOrbitV = np.transpose(f(portionOrbitTime, allOrbitVi))

        # This is how much I would like to move the orbit
        wantedCorrectionH = zeroOrbitH - portionOrbitH
        wantedCorrectionV = zeroOrbitV - portionOrbitV
    else:
        # just go to zero
        wantedCorrectionH = -portionOrbitH
        wantedCorrectionV = -portionOrbitV

    # filter out some bpms where you don't want to move the beam
    if frozenBPM_patterns is not None:
        wantedCorrectionH[:, [any([re.search(pattern, name) for pattern in frozenBPM_patterns]) for name in treatedBPMData['h_names']]] = 0
        wantedCorrectionV[:, [any([re.search(pattern, name) for pattern in frozenBPM_patterns]) for name in treatedBPMData['v_names']]] = 0

    # compute inverted matrices
    RM_H_inv = pd.DataFrame(np.linalg.pinv(RM_H.values[enabledBPM_H, :]), RM_H.columns, RM_H.index[enabledBPM_H])
    RM_V_inv = pd.DataFrame(np.linalg.pinv(RM_V.values[enabledBPM_V, :]), RM_V.columns, RM_V.index[enabledBPM_V])
    # check inversion
    RM_H_inv.dot(RM_H.iloc[enabledBPM_H])
    RM_V_inv.dot(RM_V.iloc[enabledBPM_V])

    # and easy to calculate strength now
    # (needed to invert the sign. probably different conventions: 23:15)
    neededHcor = -np.matmul(RM_H_inv.values, np.transpose(wantedCorrectionH[:, enabledBPM_H]/1000))
    neededVcor = -np.matmul(RM_V_inv.values, np.transpose(wantedCorrectionV[:, enabledBPM_V]/1000))

    # and what to expect ################
    expectedHorbitDelta = np.transpose(np.matmul(RM_H.values, neededHcor))*1000
    expectedVorbitDelta = np.transpose(np.matmul(RM_V.values, neededVcor))*1000
    expectedHorbit = allOrbitH
    # minus added to recover same conventions...
    expectedHorbit[orbit_time_filter, :] = expectedHorbit[orbit_time_filter, :] - expectedHorbitDelta
    expectedVorbit = allOrbitV
    expectedVorbit[orbit_time_filter, :] = expectedVorbit[orbit_time_filter, :] - expectedVorbitDelta
    #####################################


    # now we need to add them to the correctors...
    # a) use momentum time to add new points
    momentum_T = momentum['JAPC_FUNCTION']['X']
    momentum_T = momentum_T[(momentum_T >= startTime) & (momentum_T <= endTime)]
    newValueH = initValueH.copy()
    for i, corrector in enumerate(RM_H.columns):
        auxX = newValueH[corrector]['value']['JAPC_FUNCTION']['X']
        auxY = newValueH[corrector]['value']['JAPC_FUNCTION']['Y']
        fold = interp1d(auxX, auxY, fill_value=(0, 0), bounds_error=False)
        auxXnew = np.concatenate((auxX[auxX < startTime], momentum_T, auxX[auxX > endTime]), axis=None)
        fnew = interp1d(portionOrbitTime, neededHcor[i], fill_value=(0, 0), bounds_error=False)
        newAuxY = fold(auxXnew) + fnew(auxXnew)
        newValueH[corrector]['value']['JAPC_FUNCTION']['X'] = auxXnew
        newValueH[corrector]['value']['JAPC_FUNCTION']['Y'] = newAuxY

    newValueV = initValueV.copy()
    for i, corrector in enumerate(RM_V.columns):
        auxX = newValueV[corrector]['value']['JAPC_FUNCTION']['X']
        auxY = newValueV[corrector]['value']['JAPC_FUNCTION']['Y']
        fold = interp1d(auxX, auxY, fill_value=(0, 0), bounds_error=False)
        auxXnew = np.concatenate((auxX[auxX < startTime], momentum_T, auxX[auxX > endTime]), axis=None)
        fnew = interp1d(portionOrbitTime, neededVcor[i], fill_value=(0, 0), bounds_error=False)
        newAuxY = fold(auxXnew) + fnew(auxXnew)
        newValueV[corrector]['value']['JAPC_FUNCTION']['X'] = auxXnew
        newValueV[corrector]['value']['JAPC_FUNCTION']['Y'] = newAuxY

    return (newValueH, newValueV, expectedHorbit, expectedVorbit)



def treatLNRBINTData(data, start_time_ms = None, expected_tau_s=np.inf, moving_avg_samples = 1):
    '''
    treatLNRBINTData(data, start_time_ms = None, expected_tau_s=np.inf, moving_avg_samples = 1)
    
    treats Intensity data coming from LNR.BINT device, eventually compensating for expected lifetime of H-
    (you need to iteratively play with parameter for the time being...)
    
    It also allows to make some moving average sampling, so to smooth signal.
    
    It returns (aux_time, aux_value, aux_der) where aux_der is the pure gradient (no time or amplitude normalisation)
    
    '''
    
    aux_time = data['firstSampleTimeStampMs'] + 1000*data['dt']*np.arange(len(data['calibratedIntensity']))
    aux_value = data['calibratedIntensity']
    
    # find t0, start of signal
    if start_time_ms is None:
        t_0 = aux_time[np.where(aux_value > 0.1*np.max(aux_value))[0][0]]
    else:
        t_0 = start_time_ms
    
    # perform lifetime compensation 
    # Idea is that original signal is I_0 * exp(-t/tau). Hence, simply divide by exp(-t/tau) to get
    if expected_tau_s is not None:
        lifetime_compensation = 1/np.exp(-(aux_time-t_0)/(expected_tau_s*1000))
        aux_value = aux_value*lifetime_compensation
    
    # compute moving average
    aux_value = moving_average(aux_value, moving_avg_samples)
    
    # compute also derivative of signal
    aux_der = np.gradient(aux_value, 2)

    return (aux_time, aux_value, aux_der)

def treatIPMData(data, planes=['H', 'V'], smooth=5, machine='LEI',
                 scout=False, custom_channel_mapping=None, fast=False,
                 baseline = [], baseline_interval_ms=None,
                 moving_average_samples = None):
    ''' treats IPM data and extract profiles etc...

    :param data: data acquired from a "complete acquistion" from pyjapcscout, for example, 'ADE.BGI.H/Acquisition' AND 'ADE.BGI.H/Setting',
    :param planes: which planes to look for, default=['H', 'V']:
    :param smooth: default=5
    :param machine: if 'LEI' or 'ADE'
    :param scout: use data from pyjapcscout (i.e. looking for a "value" filed ...
    :param custom_channel_mapping: if you want to rempa some channels on some other... default=None
    :param fast: don't do all fancy computation, and just return the "raw" data. default=False
    :param baseline: default=[], i.e. don't do any baseline subtraction
    :param baseline_interval_ms: default=None, i.e. don't do any baseline subtraction.
        If provided, override (if provided) the baseline
    :param moving_average_samples: default=None, number of time samples to average
    :return:
    '''

    # Define a couple of functions used to treat the data:
    #  Standard Gaussian with baseline offset:
    _fitfunc = lambda p, x: p[3] + p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))
    #  Simple formula to compute difference of data...
    _errfunc = lambda p, x, y: _fitfunc(p, x) - y

    if scout:
        BGIdata    = lambda p: data[machine + '.BGI.' + p + '/Acquisition']['value']['integrationData']
        BGIsetting = lambda p: data[machine + '.BGI.' + p + '/Setting']['value']
        StartBGI   = lambda p: data[machine + '.BGI.' + p + '/Acquisition']['value']['armCycleTime']
    else:
        BGIdata    = lambda p: data[machine + '.BGI.' + p + '/Acquisition']['integrationData']
        BGIsetting = lambda p: data[machine + '.BGI.' + p + '/Setting']
        StartBGI   = lambda p: data[machine + '.BGI.' + p + '/Acquisition']['armCycleTime']

    if machine == 'LEI':
        n_channels = 50
        channels = {p: np.arange(n_channels) for p in planes}
        # channels['H'][20] = 42
        # channels['H'][32] = 20
        # channels['H'][34] = 40
        # channels['H'][36] = 38
        # channels['H'][38] = 36
        # channels['H'][40] = 34
        # channels['H'][42] = 32
        # channels['H'].reverse()
    elif machine == 'ADE':
        n_channels = 50
        channels = {p: np.arange(n_channels) for p in planes}
        # channels['H'][3] = 2 # not behaving channel
    else:
        raise ValueError('Unknown machine '+machine)

    if custom_channel_mapping is not None:
        print('Overwriting default channel mapping...')
        channels = custom_channel_mapping       

    # prepare output
    output = dict()
    # Extract basic data
    dt         = {p: BGIsetting(p)['integrationPeriod'] / 2 for p in planes}
    #time       = {p: np.arange(StartBGI(p), BGIsetting(p)['measTimeWindow'], dt[p]) for p in planes}
    time       = {p: np.arange(StartBGI(p), StartBGI(p)+BGIsetting(p)['measTimeWindow'], dt[p]) for p in planes}
    pos        = {p: np.arange(n_channels) - n_channels / 2. for p in planes}
    BGI        = {p: - (np.float_(BGIdata(p)).T)[channels[p], :len(time[p])] for p in planes}

    #  - make baseline subtraction
    baseline = dict()
    if isinstance(baseline, np.ndarray) and len(baseline) == n_channels:
        pass
    elif isinstance(baseline_interval_ms, np.ndarray) and len(baseline_interval_ms) == 2:
        for p in planes:
            _idx = (time[p] >= baseline_interval_ms[0]) & (time[p] <= baseline_interval_ms[1])
            baseline[p] = np.mean(BGI[p][:, _idx], axis=1)
    else:
        baseline = {p: np.zeros(n_channels) for p in planes}
    for p in planes:
        BGI[p] = (BGI[p].transpose() - baseline[p]).transpose()

    ### integrate several profiles, i.e. run a moving average...
    if moving_average_samples is not None:
        def moving_average(x, w):
            _new_x = np.convolve(x, np.ones(w), 'valid') / w
            _new_x = np.concatenate([np.zeros(w - 1), _new_x])
            return _new_x
        for p in planes:
            for i_channel in np.arange(n_channels):
                BGI[p][i_channel,:] = moving_average(BGI[p][i_channel,:], moving_average_samples)

    output['dt']   = dt
    output['time'] = time
    output['pos']  = pos
    output['BGI']  = BGI
    output['baseline']             = baseline
    output['baseline_interval_ms'] = baseline_interval_ms
    if fast:
        return output

    # Do more computations:
    #  - treat as simple FWHM
    FWHM_all = {p: np.array([compute_single_fwhmfit(pos[p], BGI[p][:,i]) for i in range(BGI[p].shape[1])]).transpose() for p in planes}
    FWHM     = {p: FWHM_all[p][2,:] for p in planes}
    #  - do Gaussian fits
    Gauss_all   = {p: np.array([compute_single_gaussianfit(pos[p], BGI[p][:,i]) for i in range(BGI[p].shape[1])]).transpose() for p in planes}
    Gauss_sigma = {p: Gauss_all[p][2,:] for p in planes}
    Gauss_mu    = {p: Gauss_all[p][1,:] for p in planes}
    #  - do Moment fits
    moment_all   = {p: np.array([compute_single_momentfit(pos[p], BGI[p][:,i]) for i in range(BGI[p].shape[1])]).transpose() for p in planes}
    moment_sigma = {p: moment_all[p][1,:] for p in planes}
    moment_mu    = {p: moment_all[p][0,:] for p in planes}
    
    #  - repeat, for smoothed data
    BGI_smooth = {p: ndimage.uniform_filter(BGI[p], size=(0, smooth)) for p in planes}
    FWHM_smooth_all = {p: np.array([compute_single_fwhmfit(pos[p], BGI_smooth[p][:,i]) for i in range(BGI_smooth[p].shape[1])]).transpose() for p in planes}
    FWHM_smooth     = {p: FWHM_smooth_all[p][2,:] for p in planes}
    #  - do Gaussian fits
    Gauss_smooth_all   = {p: np.array([compute_single_gaussianfit(pos[p], BGI_smooth[p][:,i]) for i in range(BGI_smooth[p].shape[1])]).transpose() for p in planes}
    Gauss_smooth_sigma = {p: Gauss_smooth_all[p][2,:] for p in planes}
    Gauss_smooth_mu    = {p: Gauss_smooth_all[p][1,:] for p in planes}
    #  - do Moment fits
    moment_smooth_all   = {p: np.array([compute_single_momentfit(pos[p], BGI_smooth[p][:,i]) for i in range(BGI_smooth[p].shape[1])]).transpose() for p in planes}
    moment_smooth_sigma = {p: moment_smooth_all[p][1,:] for p in planes}
    moment_smooth_mu    = {p: moment_smooth_all[p][0,:] for p in planes}

    # return everything in a dictionary
    output['FWHM_all']           = FWHM_all
    output['FWHM']               = FWHM      
    output['Gauss_all']          = Gauss_all
    output['Gauss_sigma']        = Gauss_sigma
    output['Gauss_mu']           = Gauss_mu
    output['moment_all']         = moment_all 
    output['moment_sigma']       = moment_sigma
    output['moment_mu']          = moment_mu
    output['BGI_smooth']         = BGI_smooth
    output['FWHM_smooth_all']    = FWHM_smooth_all
    output['FWHM_smooth']        = FWHM_smooth     
    output['Gauss_smooth_all']   = Gauss_smooth_all
    output['Gauss_smooth_sigma'] = Gauss_smooth_sigma
    output['Gauss_smooth_mu']    = Gauss_smooth_mu
    output['moment_smooth_all']  = moment_smooth_all 
    output['moment_smooth_sigma']= moment_smooth_sigma
    output['moment_smooth_mu']   = moment_smooth_mu
    
    return output

def treatSchottkyData(values, settings, return_p=True, ring_C_m=ELENA_C_m):
    _frequency_resolution = settings['window1FrequencyResolution_hz']['JAPC_ENUM']['string']
    if _frequency_resolution == 'FRES_48HZ82' : bin_width = 48.82; time_step = 20.48 + 0.5
    if _frequency_resolution == 'FRES_24HZ41' : bin_width = 24.41; time_step = 40.96 + 0.5
    if _frequency_resolution == 'FRES_12HZ21' : bin_width = 12.21; time_step = 81.92 + 0.5
    if _frequency_resolution == 'FRES_06HZ10' : bin_width = 6.1  ; time_step = 163.84 + 0.5
    if _frequency_resolution == 'FRES_03HZ05' : bin_width = 3.05 ; time_step = 327.68 + 0.5
    if _frequency_resolution == 'FRES_01HZ53' : bin_width = 1.53 ; time_step = 655.36 + 0.5

    psd = np.copy(values['schottkyWindow1Spectra'])
    #psd = np.copy(values['psd'])
    number_of_ffts, number_of_fft_bins = np.shape(psd)

    _windows_harmonic = int(settings['window1Harmonic']['JAPC_ENUM']['string'][1:])
    time_ms = np.arange(settings['window1StartTime_ms'], settings['window1StartTime_ms'] + time_step * number_of_ffts, time_step )
    freq_array = np.arange((settings['window1Frev_hz'] * _windows_harmonic) - np.trunc(bin_width * number_of_fft_bins/2),
                               (settings['window1Frev_hz'] * _windows_harmonic) + np.trunc(bin_width * number_of_fft_bins/2),
                                 bin_width)
    if return_p:
        return [psd.transpose(), time_ms, convert_f_to_p(freq_array, ring_C_m, h=_windows_harmonic)]
    else:
        return [psd.transpose(), time_ms, freq_array]

def analyseProfileEvolution(data, pos_array):
    '''
    It takes a 2D array of data which is supposed to include the evolution of a profile (transverse of longitudinal) 
    as a function of time, where time is the "column" axis of the 2D data.

    It returns a dict() with analysis of each profile in terms of Gaussian fit or other methods
 
    :param data: is a NxM data
    :param pos_array: is a N array with position coordinate (e.g. horizontal, vertical, longitudinal, ...)  
    '''

    # Do computations:
    #  - treat as simple FWHM
    FWHM_all = np.array([compute_single_fwhmfit(pos_array, data[:,i]) for i in range(data.shape[1])]).transpose()
    FWHM     = FWHM_all[2,:]
    #  - do Gaussian fits
    Gauss_all   = np.array([compute_single_gaussianfit(pos_array, data[:,i]) for i in range(data.shape[1])]).transpose()
    Gauss_sigma = Gauss_all[2,:]
    Gauss_mu    = Gauss_all[1,:]
    #  - do Moment fits
    moment_all   = np.array([compute_single_momentfit(pos_array, data[:,i]) for i in range(data.shape[1])]).transpose()
    moment_sigma = moment_all[1,:] 
    moment_mu    = moment_all[0,:] 

    # return everything in a dictionary
    output = dict()
    output['FWHM_all']           = FWHM_all
    output['FWHM']               = FWHM      
    output['Gauss_all']          = Gauss_all
    output['Gauss_sigma']        = Gauss_sigma
    output['Gauss_mu']           = Gauss_mu
    output['moment_all']         = moment_all 
    output['moment_sigma']       = moment_sigma
    output['moment_mu']          = moment_mu
    return output


def generate_smooth_sin(L_ms, A, N, npoints):
    '''
    It generates a small sinusoid of amplitude A, with N periods over a time lenght of L_ms with npoints points
    It returns a tuple with x and y values
    :param L_ms:
    :param A:
    :param N:
    :param npoints:
    :return:
    '''
    # at pi/2 derivative of sin is 1
    dataX = np.linspace((np.pi / 4 - 2), 2*np.pi*N - (np.pi / 4 - 2), npoints)
    dataY = np.sin(dataX)
    auxIdx = dataX < np.pi / 4
    dataY[auxIdx] = (np.sqrt(2)/8)*(dataX[auxIdx] - (np.pi/4 - 2))**2

    auxIdx = dataX > (2*np.pi*N - np.pi / 4)
    dataY[auxIdx] = - (np.sqrt(2) / 8) * (dataX[auxIdx]  -  (2*np.pi*N - (np.pi / 4 - 2))) ** 2

    # now scale everything
    dataX = dataX - dataX[0]
    dataX = (L_ms/dataX[-1])*dataX
    dataY = A*dataY
    return (dataX, dataY)


def add_functions(A, B):
    '''
    take two functions (i.e. dict with X and Y fields) and adds them together for each points defined in both functions
    A new function is returned

    :param A:
    :param B:
    :return:
    '''
    allTime = np.sort(np.concatenate([A['X'], B['X']]))

    Aint = interp1d(A['X'], A['Y'], fill_value=(0, 0), bounds_error=False)
    Bint = interp1d(B['X'], B['Y'], fill_value=(0, 0), bounds_error=False)
    C = dict()
    C['X'] = allTime
    C['Y'] = Aint(C['X'])+Bint(C['X'])

    return C

def imagesc(im, x = None, y = None, **kwargs):
    """
    Wrapper for PyPlot's `imshow` to imitate Matlab-style IMAGESC.

    `imagesc(z; x, y)` treats `z` as a 2D array to visualize, with `x` giving pixel coordinates
    across a row and `y` giving pixel columns *down* a column.

    Omitting `x` and/or `y` implies `1:size(z, 2)` and `1:size(z, 1)` respectively.

    PyPlot will show the image using a carefully-constructed call to `PyPlot.imshow` where:
    - the extent is carefully initialized so the plot's ticks line up exactly with `x` and `y`
    - the origin is at the lower-left of the window
    - the aspect ratio is fluid (uses the full window)
    - no interpolation is applied.
    """
    if x is None:
        x = np.arange(np.shape(im)[1])
    if y is None:
        y = np.arange(np.shape(im)[0])

    def _extents(f):
        delta = f[2] - f[1]
        return [f[1] - delta / 2, f[-1] + delta / 2]
    plt.imshow(im, extent=(_extents(x)[0], _extents(x)[-1], _extents(y)[0],  _extents(y)[-1]), aspect="auto", origin="lower", interpolation="none", **kwargs)


def compute_single_moment( x, y, mu=0, n=1, guess_offset=True):
    '''
    Returns the n-th moment of y(x) data with respect to mu.
    It can be used to compute the mean value and variance of a distribution-like function

    Example:
        data = np.random.randn(130000)*0.7 + 0.15

        histo = np.histogram(data, bins=20)
        x = (histo[1][1:] + histo[1][0:-1] )/ 2
        y = histo[0]

        mean = compute_single_moment(x,y)
        sigma = np.sqrt(compute_single_moment(x,y, mu=mu, n=2))

    :param x: x-coordinate of hist-like data
    :param y: count-value of hist-like data
    :param mu: reference of the moment, default = 0
    :param n: moment ordinal, default =1
    :return: the n-th moment of y(x) data with respect to mu
    '''
    if guess_offset:
        auxY = y - np.min(y)
    else:
        auxY = y
    return np.sum((x - mu) ** n * auxY) / np.sum(auxY)

def compute_single_momentfit(x, y, guess_offset=True):
    '''
    It utilizes first and second moment analysis of the data to extract mean and rms
    of a distribution.

    :param guess_offset: if to bring the signal down to zero, default=True
    '''
    mu = compute_single_moment(x, y, guess_offset=guess_offset)
    rms = np.sqrt(compute_single_moment(x, y, mu=mu, n=2, guess_offset=guess_offset))
    return [mu, rms]

def compute_single_fwhmfit(x, y):
    '''
    Returns the [amplitude, mean, FWHM, baseline_offset] of a given distribution
    just from basic FWHM analysis. 

    Note: NOT very precise at the moment! Davide, Oct 2021
    '''
    indx_max = np.argmax(y)
    mu0 = x[indx_max]
    offs0 = min(y)
    ampl = max(y) - offs0
    x1 = x[np.searchsorted(y[:indx_max], offs0 + ampl / 2)]
    x2_idx = indx_max + np.searchsorted(-y[indx_max:], -offs0 - ampl / 2)
    if x2_idx >= len(x):
        x2 = x[-1]
    else:
        x2 = x[x2_idx]
    FWHM0 = x2 - x1
    return [ampl, mu0, FWHM0, offs0]

def Gaussian(x, p=[1/(np.sqrt(2*np.pi)*1),0,1,0]):
    '''
    Returns y coordinate of Gaussian distribution for given p=[amplitude, mean, sigma, baseline_offset]
    '''
    return p[3] + p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))

def compute_single_gaussianfit(x, y):
    '''
    Returns the [amplitude, mean, sigma, baseline_offset] of a given distribution
    just from least-square fit of Gaussian distribution.

    '''
    _fitfunc = lambda p, x: p[3] + p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))
    _errfunc = lambda p, x, y: _fitfunc(p, x) - y

    # Guess parameters from FWHM function:
    p0 = compute_single_fwhmfit(x, y)
    # convert FWHM into sigma
    p0[2] = p0[2] / 2.355
    
    try:
        # Do the least square fit:
        p1, success = op.leastsq(_errfunc, p0[:], args=(x, y))
        if not success:
            raise ValueError
    except:
        print('Impossible to compute Gaussian fit on give data... ')
        p1 = 4 * [np.nan]
    return p1


def convert_f_to_p(f_Hz, C_m, h=1, m0_MeV_c2 = mp_MeV_c2):
    '''
    It convert an RF frequency into momentum assuming a given circumference

    :param f_Hz: frequency in Hz to convert
    :param C_m: ring circumference in m 
    :param h: harmonic of the frequency (default 1)
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: the converted frequency in momentum in MeV/c
    '''
    beta    = (f_Hz/h) * C_m / clight_m_s
    gamma   = 1/np.sqrt(1-beta**2)
    p_MeV_c = m0_MeV_c2*beta*gamma
    return p_MeV_c

def convert_p_to_f(p_MeV_c, C_m, m0_MeV_c2 = mp_MeV_c2):
    '''
    It convert an beam momentum to revolution frequency assuming a given circumference

    :param p_MeV_c: momentum of the particle in MeV/c
    :param C_m: ring circumference in m 
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: the converted momentum in revolution frequency Hz
    '''

    beta = p_to_beta(p_MeV_c, m0_MeV_c2)
    f_Hz = (beta*clight_m_s) / C_m
    return f_Hz

def convert_f_to_dp(f_Hz, f0_Hz, C0_m, gamma_tr, h=1, m0_MeV_c2 = mp_MeV_c2):
    '''
    It convert an Schottky frequency into momentum deviation, path length deviation,
    and central momentum for a given circumference, central revolution frequency, 
    gamma transition, harmonic, and rest mass of the particle.

    :param f_Hz: frequency in Hz to convert
    :param f0_Hz: central frequency at given h
    :param C0_m: ring central circumference in m 
    :param gamma_tr: gamma transition: needed for proper computation of momentum due to off-energy different path length 
    :param h: harmonic of the frequency (default 1)
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: (df_f0, dp_p0, dC_C0, p0_MeV_c)
    '''
    beta_0    = (f0_Hz/h) * C0_m / clight_m_s
    gamma_0   = 1/np.sqrt(1-beta_0**2)
    p0_MeV_c = m0_MeV_c2*beta_0*gamma_0

    alpha = 1/gamma_tr**2
    slip_factor = 1/gamma_0**2 - alpha

    df_f0 = (f_Hz-f0_Hz)/f0_Hz

    dp_p0 = (1/slip_factor)*(df_f0/h)
    # correction - Dec 2022
    dp_p0 = (1/slip_factor)*(df_f0)
    dC_C0 = alpha * dp_p0
    return (df_f0, dp_p0, dC_C0, p0_MeV_c)

def compute_delta_v(p0_MeV_c, p1_MeV_c, m0_MeV_c2 = mp_MeV_c2):
    '''
    computes delta velocity in m/s starting from delta moment in MeV/c and assuming a given rest mass

    :param p0_MeV_c: reference momentum in MeV/c
    :param p1_MeV_c: momentum in MeV/c
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: V(p1_MeV_c)-V(p0_MeV_c) in m/s
    '''

    auxP = p0_MeV_c
    auxGamma = np.sqrt((auxP/m0_MeV_c2)**2 + 1)
    auxBeta = np.sqrt(auxGamma**2 - 1)/auxGamma
    auxV0_m_s = auxBeta*clight_m_s

    auxP = p1_MeV_c
    auxGamma = np.sqrt((auxP/m0_MeV_c2)**2 + 1)
    auxBeta = np.sqrt(auxGamma**2 - 1)/auxGamma
    auxVf_m_s = auxBeta*clight_m_s

    return auxVf_m_s-auxV0_m_s


def p_to_gamma(p_MeV_c, m0_MeV_c2 = mp_MeV_c2):
    '''
    converts a given momentum in MeV/c into relativistic gamma assuming a given rest mass

    :param p_MeV_c:  momentum in MeV/c
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: relativistic gamma
    '''
    return np.sqrt((p_MeV_c/m0_MeV_c2)**2 + 1)

def p_to_beta(p_MeV_c, m0_MeV_c2 = mp_MeV_c2):
    '''
    converts a given momentum in MeV/c into relativistic beta assuming a given rest mass

    :param p_MeV_c: momentum in MeV/c
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: relativistic beta
    '''
    auxGamma = p_to_gamma(p_MeV_c, m0_MeV_c2)
    auxBeta = np.sqrt(auxGamma**2 - 1)/auxGamma
    return auxBeta

def treat_BCCCA_fast(data, assumed_Q0 = 5e7, iterative_adjust_Q0 = False, iterative_adjust_threshold = 1e6, p0_MeV_c=3575):
    '''
    If we would have the actual current, and momenta, then everything should look like:
        _p_m = _p/mp_MeV_c2
        _beta_rel = _p_m / np.sqrt(_p_m**2 + 1)
        q_charges = _I * (AD_C_m / clight_m_s) * (1 / e_C) * (1 / _beta_rel)
         - or -
        _I = q_charges * (clight_m_s * e_C / AD_C_m) * _beta_rel

    Instead, we have only the delta current from an initial zero, i.e.
        DeltaI_t = I_t - I_0 where I_0 is in principle the current that corresponds at injection energy to the injected charges Q_0 (unknown)

    Let's then call Q_i = Q_0 - DeltaQ_i :
        DeltaI_t = (Q_0 - DeltaQ_t) * (clight_m_s * e_C / AD_C_m) * _beta_rel_t - Q_0 * (clight_m_s * e_C / AD_C_m) * _beta_rel_0
                 = - DeltaQ_t * (clight_m_s * e_C / AD_C_m) * _beta_rel_t + Q_0 * (clight_m_s * e_C / AD_C_m) * (_beta_rel_t -_beta_rel_0)
        DeltaI_t * ( AD_C_m / (clight_m_s * e_C) ) = Q_0 * (_beta_rel_t -_beta_rel_0) - DeltaQ_t * _beta_rel_t
    From last equation:
        - until beta_rel is not changing, only explanation is DeltaQ_t -> i.e. losses
        - when beta_rel has changed, and assuming an initial Q_0, then two options to explain DeltaI_t:
            - either we have lost part of the beam (DeltaQ_t increasing)
            - or we the initial charge is wrong

    The default behavior, is to assume an initial Q_0 (from the user, default 5e7).
    If one has a history data, in which momenta is changing, then one can try iteratively to find the actual Q_0 by assuming that losses cannot DECREASE

    :param data: acquisition-like data from 'ADE.BCCCA/FastAcquisition'. If it contains an array, iterative proccess can be possible.
    :param assumed_Q0: initial intensity - assumed_Q0 = 5e7
    :param iterative_adjust_Q0: (default False)
    :param iterative_adjust_threshold:  (default 1e6)
    :param p0_MeV_c: injection energy assumed
    :return:
    '''

    _deltaI_t   = data['fastDataDeltaCurrentA']
    _p_t        = data['fastDataMomentumMevPerC']
    _p_m_t      = _p_t/mp_MeV_c2
    _beta_t     = _p_m_t / np.sqrt(np.power(_p_m_t, 2) + 1)
    _p0_m       = p0_MeV_c/mp_MeV_c2
    _beta_0     = _p0_m / np.sqrt(np.power(_p0_m, 2) + 1)


    if iterative_adjust_Q0:
        # assuming that we cannot "gain" particles along the cycle...
        print('asdf')

        # data must be an array...
        if len(np.shape(_deltaI_t)) == 0:
            raise ValueError('for iterative process, you need arrays of data...')

        # prepare output
        DeltaQ_t = np.zeros(np.shape(_deltaI_t))
        Q_0 = np.zeros(np.shape(_deltaI_t))

        for i, deltaI in enumerate(_deltaI_t):
            if i == 0:
                Q_0[i] = assumed_Q0
                DeltaQ_t[i] = 0
                continue
            Q_0[i] = Q_0[i-1]
            DeltaQ_t[:i+1] = Q_0[i] * (_beta_t[:i+1] -_beta_0)/_beta_t[:i+1] - (AD_C_m / (clight_m_s * e_C)) * _deltaI_t[:i+1]/_beta_t[:i+1]

            # if this threshold is too low, then system is unstable
            _loop_protection = 0
            while DeltaQ_t[i] < (np.max(DeltaQ_t[:i+1]) - iterative_adjust_threshold):
                Q_0[i] = Q_0[i] - 1e6
                DeltaQ_t[:i + 1] = Q_0[i] * (_beta_t[:i + 1] - _beta_0) / _beta_t[:i + 1] - ( AD_C_m / (clight_m_s * e_C)) * _deltaI_t[:i + 1] / _beta_t[:i + 1]

                # protect by too many loops
                _loop_protection = _loop_protection+1
                if (_loop_protection > 100) or (Q_0[i] < 0):
                    # restart from assumed Q0 and continue...
                    Q_0[i] = assumed_Q0
                    break
    else:
        # simply use assumed intensity from user
        Q_0 = assumed_Q0
        DeltaQ_t = assumed_Q0 * (_beta_t - _beta_0) / _beta_t - (AD_C_m / (clight_m_s * e_C)) * _deltaI_t / _beta_t

    return (Q_0, DeltaQ_t)


def gamma_to_Ek(gamma_rel, m0_MeV_c2 = mp_MeV_c2):
    '''
    converts a given relativistic gamma into kinetic energy in MeV assuming a given rest mass

    :param gamma_rel:  input relativistic gamma 
    :param m0_MeV_c2: rest mass of the particle in MeV/c^2 (default mass of proton)
    :return: kinetic energy in MeV
    '''
    return (gamma_rel - 1) * m0_MeV_c2

def setup_nice_plots(small_figure = False, usetex = False):
    '''
    Setup matplotlib to use latex for output.

    #plt.savefig(f'../final_figures/bpm_compare1.png', format='png', dpi=300)
    #plt.savefig(f'../../figures/ofb_study/single_bpms.eps', format='eps', dpi=400)
    plt.savefig(f'../figures/bpm_study/single_bpms.pgf', dpi=400)

    Using code/style by Joel Andersson.
    '''

    # 
    pgf_big_pic = {
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": usetex,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 11,
        "font.size": 11,
        "legend.fontsize": 10,               # Make the legend/label fonts
        "xtick.labelsize": 10,               # a little smaller
        "ytick.labelsize": 10,
        "figure.figsize": [8.0, 4.0],     # default fig size of 0.9 textwidth
        'figure.constrained_layout.use': True, ## When True, automatically make plot
                                            # elements fit on the figure. (Not compatible
                                            # with `autolayout`, above).
        'figure.constrained_layout.h_pad': 3./72, ## Padding around axes  objects. Float representing
        'figure.constrained_layout.w_pad': 3./72, ##  inches. Default is 3./72. inches (3 pts)

        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            #r"\usepackage{CormorantGaramond}",  # Next 3 lines enforces my CormorantGaramond font!
            #r"\let\oldnormalfont\normalfont",
            #r"\def\normalfont{\oldnormalfont\mdseries}"
            ] }

    pgf_small_pic = {
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": usetex,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 9,
        "font.size": 7,
        "legend.fontsize": 7,               # Make the legend/label fonts
        "xtick.labelsize": 7,               # a little smaller
        "ytick.labelsize": 7,
        "figure.figsize": [2.7, 2.35],     # default fig size of 0.9 textwidth
        'figure.constrained_layout.use': True, ## When True, automatically make plot
                                            # elements fit on the figure. (Not compatible
                                            # with `autolayout`, above).
        'figure.constrained_layout.h_pad': 3./72, ## Padding around axes  objects. Float representing
        'figure.constrained_layout.w_pad': 3./72, ##  inches. Default is 3./72. inches (3 pts)

        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            #r"\usepackage{CormorantGaramond}",  # Next 3 lines enforces my CormorantGaramond font!
            #r"\let\oldnormalfont\normalfont",
            #r"\def\normalfont{\oldnormalfont\mdseries}"
            ] }
    
    if small_figure:
        mpl.rcParams.update(pgf_small_pic)
    else:
        mpl.rcParams.update(pgf_big_pic)


# unroll function list somehow
def unroll_AD_function_list(functionlist, cycle_description=None):
    if cycle_description is not None:
        ramp_starts = ad_structure.expand_xml(cycle_description).rampStarts
        if len(ramp_starts) == (len(functionlist) - 1):
            # for some unknown reasons to me, some functions come with a meaningless first segment.
            # ... to be checked with Lajos
            print('Warning: number of functions is not the same as what in cycle description, but probably due to additoinal first useless segment. Will try to continue.')
            ramp_starts = np.insert(ramp_starts, 0, -1)
        elif len(ramp_starts) != len(functionlist):
            raise ValueError(f'Number of functions ({len(functionlist)}) is not compatible with provided cycle description ({len(ramp_starts)} ramps expected)')
    else:
        ramp_starts = np.zeros(len(functionlist))

    output = {'X': np.array([]), 'Y': np.array([])}
    for i, piece in enumerate(functionlist):
        auxX = piece['JAPC_FUNCTION']['X']
        auxY = piece['JAPC_FUNCTION']['Y']
        if len(output['X']) == 0:
            output['X'] = auxX
            output['Y'] = auxY
        elif len(output['X']) > 0:
            if cycle_description is not None:
                auxX = auxX + ramp_starts[i]
            else:
                auxX = auxX + output['X'][-1]
            output['X'] = np.append(output['X'], auxX)
            output['Y'] = np.append(output['Y'], auxY)
        else:
            print('Something wrong here - check `unroll_AD_function_list()` code!')
    return output


def extract_events(timing_description):
    '''Extract events and their time in a cycle.
    Works for both AD and ELENA:
       - 'DXC.CYCLEDEF-CT/CycleDefinition#xml'
       - 'AXC.CYCLEDEF-CT/CycleDefinition#xml'
    output is a dict with all events name, and the associated offset time
    '''
    all_events = dict()
    root = ET.fromstring(timing_description)
    _start_time = 0
    for section in root.find('cycle').findall('section'):
        for event in section.find('timingEvents'):
            all_events[event.get('name')] = _start_time + int(event.get('offset'))
        # increase start time by segment duration
        _start_time = _start_time + int(section.get('duration'))
    return all_events

def moving_average(data, n_samples = 1):
    '''
    moving_average(data, n_samples = 1)
    Returns the same data but with a moving average of n_samples length
    '''
    return np.convolve(data, np.ones(n_samples), 'same') / n_samples

    
#### APPENDIX
# TO SELF-RELOAD
def help_functions_reload():
    from importlib import reload # python 2.7 does not require this
    import help_functions
    reload( help_functions )
