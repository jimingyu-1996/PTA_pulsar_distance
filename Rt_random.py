import numpy as np
import astropy.units as u
import astropy.constants as c
from time_residual import time_residual, time_residual_num, time_residual_deltaphi
from fisher_SNR_30 import fisher

def Rt(**kw):
    np.random.rand(100)
    psi_c = kw['phi_c']
    omega = kw['omega']
    Mc = kw['Mc']
    psi = kw['psi']
    T = kw['T']
    num = kw['num']
    D = []
    pulsars = kw['pulsars']
    for p in list(pulsars.keys()):
        D.append(pulsars[p]['D_corr'])

    paras = kw.copy()
    events = fisher(**paras)
    paras['logdL'] = np.log(events.dL())
    paras['T'] = T
    paras['num'] = num
    rt = []
    rt_random = []
    for p in list(pulsars.keys()):
        paras['pulsar'] = pulsars[p].copy()
        ri = time_residual(**paras)
        rt_p = ri.Rt_bi()
        rt.append(rt_p)
        noise = ((pulsars[p]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
        rt_random_p = np.random.normal(loc=rt_p, scale=noise)
        rt_random.append(rt_random_p)
        
    return rt_random, rt 

def Rt_num(**kw):
    np.random.rand(100)
    psi_0 = kw['phi_0']
    omega = kw['omega']
    Mc = kw['Mc']
    psi = kw['psi']
    T = kw['T']
    D = []
    num = kw['num']
    pulsars = kw['pulsars']
    for p in list(pulsars.keys()):
        D.append(pulsars[p]['D_corr'])

    paras = kw.copy()
    T_num = num
    paras['T'] = T
    paras['num']=T_num
    rt = []
    rt_random = []
    for p in list(pulsars.keys()):
        paras['pulsar'] = pulsars[p].copy()
        ri = time_residual_num(**paras)
        rt_p = ri.Rt_bi()
        rt.append(rt_p)
        noise = ((pulsars[p]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
        rt_random_p = np.random.normal(loc=rt_p, scale=noise)
        rt_random.append(rt_random_p)
        
    return rt_random, rt 

def Rt_mono(**kw):
    np.random.rand(100)
    psi_0 = kw['phi_0']
    omega_earth = kw['omega_earth']
    Mc = kw['Mc']
    psi = kw['psi']
    T = kw['T']
    T_num = kw['num']
    D = []
    pulsars = kw['pulsars']
    delta_phi_0 = kw['delta_phi_0']
    for p in list(pulsars.keys()):
        D.append(pulsars[p]['D_corr'])

    paras = kw.copy()
    events = fisher(**paras)
    paras['logdL'] = np.log(events.dL())
    paras['T'] = T
    paras['num']=T_num
    rt = []
    rt_random = []
    for p in list(pulsars.keys()):
        paras['pulsar'] = pulsars[p].copy()
        ri = time_residual_deltaphi(**paras)
        rt_p = ri.Rt_mono()
        rt.append(rt_p)
        noise = ((pulsars[p]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
        rt_random_p = np.random.normal(loc=rt_p, scale=noise)
        rt_random.append(rt_random_p)
        
    return rt_random, rt 
