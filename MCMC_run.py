import argparse
import numpy as np
import emcee
import matplotlib.pyplot as plt
#import corner
import astropy.units as u
import astropy.constants as c
from time_residual import time_residual, time_residual_deltaphi, time_residual_num
import Rt_random
from fisher_num_SNR_30 import fisher
import json
from bilby import run_sampler
from Rt_log_likelihood import Residualtime_phi_Deltat_num
from bilby.core.prior import Uniform, LogUniform, Sine, Gaussian
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--npool', type=int, default=1)
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--n', type=int, nargs=2, default=[0, 1])
parser.add_argument('--outdir', type=str)
parser.add_argument('--label', type=str, default='PTA')
parser.add_argument('--omega', type=float, default=300.)
parser.add_argument('--Mc', type=float, default=1000.)
parser.add_argument('--T', type=float, default=10.)
parser.add_argument('--dL', type=float, default=50.)    
args = parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
outdir = args.outdir
n_start, n_end = args.n
if n_start >= n_end:
    raise Exception("n_start should be smaller that n_end.")
if n_end > args.N:
    raise Exception("n_end should be smaller that N.")

f = open("pulsar_EPTA.json")
pulsars = json.load(f)

parameters_global = ['alpha', 'delta', 'psi', 'iota',
                                'Mc', 'omega_earth', 'logdL', 'phi_0']
parameters_pulsar = []
for i, pulsar in enumerate(list(pulsars.keys())):
    pulsars[pulsar]['noise'] = ((2. * 1.e-8)**2. * 2. * 2. * u.wk).cgs.value
    if i > 20:
        pulsars.pop(pulsar)
    else:
        parameters_pulsar.append(pulsar)
num_pulsar = len(list(pulsars.keys()))
num_parameter = num_pulsar + 8

np.random.seed(args.seed)

np.random.rand(100)
n_BBH = args.N
SMBBH_alpha = np.random.rand(n_BBH) * np.pi * 2.
SMBBH_delta = np.arccos(np.random.rand(n_BBH) * 2. - 1.) - np.pi / 2.
SMBBH_iota = np.arccos(np.random.rand(n_BBH) * 2. - 1.)
SMBBH_psi = np.random.rand(n_BBH) * np.pi * 2.
SMBBH_phi_0 = np.random.rand(n_BBH) * np.pi * 2.
SMBBH_omega = np.ones(n_BBH) * args.omega
SMBBH_Mc = np.ones(n_BBH) * args.Mc
SMBBH_dL = np.ones(n_BBH) * args.dL

paras = []
for i in range(n_BBH):
    para_i = {}
    para_i['dL'] = SMBBH_dL[i]
    para_i['logdL'] = np.log(SMBBH_dL[i])
    para_i['alpha'] = SMBBH_alpha[i]
    para_i['delta'] = SMBBH_delta[i]
    para_i['T'] = args.T 
    para_i['psi'] = SMBBH_psi[i]
    para_i['iota'] = SMBBH_iota[i]
    para_i['num'] = 1000 
    para_i['Mc'] = SMBBH_Mc[i] 
    para_i['omega'] = SMBBH_omega[i]
    para_i['omega_earth'] = SMBBH_omega[i]
    para_i['phi_0'] = SMBBH_phi_0[i]
    para_i['phi_c'] = SMBBH_phi_0[i]
    para_i['pulsars'] = pulsars
    para_i['num'] =  int((para_i['T'] * u.a  / 2.0 / u.wk).cgs.value) + 1
    para_i['T'] = (2.0 * u.wk * (para_i['num'] - 1) / u.a).cgs.value
    snr2_sum = 0.
    for p in parameters_pulsar:
        para_i['pulsar'] = pulsars[p]
        events = time_residual_num(**para_i)
        snr_i = events.snr()
        #print(snr_i)
        snr2_sum = snr2_sum + snr_i**2
        noise = ((pulsars[p]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
        sigma_D = 2.34e-3 / np.cos(pulsars[p]['dec'])**2. * (para_i['num'] / 100.)**-0.5 * pulsars[p]['D_corr']**2. * (noise / 1.e-8)
        pulsars[p]['sigma_D'] = sigma_D 
    #para_i['logdL'] = np.log(np.exp(para_i['logdL']) / 30. * snr2_sum**0.5)
    paras.append(para_i)

if n_start > 0:    
    for i in range(n_start):
        para = paras[i]
        rt_random, rt = Rt_random.Rt_num(**para)


for i in range(n_start, n_end):
    para = paras[i]
    rt_random, rt = Rt_random.Rt_num(**para)
    injection_parameters = {}
    for pulsar in parameters_pulsar:
        para['pulsar'] = pulsars[pulsar].copy()
        #para['pulsar']['noise'] = ((1. * 1.e-8)**2. * 2. * 2. * u.wk).cgs.value
        events = time_residual(**para)
        r1 = events.Rt_bi()
        omega_earth, omega_pulsar = events.omega_bi()
        print(omega_pulsar)
        delta_t = events.delta_t
        #pulsars[pulsar]['Domega'] = omega_earth - omega_pulsar
        delta_phi_0 = para['omega_earth'] * 1.e-9 * delta_t - (3. / 5. * 2.**(4. / 3.)
                        * ((c.G * para['Mc'] * 1.e6 * c.M_sun / c.c**3).cgs.value)**(5. / 3.) * (para['omega_earth'] * 1.e-9)**(11. / 3.) * delta_t**2.)
        delta_phi_0 = delta_phi_0 - np.floor(delta_phi_0 / (2. * np.pi)) * 2. * np.pi
        pulsars[pulsar]['delta_phi_0'] = delta_phi_0
        
    
    
    for key in parameters_global:
        injection_parameters[key] = para[key]
        
    
    priors = {
        'alpha' : injection_parameters['alpha'],
        'delta' : injection_parameters['delta'],
        'iota' : Sine(latex_label=r'$\iota$'),
        'logdL' : injection_parameters['logdL'],
        'psi': Uniform(0., np.pi, latex_label=r'$\psi$', boundary='periodic'),
        'phi_0' : Uniform(0., 2.* np.pi, latex_label=r'$\phi_0$', boundary='periodic'),
        'omega_earth' : Uniform(0., 
                             injection_parameters['omega_earth'] * 2., 
                             latex_label=r'$\omega_\mathrm{e}$'),
        'Mc' : LogUniform(injection_parameters['Mc'] / 10.,
                          injection_parameters['Mc'] * 10., 
                          latex_label=r'$\mathcal{M}_c$')
    }
    for key in parameters_pulsar:
        phi_key = key + ':phi'
        D_key = key + ':D'
        injection_parameters[phi_key] = pulsars[key]['delta_phi_0']
        injection_parameters[D_key] = pulsars[key]['D_corr']
        priors[phi_key] = Uniform(0., 2. * np.pi, latex_label=phi_key, boundary='periodic')
        priors[D_key] = Gaussian(injection_parameters[D_key],  pulsars[key]['sigma_D'],
                             latex_label=D_key)
    likelihood = Residualtime_phi_Deltat_num(rt_random, **para)
    outdir_i = os.path.join(outdir, 'omega' + str(int(para['omega'])) + '_Mc' + str(int(para['Mc'])) + '_dL' + str(int(para['dL'] + 0.1)) + '_T' + str(int(para['T'] + 0.1)))
    if not os.path.isdir(outdir_i): os.makedirs(outdir_i)
    label = args.label + '_' + str(i)
    result = run_sampler(
        likelihood = likelihood,
        priors = priors,
        sample = "rwalk",
        injection_parameters = injection_parameters,
        label = label,
        outdir = outdir_i,
        npool = args.npool,
        nlive=1000,
        walks=144,
        nact=20,
        maxmcmc=10000,
    )
    corner_path = outdir_i + '/corner' + str(i) + '.png'
    result.plot_corner(save=True, filename=corner_path)
    json_path = outdir_i + '/injection_' + str(i) + '.json'
    with open(json_path, "w") as outfile: 
        json.dump(injection_parameters, outfile)
