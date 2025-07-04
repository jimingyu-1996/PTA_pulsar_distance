import numpy as np
import astropy.units as u
import astropy.constants as c
from time_residual import time_residual, time_residual_num, time_residual_deltaphi
from fisher_SNR_30 import fisher
from bilby import Likelihood

class Residualtime(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={key: None for key in D_keys}
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega': None, 'Mc': None, 'phi_c': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['pulsar']['D_corr'] = self.parameters[pulsar]
            paras['T'] = self.T
            paras['num'] = self.num
            ri = time_residual(**paras)
            rt_p = ri.Rt_bi()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_multi(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        self.events_num = kw['events_num']
        keys = kw['keys'].copy()
        self.keys_source = kw['keys_source'].copy()
        parameters={key: None for key in keys}
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for i in range(self.events_num):
            para_i = paras.copy()
            para_i['T'] = self.T[i]
            para_i['num'] = self.num[i]
            for key in list(self.keys_source.keys()):
                key_para = self.keys_source[key][i]
                para_i[key] = paras[key_para]
            for p, pulsar in enumerate(list(self.pulsars.keys())):
                para_i['pulsar'] = self.pulsars[pulsar].copy()
                para_i['pulsar']['D_corr'] = self.parameters[pulsar]
                ri = time_residual(**para_i)
                rt_p = ri.Rt_bi()
                noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
                res = rt_p - self.rt_data[i][p]
                log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_num(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={key: None for key in D_keys}
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega': None, 'Mc': None, 'phi_0': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['pulsar']['D_corr'] = paras['pulsar']['D_corr'] + self.parameters[pulsar]
            paras['T'] = self.T
            paras['num'] = self.num
            if 'phi_p' in paras:
                paras.pop('phi_p')
            ri = time_residual_num(**paras)
            rt_p = ri.Rt_bi()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_multi_num(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        self.events_num = kw['events_num']
        keys = kw['keys'].copy()
        self.keys_source = kw['keys_source'].copy()
        parameters={key: None for key in keys}
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for i in range(self.events_num):
            para_i = paras.copy()
            para_i['T'] = self.T[i]
            para_i['num'] = self.num[i]
            for key in list(self.keys_source.keys()):
                key_para = self.keys_source[key][i]
                para_i[key] = paras[key_para]
            for p, pulsar in enumerate(list(self.pulsars.keys())):
                para_i['pulsar'] = self.pulsars[pulsar].copy()
                para_i['pulsar']['D_corr'] = self.parameters[pulsar]
                if 'phi_p' in para_i:
                    para_i.pop('phi_p')
                ri = time_residual_num(**para_i)
                rt_p = ri.Rt_bi()
                noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
                res = rt_p - self.rt_data[i][p]
                log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
    
class Residualtime_num_phi_p(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={}
        for key in D_keys:
            phi_key = key + ':phi'
            parameters[phi_key] = None
            omega_key = key + ':omega'
            parameters[omega_key] = None
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega_earth': None, 'Mc': None, 'phi_0': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['T'] = self.T
            paras['num'] = self.num
            phi_key = pulsar + ':phi'
            omega_key = pulsar + ':omega'
            paras['delta_phi_0'] = self.parameters[phi_key]
            paras['omega_pulsar'] = self.parameters[omega_key]
            ri = time_residual_deltaphi(**paras)
            rt_p = ri.Rt_bi()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_phi_Domega_num(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={}
        for key in D_keys:
            phi_key = key + ':phi'
            parameters[phi_key] = None
            omega_key = key + ':Domega'
            parameters[omega_key] = None
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega_earth': None, 'Mc': None, 'phi_0': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['T'] = self.T
            paras['num'] = self.num
            phi_key = pulsar + ':phi'
            omega_key = pulsar + ':Domega'
            paras['delta_phi_0'] = self.parameters[phi_key]
            paras['Domega'] = self.parameters[omega_key]
            ri = time_residual_deltaphi(**paras)
            rt_p = ri.Rt_bi()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_phi_Deltat_num(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={}
        for key in D_keys:
            phi_key = key + ':phi'
            parameters[phi_key] = None
            D_key = key + ':D'
            parameters[D_key] = None
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega_earth': None, 'Mc': None, 'phi_0': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['T'] = self.T
            paras['num'] = self.num
            phi_key = pulsar + ':phi'
            D_key = pulsar + ':D'
            paras['delta_phi_0'] = self.parameters[phi_key]
            #paras['Domega'] = self.parameters[omega_key]
            ri = time_residual_deltaphi(**paras)
            rt_p = ri.Rt_bi()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
    
class Residualtime_phi_mono(Likelihood):

    def __init__(self, rt_data, **kw):
        self.rt_data = rt_data
        self.num = kw['num']
        self.T = kw['T']
        self.pulsars = kw['pulsars'].copy()
        D_keys = list(self.pulsars.keys())
        parameters={}
        for key in D_keys:
            phi_key = key + ':phi'
            parameters[phi_key] = None
#            D_key = key + ':D'
#            parameters[D_key] = None
        parameters_global = {'alpha': None, 'delta': None, 'psi': None, 'iota': None, 'omega_earth': None, 'Mc': None, 'phi_0': None, 'logdL': None}
        parameters.update(parameters_global)
        super().__init__(parameters=parameters)

        
    def log_likelihood(self):
        log_p = 0.
        paras = self.parameters.copy()
        for p, pulsar in enumerate(list(self.pulsars.keys())):
            paras = self.parameters.copy()
            paras['pulsar'] = self.pulsars[pulsar].copy()
            paras['T'] = self.T
            paras['num'] = self.num
            phi_key = pulsar + ':phi'
            #D_key = pulsar + ':D'
            paras['delta_phi_0'] = self.parameters[phi_key]
            #paras['Domega'] = self.parameters[omega_key]
            ri = time_residual_deltaphi(**paras)
            rt_p = ri.Rt_mono()
            noise = ((self.pulsars[pulsar]['noise'] / 2. / 2.0 / u.wk)**0.5).cgs.value
            res = rt_p - self.rt_data[p]
            log_p = log_p - 0.5 * np.sum((res / noise) ** 2)

        return log_p
    
