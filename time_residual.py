#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import astropy.constants as c
from gw import gw_event
from pulsar import psr
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import math

class time_residual(object):

    def __init__(self, **kw):

        self._dL = (np.exp(kw['logdL']) * c.kpc * 1.0e3).cgs.value
#        self._Mc = (Mc * c.M_sun).cgs.value
        self._alpha = kw['alpha']
        self._delta = kw['delta']
        self._psi = kw['psi']
        self._iota = kw['iota']
        self._alpha_p = kw['pulsar']['ra'] #* np.pi / 180.
        self._delta_p = kw['pulsar']['dec'] #* np.pi / 180.
        self._D = (kw['pulsar']['D_corr'] * c.kpc).cgs.value
        self._c = c.c.cgs.value
        self._G = c.G.cgs.value
        self._a = (1. * u.a).cgs.value
        if 'num' in kw:
            self._n = kw['num']
        else:
            self._n = 1000

        if self._n == 1:
            self._T = np.array([(kw['T'] * u.a).cgs.value])
        else:
            self._T = np.linspace(0., kw['T'] * self._a, num=self._n)
            
        self._pulsars = psr(D=self._D, alpha=self._alpha, delta=self._delta, 
                           alpha_p = self._alpha_p, delta_p=self._delta_p)
        self.delta_t = self._pulsars.delta_t()


        if 'h0' in kw:
            self._h0 = kw['h0']
            if 'phi_0' in kw:
                self._phi_0 = kw['phi_0']
            else:
                self._phi_0 = 0.
            try:
                self._omega = kw['omega'] * 1.0e-9
            except KeyError:
                print("Error: no omega")
        elif 'Mc' in kw:
            self._Mc = (kw['Mc'] * 1.e6 * c.M_sun).cgs.value
            self._events = gw_event(dL=self._dL, Mc=self._Mc, iota=self._iota, psi=self._psi)
            if 't_to_merge' in kw:
                self._t = (1.0e6 * kw['t_to_merge'] * u.a).cgs.value
                self._omega = self._events.t_to_omega(self._t)
                self._h0 = self._events.t_to_h0(self._t)
            elif 'omega' in kw:
                self._omega = kw['omega'] * 1.0e-9
                self._h0 = self._events.omega_to_h0(self._omega)
                self._t = self._events.omega_to_t(self._omega)
            else: 
                raise Exception("Please import omega or time to merge")
            if 'tanphi_c' in kw:
                self._phi_c = np.arctan(kw['tanphi_c'] * 2.)
            else: 
                self._phi_c = 0.0
            self._phi_0 = self._events.t_to_phi(self._t, self._phi_c)
        elif 'omega_earth' and 'omega_pulsar' in kw:
            self._omega_earth = kw['omega_earth'] * 1.0e-9
            self._omega_pulsar = kw['omega_pulsar'] * 1.0e-9
            self._Mc = 5.**-0.6 * self._c**3, * (self._omega_earth**(-8. / 3.) 
                        - self._omega_pulsar**(-8. / 3.))**0.6 / 2.**-3.2 / self._G / self.delta_t**0.6
            self._events = gw_event(dL=self._dL, Mc=self._Mc, iota=self._iota, psi=self._psi)
        else:
            raise Exception("No GW waveform parameters")



    def h_t(self):
        
        t_list = self._t - self._T
        hplus, hcross = self._events.ht(t_list)
        
        return hplus, hcross

    def h0_t(self):
        
        t_list = self._t - self._T

        return self._events.t_to_h0(t_list)

    def T_series(self):

        return self._T

    def costheta(self):

        return self._pulsars.costheta()

    def Rt_mono(self):

        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        delta_t = self._pulsars.delta_t()
        Rt = - self._h0 / self._omega * np.sin(self._omega * delta_t / 2.) * (
             f_plus * (-cosiota * np.sin(2. * self._psi) 
             * np.sin(self._omega * self._T - self._omega * delta_t / 2. + self._phi_0)
           + 0.5 * (1. + cosiota**2) * np.cos(2. * self._psi)
             * np.cos(self._omega * self._T - self._omega * delta_t / 2. + self._phi_0))
           + f_cross * (cosiota * np.cos(2. * self._psi)
             * np.sin(self._omega * self._T - self._omega * delta_t / 2. + self._phi_0)
           + 0.5 * (1. + cosiota**2) * np.sin(2. * self._psi)
             * np.cos(self._omega * self._T - self._omega * delta_t / 2. + self._phi_0)))
       
        return Rt
    
    
    def Rt_bi(self):

        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        delta_t = self.delta_t
        
        t_earth = self._t - self._T
        t_pulsar = t_earth + delta_t
#        print((delta_t / u.a).cgs.value)
        phi_earth = self._events.t_to_phi(t_earth, self._phi_c)
        phi_pulsar = self._events.t_to_phi(t_pulsar, self._phi_c)
        omega_earth = self._events.t_to_omega(t_earth[0])
        omega_pulsar = self._events.t_to_omega(t_pulsar[0])
        self._omega_earth = omega_earth * 1.e9
        self._omega_pulsar = omega_pulsar * 1.e9
#        print(omega_earth, omega_pulsar)
#        dT = self._T[1] - self._T[2]
#        for i in range(self._n - 1):
#            phi_earth.append(phi_earth[-1] + omega_earth * dT)
#            phi_pulsar.append(phi_pulsar[-1] + omega_pulsar * dT)
        A_eff_earth = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * omega_earth**(-1./3.)
    
        A_eff_pulsar = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * omega_pulsar**(-1./3.)
        
        A_earth = np.cos(phi_earth) * A_eff_earth

        B_earth = -np.sin(phi_earth) * A_eff_earth

        A_pulsar = np.cos(phi_pulsar)* A_eff_pulsar

        B_pulsar = -np.sin(phi_pulsar)* A_eff_pulsar
        
        sin2psi = np.sin(2. * self._psi)
        cos2psi = np.cos(2. * self._psi)
        Rt = (cosiota * (f_plus * sin2psi - f_cross * cos2psi) * (
             A_pulsar - A_earth) - 
             (1. + cosiota**2) / 2. * (f_plus * cos2psi + f_cross * sin2psi) * (
             B_pulsar - B_earth))
        return Rt

    def Rt_bi_omega2(self):

        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        delta_t = self.delta_t
        
        t_earth = self._t - self._T
        t_pulsar = t_earth + delta_t
#        print((delta_t / u.a).cgs.value)
        phi_earth = self._events.t_to_phi(t_earth, self._phi_c)
        phi_pulsar = self._events.t_to_phi(t_pulsar, self._phi_c)
        omega_earth = self._omega_earth
        omega_pulsar = self._omega_pulsar
#        print(omega_earth, omega_pulsar)
#        dT = self._T[1] - self._T[2]
#        for i in range(self._n - 1):
#            phi_earth.append(phi_earth[-1] + omega_earth * dT)
#            phi_pulsar.append(phi_pulsar[-1] + omega_pulsar * dT)
        A_eff = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * omega_earth**(-1./3.)
        A_earth = np.cos(phi_earth)

        B_earth = -np.sin(phi_earth)

        A_pulsar = np.cos(phi_pulsar)

        B_pulsar = -np.sin(phi_pulsar)
        
        sin2psi = np.sin(2. * self._psi)
        cos2psi = np.cos(2. * self._psi)
        Rt = (cosiota * (f_plus * sin2psi - f_cross * cos2psi) * (
             A_pulsar - A_earth) - 
             (1. + cosiota**2) / 2. * (f_plus * cos2psi + f_cross * sin2psi) * (
             B_pulsar - B_earth)) * A_eff
        return Rt

    def Rt_normal(self):

        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        delta_t = self._pulsars.delta_t()

        t_earth = self._t - self._T
        t_pulsar = t_earth + delta_t
        t_earth_middle = []
        t_pulsar_middle = []
        for i in range(self._n - 1):
            t_earth_middle.append((t_earth[i] + t_earth[i+1]) / 2.)
            t_pulsar_middle.append((t_pulsar[i] + t_pulsar[i+1]) / 2.)
        t_earth_middle = np.array(t_earth_middle)
        t_pulsar_middle = np.array(t_pulsar_middle)

        phi_earth = self._events.t_to_phi(t_earth, self._phi_c)
        phi_pulsar = self._events.t_to_phi(t_pulsar, self._phi_c)
        h0_earth = self._events.t_to_h0(t_earth)
        h0_pulsar = self._events.t_to_h0(t_pulsar)
        phi_earth_middle = self._events.t_to_phi(t_earth_middle, self._phi_c)
        phi_pulsar_middle = self._events.t_to_phi(t_pulsar_middle, self._phi_c)
        h0_earth_middle = self._events.t_to_h0(t_earth_middle)
        h0_pulsar_middle = self._events.t_to_h0(t_pulsar_middle)

        A_earth = np.zeros_like(t_earth)
        A_pulsar = np.zeros_like(t_earth)
        B_earth = np.zeros_like(t_earth)
        B_pulsar = np.zeros_like(t_earth)

        dt = t_earth[1] - t_earth[0]

        omega_earth_0 = self._events.t_to_omega(t_earth[0])
        omega_pulsar_0 = self._events.t_to_omega(t_pulsar[0])
        A_earth[0] = ((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_earth_0**(-1./3.)
                * np.cos(phi_earth[0])).cgs.value

        B_earth[0] = -((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_earth_0**(-1./3.)
                * np.sin(phi_earth[0])).cgs.value

        A_pulsar[0] = ((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_pulsar_0**(-1./3.)
                 * np.cos(phi_pulsar[0])).cgs.value

        B_pulsar[0] = -((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_pulsar_0**(-1./3.)
                 * np.sin(phi_pulsar[0])).cgs.value
        

        for i in range(self._n - 1):

            A_earth[i+1] = A_earth[i] - 0.5 / 6. * (h0_earth[i] * np.sin(phi_earth[i])
                         + h0_earth[i+1] * np.sin(phi_earth[i+1]) 
                         + 4. * h0_earth_middle[i] * np.sin(phi_earth_middle[i])) * dt

            B_earth[i+1] = B_earth[i] - 0.5 / 6. * (h0_earth[i] * np.cos(phi_earth[i])
                         + h0_earth[i+1] * np.cos(phi_earth[i+1])
                         + 4. * h0_earth_middle[i] * np.cos(phi_earth_middle[i])) * dt

            A_pulsar[i+1] = A_pulsar[i] - 0.5 / 6. * (h0_pulsar[i] * np.sin(phi_pulsar[i])
                          + h0_pulsar[i+1] * np.sin(phi_pulsar[i+1])
                          + 4. * h0_pulsar_middle[i] * np.sin(phi_pulsar_middle[i])) * dt

            B_pulsar[i+1] = B_pulsar[i] - 0.5 / 6. * (h0_pulsar[i] * np.cos(phi_pulsar[i])
                          + h0_pulsar[i+1] * np.cos(phi_pulsar[i+1])
                          + 4. * h0_pulsar_middle[i] * np.cos(phi_pulsar_middle[i])) * dt

#            A_earth[i+1] = A_earth[i] - 0.5 * 0.5 * (h0_earth[i] * np.sin(phi_earth[i])
#                         + h0_earth[i+1] * np.sin(phi_earth[i+1])) * dt 

#            B_earth[i+1] = B_earth[i] - 0.5 * 0.5 * (h0_earth[i] * np.cos(phi_earth[i])
#                         + h0_earth[i+1] * np.cos(phi_earth[i+1])) * dt

#            A_pulsar[i+1] = A_pulsar[i] - 0.5 * 0.5 * (h0_pulsar[i] * np.sin(phi_pulsar[i])
#                          + h0_pulsar[i+1] * np.sin(phi_pulsar[i+1])) * dt

#            B_pulsar[i+1] = B_pulsar[i] - 0.5 * 0.5 * (h0_pulsar[i] * np.cos(phi_pulsar[i])
#                          + h0_pulsar[i+1] * np.cos(phi_pulsar[i+1])) * dt 



        Rt = (cosiota * (f_plus * np.sin(2. * self._psi) - f_cross * np.cos(2. * self._psi)) * (
             A_pulsar - A_earth) -
             (1. + cosiota**2) / 2. * (f_plus * np.cos(2. * self._psi) + f_cross * np.sin(2. * self._psi)) * (
             B_pulsar - B_earth))

        Rt_max = np.max(Rt)
        Rt_min = np.min(Rt)       
 
        return Rt

    def Rt_normal_ode(self):

        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        delta_t = self._pulsars.delta_t()

        t_earth = self._t - self._T
        t_pulsar = t_earth + delta_t
        t_earth_middle = []
        t_pulsar_middle = []

        phi_earth = self._events.t_to_phi(t_earth, self._phi_c)
        phi_pulsar = self._events.t_to_phi(t_pulsar, self._phi_c)
        h0_earth = self._events.t_to_h0(t_earth)
        h0_pulsar = self._events.t_to_h0(t_pulsar)    

        A_earth = np.zeros_like(t_earth)
        A_pulsar = np.zeros_like(t_earth)
        B_earth = np.zeros_like(t_earth)
        B_pulsar = np.zeros_like(t_earth)

        dt = t_earth[1] - t_earth[0]

        omega_earth_0 = self._events.t_to_omega(t_earth[0])
        omega_pulsar_0 = self._events.t_to_omega(t_pulsar[0])
        A_earth_0 = ((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_earth_0**(-1./3.)
                * np.cos(phi_earth[0])).cgs.value

        B_earth_0 = -((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_earth_0**(-1./3.)
                * np.sin(phi_earth[0])).cgs.value                

        A_pulsar_0 = ((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_pulsar_0**(-1./3.)
                 * np.cos(phi_pulsar[0])).cgs.value

        B_pulsar_0 = -((2. * c.G * self._Mc)**(5./3.) / self._dL / c.c**4 * omega_pulsar_0**(-1./3.)
                 * np.sin(phi_pulsar[0])).cgs.value

        def dif_At(t, y):
            h0 = self._events.t_to_h0(t)
            phi = self._events.t_to_phi(t, self._phi_c)
            return -0.5 * h0 * np.sin(phi)

        def dif_Bt(t, y):
            h0 = self._events.t_to_h0(t)
            phi = self._events.t_to_phi(t, self._phi_c)
            return -0.5 * h0 * np.cos(phi)

        method = 'DOP853'
        rtol = 1.e-15
        atol = 1.e-15
#        A_pulsar = solve_ivp(dif_At, (t_pulsar[0], t_pulsar[-1]), 
#                             [A_pulsar_0], rtol=rtol, atol=atol, method=method, t_eval=t_pulsar).y[0]

#        A_earth = solve_ivp(dif_At, (t_earth[0], t_earth[-1]), 
#                             [A_earth_0], rtol=rtol, atol=atol, method=method, t_eval=t_earth).y[0]

#        B_pulsar = solve_ivp(dif_Bt, (t_pulsar[0], t_pulsar[-1]), 
#                             [B_pulsar_0], rtol=rtol, atol=atol, method=method, t_eval=t_pulsar).y[0]

#        B_earth = solve_ivp(dif_Bt, (t_earth[0], t_earth[-1]), 
#                             [B_earth_0], rtol=rtol, atol=atol, method=method, t_eval=t_earth).y[0]
        solver_A = ode(dif_At).set_integrator('vode', rtol=rtol, atol=atol, method='bdf', order=15)

        solver_B = ode(dif_Bt).set_integrator('vode', rtol=rtol, atol=atol, method='bdf', order=15)

        A_earth = np.zeros_like(t_earth)
        A_pulsar = np.zeros_like(t_earth)
        B_earth = np.zeros_like(t_earth)
        B_pulsar = np.zeros_like(t_earth)

        A_earth[0] = A_earth_0
        A_pulsar[0] = A_pulsar_0
        B_earth[0] = B_earth_0
        B_pulsar[0] = B_pulsar_0

        solver_A.set_initial_value(A_earth_0, t_earth[0])
        solver_B.set_initial_value(B_earth_0, t_earth[0])
        for i in range(self._n - 1):
            A_earth[i+1] = solver_A.integrate(t_earth[i+1])[0]

        for i in range(self._n - 1):
            B_earth[i+1] = solver_B.integrate(t_earth[i+1])[0]

        solver_A.set_initial_value(A_pulsar_0, t_pulsar[0])
        solver_B.set_initial_value(B_pulsar_0, t_pulsar[0])
        for i in range(self._n - 1):
            A_pulsar[i+1] = solver_A.integrate(t_pulsar[i+1])[0]

        for i in range(self._n - 1):
            B_pulsar[i+1] = solver_B.integrate(t_pulsar[i+1])[0]



        Rt = (cosiota * (f_plus * np.sin(2. * self._psi) - f_cross * np.cos(2. * self._psi)) * (
             A_pulsar - A_earth) -
             (1. + cosiota**2) / 2. * (f_plus * np.cos(2. * self._psi) + f_cross * np.sin(2. * self._psi)) * (
             B_pulsar - B_earth))

        return Rt
                 


    def Phi(self):

        t_earth = self._t - self._T
        phi_earth = self._events.t_to_phi(t_earth, self._phi_c)
        delta_t = self._pulsars.delta_t()
        t_pulsar = t_earth + delta_t
        phi_pulsar = self._events.t_to_phi(t_pulsar, self._phi_c)
        return phi_earth, phi_pulsar

    def Phi_D(self):

        delta_t = self._pulsars.delta_t()
        t = self._t - self._T[int(self._n / 2)] + delta_t
        phi_pulsar = self._events.t_to_phi(t, self._phi_c)
        return phi_pulsar
        
    def D_Phi(self, phi):
    
        t = self._events.phi_to_t(phi, self._phi_c)
        delta_t = t - self._t + self._T[int(self._n / 2)]
        D = (delta_t / (1. - self.costheta()) * c.c / c.kpc).cgs.value
        return D

    def omega_bi(self):

        return self._omega_earth, self._omega_pulsar
    
    
class time_residual_num(object):
    
    def __init__(self, **kw):

        self._dL = (np.exp(kw['logdL']) * c.kpc * 1.0e3).cgs.value
#        self._Mc = (Mc * c.M_sun).cgs.value
        self._alpha = kw['alpha']
        self._delta = kw['delta']
        self._psi = kw['psi']
        self._iota = kw['iota']
        self._alpha_p = kw['pulsar']['ra'] #* np.pi / 180.
        self._delta_p = kw['pulsar']['dec'] #* np.pi / 180.
        self._D = (kw['pulsar']['D_corr'] * c.kpc).cgs.value
        self._G = c.G.cgs.value
        self._c = c.c.cgs.value
        self._noise = kw['pulsar']['noise']
        if 'num' in kw:
            self._n = kw['num']
        else:
            self._n = 1000

        if self._n == 1:
            self._T = np.array([(kw['T'] * u.a).cgs.value])
        else:
            self._T = np.linspace((0. * u.a).cgs.value, (kw['T'] * u.a).cgs.value, num=self._n)
            
        self._Mc = (kw['Mc'] * 1.e6 * c.M_sun).cgs.value
        self._omega = kw['omega'] * 1.0e-9
        self._phi_0 = kw['phi_0']
        self._pulsars = psr(D=self._D, alpha=self._alpha, delta=self._delta, 
                           alpha_p = self._alpha_p, delta_p=self._delta_p)
        self.delta_t = self._pulsars.delta_t()
        if 'phi_p' in kw:
            self._phi_p_0 = kw['phi_p']
        else:
            delta_phi_0 = self._omega * self.delta_t - (3. / 5. * 2.**(4. / 3.)
                        * (self._G * self._Mc / self._c**3) **(5. / 3.) * self._omega**(11. / 3.) * self.delta_t**2.)
            delta_phi_0 = delta_phi_0 - np.floor(delta_phi_0 / (2. * np.pi)) * 2. * np.pi
            self._phi_p_0 = self._phi_0 - delta_phi_0
            
    def Rt_bi(self):
        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        
        Delta_omega = 3. / 5. * 2.**(7. / 3.) * (self._G * self._Mc / self._c**3) **(5. / 3.) * self._omega**(11. / 3.) * self.delta_t


        omega_earth = self._omega
        omega_pulsar = self._omega - Delta_omega
        phi_earth = omega_earth * self._T + self._phi_0
        phi_pulsar = omega_pulsar * self._T + self._phi_p_0 
        A_eff_earth = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * omega_earth**(-1./3.)
    
        A_eff_pulsar = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * omega_pulsar**(-1./3.)
        
        A_earth = np.cos(phi_earth) * A_eff_earth

        B_earth = -np.sin(phi_earth) * A_eff_earth

        A_pulsar = np.cos(phi_pulsar)* A_eff_pulsar

        B_pulsar = -np.sin(phi_pulsar)* A_eff_pulsar
        
        sin2psi = np.sin(2. * self._psi)
        cos2psi = np.cos(2. * self._psi)
        Rt = (cosiota * (f_plus * sin2psi - f_cross * cos2psi) * (
             A_pulsar - A_earth) - 
             (1. + cosiota**2) / 2. * (f_plus * cos2psi + f_cross * sin2psi) * (
             B_pulsar - B_earth))
        return Rt
    
        
        return Rt
    
    def T_series(self):

        return self._T

    def phi_p(self):
        
        return self._phi_p
    
    def snr(self):
        
        noise = ((self._noise / 2. / 2.0 / u.wk)**0.5).cgs.value
        rt = self.Rt_bi()
        snr = np.sum((rt / noise)**2.)**0.5
        
        return snr
    
    
class time_residual_deltaphi(object):
    
    def __init__(self, **kw):
    
        self._dL = (np.exp(kw['logdL']) * c.kpc * 1.0e3).cgs.value
#        self._Mc = (Mc * c.M_sun).cgs.value
        self._alpha = kw['alpha']
        self._delta = kw['delta']
        self._psi = kw['psi']
        self._iota = kw['iota']
        self._alpha_p = kw['pulsar']['ra'] #* np.pi / 180.
        self._delta_p = kw['pulsar']['dec'] #* np.pi / 180.
        self._D = (kw['pulsar']['D_corr'] * c.kpc).cgs.value
        self._noise = kw['pulsar']['noise']
        self._c = c.c.cgs.value
        self._G = c.G.cgs.value
        self._a = (1. * u.a).cgs.value
        if 'num' in kw:
            self._n = kw['num']
        else:
            self._n = 1000

        if self._n == 1:
            self._T = np.array([(kw['T'] * u.a).cgs.value])
        else:
            self._T = np.linspace(0., kw['T'] * self._a, num=self._n)
            
        self._pulsars = psr(D=self._D, alpha=self._alpha, delta=self._delta, 
                           alpha_p = self._alpha_p, delta_p=self._delta_p)
        self._delta_phi_0 = kw['delta_phi_0']
        self._Mc = (kw['Mc'] * 1.e6 * c.M_sun).cgs.value
        self._omega_earth = kw['omega_earth'] * 1.0e-9
        if 'omega_pulsar' in kw:
            self._omega_pulsar = kw['omega_pulsar'] * 1.0e-9
        elif 'Domega' in kw:
            delta_omega = kw['Domega'] * 1.0e-9
            self._omega_pulsar = self._omega_earth - delta_omega
        else:
            self.delta_t = self._pulsars.delta_t()
            Delta_omega = 3. / 5. * 2.**(7. / 3.) * (self._G * self._Mc / self._c**3) **(5. / 3.) * self._omega_earth**(11. / 3.) * self.delta_t
            self._omega_pulsar = self._omega_earth - Delta_omega
        self._phi_0 = kw['phi_0']

        
    def Rt_bi(self):
        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        cosiota = np.cos(self._iota)
        sin2psi = np.sin(2. * self._psi)
        cos2psi = np.cos(2. * self._psi)
        phi_earth = self._omega_earth * self._T + self._phi_0
        phi_pulsar = self._omega_pulsar * self._T + self._phi_0 - self._delta_phi_0
        
        A_eff_earth = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * self._omega_earth**(-1./3.)
        
        A_eff_pulsar = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * self._omega_pulsar**(-1./3.)
        
        A_earth = np.cos(phi_earth) * A_eff_earth

        B_earth = -np.sin(phi_earth) * A_eff_earth

        A_pulsar = np.cos(phi_pulsar) * A_eff_pulsar

        B_pulsar = -np.sin(phi_pulsar) * A_eff_pulsar

        
        
        Rt = (cosiota * (f_plus * sin2psi - f_cross * cos2psi) * (
             A_pulsar - A_earth) - 
             (1. + cosiota**2) / 2. * (f_plus * cos2psi + f_cross * sin2psi) * (
             B_pulsar - B_earth))
        return Rt

    def Rt_mono(self):
        f_plus = self._pulsars.f_plus()
        f_cross = self._pulsars.f_cross()
        
        cosiota = np.cos(self._iota)
        sin2psi = np.sin(2. * self._psi)
        cos2psi = np.cos(2. * self._psi)
        self._omega_pulsar = self._omega_earth
        phi_earth = self._omega_earth * self._T + self._phi_0
        phi_pulsar = self._omega_pulsar * self._T + self._phi_0 - self._delta_phi_0 

        A_eff_earth = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * self._omega_earth**(-1./3.)
        
        A_eff_pulsar = (2. * self._G * self._Mc)**(5./3.) / self._dL / self._c**4 * self._omega_pulsar**(-1./3.)
        
        A_earth = np.cos(phi_earth) * A_eff_earth

        B_earth = -np.sin(phi_earth) * A_eff_earth

        A_pulsar = np.cos(phi_pulsar) * A_eff_pulsar

        B_pulsar = -np.sin(phi_pulsar) * A_eff_pulsar

        
        
        Rt = (cosiota * (f_plus * sin2psi - f_cross * cos2psi) * (
             A_pulsar - A_earth) - 
             (1. + cosiota**2) / 2. * (f_plus * cos2psi + f_cross * sin2psi) * (
             B_pulsar - B_earth))
        return Rt
    
    def snr(self):
        
        noise = ((self._noise / 2. / 2.0 / u.wk)**0.5).cgs.value
        rt = self.Rt_bi()
        snr = np.sum((rt / noise)**2.)**0.5
        
        return snr