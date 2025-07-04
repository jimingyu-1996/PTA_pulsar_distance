#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import astropy.constants as c

class gw_event(object):

    def __init__(self, *, dL, Mc, iota, psi):

        self._dL = dL
        self._Mc = Mc
        self._iota = iota
        self._psi = psi
        self._c = c.c.cgs.value
        self._G = c.G.cgs.value


    def t_to_h0(self, t_to_merge):

        self._t = t_to_merge
        h0 = 2.**(4. / 3.) / self._dL * 5.**(1. / 4.) * (
             c.G * self._Mc)**(5. / 4.) / self._t**(1. / 4.) / c.c**(11. / 4.)
        return h0.cgs.value

    def omega_to_h0(self, omega):

        self._omega = omega
        h0 = 2.**(8. / 3.) / self._dL * (c.G * self._Mc)**(5. / 3.) * self._omega**(2. / 3.) / c.c**4
        return h0.cgs.value

    def t_to_omega(self, t_to_merge):

        self._t = t_to_merge
        omega = 5.**(3. / 8.) * self._c**(15. / 8.) / 4. / (self._G * self._Mc)**(5. / 8.) * self._t**(-3. / 8.)
        return omega

    def omega_to_t(self, omega):

        self._omega = omega
        t = 2.**(-16. / 3.) * self._omega**(-8. / 3.) * (self._G * self._Mc)**(-5. / 3.) * self._c**5 * 5.
        return t

    def t_to_phi(self, t_to_merge, phi_c):

        phi = 2. * self._c**(15./8.) * (t_to_merge / 5. / self._G / self._Mc)**(5./8.) + phi_c
        return phi

    def phi_to_t(self, phi, phi_c):

        t_to_merge = ((phi - phi_c) / (2. * self._c**(15./8.)))**(8./5.) * 5. * self._G * self._Mc
        return t_to_merge
        
    def ht(self, t_to_merge, phi_c = 0.0):
    
        h0 = self.t_to_h0(t_to_merge)
        omega = self.t_to_omega(t_to_merge)
        phi = self.t_to_phi(t_to_merge, phi_c)

        h_plus = h0 * (np.cos(self._iota) * np.sin(2. * self._psi) * np.sin(phi)
               - 0.5 * (1. + np.cos(self._iota)**2) * np.cos(2. * self._psi) * np.cos(phi))
               
        h_cross = -h0 * (np.cos(self._iota) * np.cos(2. * self._psi) * np.sin(phi)
                + 0.5 * (1. + np.cos(self._iota)**2) * np.sin(2. * self._psi) * np.cos(phi))
               
        return h_plus, h_cross
    
        

    def dL(self):

        return self._dL

    def Mc(self):
 
        return self._Mc
