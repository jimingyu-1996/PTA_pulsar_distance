#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import astropy.constants as c

class psr(object):

    def __init__(self, *, D, alpha, delta, alpha_p, delta_p):

        self._D = D
        self._alpha = alpha
        self._delta = delta
        self._alpha_p = alpha_p
        self._delta_p = delta_p

        self.__costheta = np.cos(self._delta) * np.cos(self._delta_p) * np.cos(
                             self._alpha - self._alpha_p) + np.sin(self._delta) * np.sin(self._delta_p)
        self.__delta_t = (self._D * (1. - self.__costheta) / c.c).cgs.value

        self.__f_plus = 1. / (1. - self.__costheta) * (
                      - (1. + np.sin(self._delta)**2.) / 2. * np.cos(self._delta_p)**2. * np.cos(
                        2.0 * (self._alpha - self._alpha_p))
                      + 0.5 * np.sin(2. * self._delta_p) * np.sin(2. * self._delta) * np.cos(
                        self._alpha - self._alpha_p)
                      - np.cos(self._delta)**2. * (1.0 - 1.5 * np.cos(self._delta_p)**2.))

        self.__f_cross = 1. / (1. - self.__costheta) * (
                       - np.cos(self._delta_p)**2. * np.sin(self._delta) * np.sin(
                         2. * (self._alpha_p - self._alpha))
                       + np.sin(2.0 * self._delta_p) * np.cos(self._delta) 
                       * np.sin(self._alpha_p - self._alpha))

    def costheta(self):
        return self.__costheta

    def delta_t(self):
        return self.__delta_t

    def f_plus(self):
        return self.__f_plus

    def f_cross(self):
        return self.__f_cross
        
    def D(self):
        return self._D

        
