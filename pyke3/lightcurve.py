import numpy as np
from abc import ABC, abstractmethod

"""
Systematics correction != Detrending
"""

class LightCurve(object):
    """
    Implements a basic time-series class for a generic lightcurve.

    Parameters
    ----------
    time : numpy array-like
        Time-line.
    flux : numpy array-like
        Data flux for every time point.
    """

    def __init__(self, time, flux):
        self.time = time
        self.flux = flux

    def detrend(self, method='arclength', **kwargs):
        """
        """
        if method == 'arclength':
            return ArcLengthDetrender.detrend(self.time, self.flux, **kwargs)

class Detrender(ABC):
    """
    """
    @abstractmethod
    def detrend():
        pass

class SystematicsCorrector(ABC):
    """
    """
    @abstractmethod
    def correct(**kwargs):
        """
        kawrgs: not only light curve, but also physical parameters like centroind
        positions
        """
        pass

class FirstDifferenceDetrender(Detrender):
    """
    First difference detrending
    """
    def detrend(time, flux):
        return LightCurve(time, flux - np.append(0, flux[1:]))

class LinearDetrender(Detrender):
    """
    """
    @staticmethod
    def detrend(time, flux):
        return LightCurve()

class ArcLengthDetrender(Detrender):
    def detrend(time, flux):
        pass

class EMDDetrender(Detrender):
    """
    Empirical Mode Decomposition Detrender
    """
    def detrend(time, flux):
        pass

class PolynomialDetrender(Detrender):
    """
    """
    def detrend(time, flux):
        pass
