import warnings
import numpy as np

from astropy.io import fits
from astropy.stats.funcs import median_absolute_deviation
import scipy.ndimage

class TargetPixelFile(object):
    """Enables extraction of raw lightcurves and centroid positions.

    Attributes
    ----------
    path : str
        Path to fits file.

    max_quality : int
        Maximum quality for the cadences.

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """

    def __init__(self, path, max_quality=1):
        self.path = path
        self.max_quality = max_quality
        self.hdu = fits.open(path)

    def good_quality_mask(self, max_quality=None):
        """Returns a boolean mask flagging the good-quality cadences.

        Parameters
        ----------
        max_quality : int or None
            Maximum tolerated quality. See ref. [1], table 2-3.
        """

        if max_quality is None:
            max_quality = self.max_quality
        return self.hdu[1].data['QUALITY'] < max_quality

    @property
    def n_cadences(self):
        """Returns the number of good-quality cadences."""
        return self.good_quality_mask().sum()

    @property
    def time(self):
        """Returns the time for all good-quality cadences."""
        return self.hdu[1].data['TIME'][self.good_quality_mask()]

    @property
    def flux(self):
        """Returns the flux for all good-quality cadences."""
        return self.hdu[1].data['FLUX'][self.good_quality_mask(), :, :]

    def median_flux(self):
        """Returns the median flux across all cadences."""
        return np.nanmedian(self.flux, axis=0)

    def mean_flux(self):
        """Returns the mean flux across all cadences."""
        return np.nanmean(self.flux, axis=0)

    def aperture_mask(self, snr_threshold=5, margin=4):
        """Returns an aperture photometry mask.

        Parameters
        ----------
        snr_threshold : float
            Background detection threshold.
        """

        # Find the pixels that are above the threshold in the median flux image
        median = self.median_flux()
        mad = median_absolute_deviation(median[np.isfinite(median)])
        # 1.4826 turns MAD into STDEV for a Gaussian
        mad_cut = 1.4826 * mad * snr_threshold

        region = np.where(median > mad_cut, 1, 0)
        # Label all contiguous regions above the threshold
        labels = scipy.ndimage.label(region)[0]
        # Central pixel coordinate
        centralpix = [1 + median.shape[0] // 2, 1 + median.shape[1] // 2]

        # find brightest pix within margin of central pix
        central_img = median[centralpix[0] - margin: centralpix[0] + margin,
                             centralpix[1] - margin: centralpix[1] + margin]
        # unravel_index converts indices into a tuple of coordinate arrays
        brightestpix = np.unravel_index(central_img.argmax(), central_img.shape)
        bpixy, bpixx = brightestpix

        # Which label corresponds to the brightest pixel?
        regnum = labels[centralpix[0] - margin + bpixy, centralpix[1] - margin + bpixx]
        if regnum == 0:
            warnings.warn('No star were found in light curve {}, '
                          'light curve will be junk!'.format(self.path),
                          UserWarning)

        aperture_mask = labels == regnum

        return aperture_mask

    def centroids(self, aperture_mask=None):
        """Returns the centroids for every cadence under a given aperture
        mask.

        Attributes
        ----------
        aperture_mask: boolean ndarray or None
            Aperture mask under which centroids will be computed. If ``None``,
            then an aperture is computed using ``aperture_mask``.

        Returns
        -------
        xc, yc: ndarrays
            centroid positions for every cadence
        """

        if aperture_mask is None:
            aperture_mask = self.aperture_mask()

        xc = np.zeros(self.n_cadences)
        yc = np.zeros(self.n_cadences)

        y, x = np.mgrid[:img.shape[0], :img.shape[1]]

        for i in range(self.n_cadences):
            xc[i] = (self.flux[i] * x).sum() / self.flux[i].sum()
            yc[i] = (self.flux[i] * y).sum() / self.flux[i].sum()

        return xc, yc

    def to_lightcurve(self, aperture_mask=None):
        """Returns a raw light curve of the TPF.

        Attributes
        ----------
        aperture_mask: boolean ndarray or None
            Aperture under which the flux will be summed up. If ``None``,
            then an aperture is computed using ``aperture_mask``.

        Returns
        -------
        lightcurve : ndarray
            Array containing the summed flux within the aperture for each
            cadence.
        """

        if aperture_mask is None:
            aperture_mask = self.aperture_mask()

        lightcurve = np.zeros(self.n_cadences)
        for i in range(self.n_cadences):
            lightcurve[i] = self.flux[i][aperture_mask].sum()

        return lightcurve


"""
Get TPF and LC objects from the archive
Given a list of EPIC or KIC return TPFs

Basics
------

LightCurve.from_epic(1231235, method='Vanderbug')
LightCurve.from_kic()

LightCurve.from_K2.from_epic

LightCurve.to_astropy_table
LightCurve.to_pandas

Extensions
----------
class MASTArchive
Channel number, ...

COROT

Priority
--------
Get a TPF from MAST, create a light curve, use whatever method to detrend (sc) it.




"""
