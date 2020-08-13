"""
coronagraphy.py
Kristina Davis
Nov 2019

This script adds functionality related to coronagraphy of the telescope system. I have decided to add a class
of occulter to the module to take care of different types of coronagraphs


"""
import os
import numpy as np
import proper
from astropy.io.fits import getdata, writeto
# import cv2
import warnings

from medis.params import tp, sp, ap, iop
import medis.optics as opx
from medis.utils import dprint


class Occulter():
    """
    produces an occulter of various modes

    valid modes:
    'Gaussian'
    'Solid'
    '8th_Order'

    most of this class is a modification of the chronograph routine found on the proper manual pg 85
    """
    def __init__(self, _mode):
        if _mode is 'Gaussian' or 'Solid' or '8th_Order':
            self.mode = _mode
        else:
            raise ValueError('please choose a valid mode: Gaussian, Solid, 8th_Order')

    def set_size(self, wf, size_in_lambda_d=0, size_in_m=0):
        """
        sets size of occulter in m, depending on if input is in units of lambda/D or in m

        :param wf: 2D wavefront
        :param size_in_lambda_d: desired occulter size in units of lambda/D
        :param size_in_m: desired occulter size in m
        :return: occulter size in m
        """
        lamda = proper.prop_get_wavelength(wf)
        dx_m = proper.prop_get_sampling(wf)
        dx_rad = proper.prop_get_sampling_radians(wf)

        if size_in_lambda_d is not 0:
            occrad_rad = size_in_lambda_d * lamda / tp.entrance_d  # occulter radius in radians
            self.size = occrad_rad * dx_m / dx_rad  # occulter radius in meters
        elif size_in_m is not 0:
            self.size = size_in_m
        else:
            raise ValueError('must set occulter size in either m or lambda/D units')

        # rescaling for focused system
        # if sp.focused_sys:
        #     self.size = wf.lamda / tp.entrance_d * ap.wvl_range[0] / wf.lamda

    def apply_occulter(self, wf):
        """
        applies the occulter by type spcified when class object was initiated

        :param wf: 2D wavefront
        :return:
        """
        # Code here pulled directly from Proper Manual pg 86
        if self.mode == "Gaussian":
            r = proper.prop_radius(wf)
            h = np.sqrt(-0.5 * self.size**2 / np.log(1 - np.sqrt(0.5)))
            gauss_spot = 1 - np.exp(-0.5 * (r/h)**2)
            # gauss_spot = shift(gauss_spot, shift=tp.occult_loc, mode='wrap')  # ???
            proper.prop_multiply(wf, gauss_spot)
        elif self.mode == "Solid":
            proper.prop_circular_obscuration(wf, self.size)
        elif self.mode == "8th_Order":
            proper.prop_8th_order_mask(wf, self.size, CIRCULAR=True)
        elif self.mode == 'Vortex':
            vortex = Vortex().occulter(wf)

    def apply_lyot(self, wf):
        """
        applies the appropriately sized Lyot stop depending on the coronagraph type

        :param wf: 2D wavefront
        :return:
        """
        if not hasattr(tp, 'lyot_size'):
            raise ValueError("must set tp.lyot_size in units fraction of the beam radius at the current surface")
        if self.mode is 'Gaussian':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)
        elif self.mode is 'Solid':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)
        elif self.mode is '8th_Order':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)
        elif self.mode == "Vortex":
            vortex = Vortex().lyotstop(wf,True)

def coronagraph(wf, occulter_mode=None):
    """
    propagates a wavefront through a coronagraph system

    We implicitly assume here that the system begins at the location of the occulting mask. The optical system leading
    up to this point must be defined in the telescope perscription. This coronagraph routine also ends at the image
    plane after the Lyot stop, if used. If no Lyot stop is used, the reimaging optics just pass the pupil image
    unaffected.

    this function is modified from the Proper Manual: Stellar Coronagraph example found on pg 85

    :param wf: a single wavefront of complex data shape=(sp.grid_size, sp.grid_size)
    :param occulter_type string defining the occulter type. Accepted types are "Gaussian", "Solid", and "8th_Order"
    :return: noting is returned but wfo has been modified
    """
    if occulter_mode is None:
        pass
    else:
        # Initialize Occulter based on Mode
        occ = Occulter(occulter_mode)
        if not hasattr(tp, 'cg_size'):
            raise ValueError("must set tp.cg_size and tp.cg_size_units to use coronagraph() in coronography.py")
        elif tp.cg_size_units == "m":
            occ.set_size(wf, size_in_m=tp.cg_size)
        else:
            occ.set_size(wf, size_in_lambda_d=tp.cg_size)

        # Apply Occulter
        occ.apply_occulter(wf)  # size saved to self in class, so don't need to pass it

        proper.prop_propagate(wf, tp.fl_cg_lens)  # propagate to coronagraph pupil reimaging lens from occulter
        opx.prop_pass_lens(wf, tp.fl_cg_lens, tp.fl_cg_lens)  # go to the middle of the 2-lens system

        # Apply Lyot
        occ.apply_lyot(wf)

        proper.prop_propagate(wf, tp.fl_cg_lens)  # propagate to reimaging lens from lyot stop
        opx.prop_pass_lens(wf, tp.fl_cg_lens, tp.fl_cg_lens)  # go to the next image plane


def apodization(wf):
    raise NotImplementedError


class Vortex():
    # https://github.com/vortex-exoplanet/HEEPS/tree/master/heeps
    def __int__(self):
        warnings.warn("Using vector vortex code quickly adapted from METIS. Still largely unverified")

    def occulter(self, wf):

        n = int(proper.prop_get_gridsize(wf))
        ofst = 0  # no offset
        ramp_sign = 1  # sign of charge is positive
        ramp_oversamp = 11.  # vortex is oversampled for a better discretization

        # f_lens = tp.f_lens #conf['F_LENS']
        # diam = tp.diam#conf['DIAM']
        charge = 2#conf['CHARGE']
        pixelsize = 5#conf['PIXEL_SCALE']
        Debug_print = False#conf['DEBUG_PRINT']

        coron_temp = os.path.join(iop.testdir, 'coron_maps/')
        if not os.path.exists(coron_temp):
            os.mkdir(coron_temp)

        if charge != 0:
            wavelength = proper.prop_get_wavelength(wf)
            gridsize = proper.prop_get_gridsize(wf)
            beam_ratio = pixelsize * 4.85e-9 / (wavelength / tp.entrance_d)
            # dprint((wavelength,gridsize,beam_ratio))
            calib = str(charge) + str('_') + str(int(beam_ratio * 100)) + str('_') + str(gridsize)
            my_file = str(coron_temp + 'zz_perf_' + calib + '_r.fits')

            if (os.path.isfile(my_file) == True):
                if (Debug_print == True):
                    print ("Charge ", charge)
                vvc = self.readfield(coron_temp, 'zz_vvc_' + calib)  # read the theoretical vortex field
                vvc = proper.prop_shift_center(vvc)
                scale_psf = wf._wfarr[0, 0]
                psf_num = self.readfield(coron_temp, 'zz_psf_' + calib)  # read the pre-vortex field
                psf0 = psf_num[0, 0]
                psf_num = psf_num / psf0 * scale_psf
                perf_num = self.readfield(coron_temp, 'zz_perf_' + calib)  # read the perfect-result vortex field
                perf_num = perf_num / psf0 * scale_psf
                wf._wfarr = (
                             wf._wfarr - psf_num) * vvc + perf_num  # the wavefront takes into account the real pupil with the perfect-result vortex field

            else:  # CAL==1: # create the vortex for a perfectly circular pupil
                if (Debug_print == True):
                    print ("Charge ", charge)

                f_lens = 200.0 * tp.entrance_d
                wf1 = proper.prop_begin(tp.entrance_d, wavelength, gridsize, beam_ratio)
                proper.prop_circular_aperture(wf1, tp.entrance_d / 2)
                proper.prop_define_entrance(wf1)
                proper.prop_propagate(wf1, f_lens, 'inizio')  # propagate wavefront
                proper.prop_lens(wf1, f_lens, 'focusing lens vortex')  # propagate through a lens
                proper.prop_propagate(wf1, f_lens, 'VC')  # propagate wavefront

                self.writefield(coron_temp, 'zz_psf_' + calib, wf1.wfarr)  # write the pre-vortex field
                nramp = int(n * ramp_oversamp)  # oversamp
                # create the vortex by creating a matrix (theta) representing the ramp (created by atan 2 gradually varying matrix, x and y)
                y1 = np.ones((nramp,), dtype=np.int)
                y2 = np.arange(0, nramp, 1.) - (nramp / 2) - int(ramp_oversamp) / 2
                y = np.outer(y2, y1)
                x = np.transpose(y)
                theta = np.arctan2(y, x)
                x = 0
                y = 0
                vvc_tmp = np.exp(1j * (ofst + ramp_sign * charge * theta))
                theta = 0
                vvc_real_resampled = cv2.resize(vvc_tmp.real, (0, 0), fx=1 / ramp_oversamp, fy=1 / ramp_oversamp,
                                                interpolation=cv2.INTER_LINEAR)  # scale the pupil to the pupil size of the simualtions
                vvc_imag_resampled = cv2.resize(vvc_tmp.imag, (0, 0), fx=1 / ramp_oversamp, fy=1 / ramp_oversamp,
                                                interpolation=cv2.INTER_LINEAR)  # scale the pupil to the pupil size of the simualtions
                vvc = np.array(vvc_real_resampled, dtype=complex)
                vvc.imag = vvc_imag_resampled
                vvcphase = np.arctan2(vvc.imag, vvc.real)  # create the vortex phase
                vvc_complex = np.array(np.zeros((n, n)), dtype=complex)
                vvc_complex.imag = vvcphase
                vvc = np.exp(vvc_complex)
                vvc_tmp = 0.
                self.writefield(coron_temp, 'zz_vvc_' + calib, vvc)  # write the theoretical vortex field

                proper.prop_multiply(wf1, vvc)
                proper.prop_propagate(wf1, f_lens, 'OAP2')
                proper.prop_lens(wf1, f_lens)
                proper.prop_propagate(wf1, f_lens, 'forward to Lyot Stop')
                proper.prop_circular_obscuration(wf1, 1., NORM=True)  # null the amplitude iside the Lyot Stop
                proper.prop_propagate(wf1, -f_lens)  # back-propagation
                proper.prop_lens(wf1, -f_lens)
                proper.prop_propagate(wf1, -f_lens)
                self.writefield(coron_temp, 'zz_perf_' + calib, wf1.wfarr)  # write the perfect-result vortex field

                vvc = self.readfield(coron_temp, 'zz_vvc_' + calib)
                vvc = proper.prop_shift_center(vvc)
                scale_psf = wf._wfarr[0, 0]
                psf_num = self.readfield(coron_temp, 'zz_psf_' + calib)  # read the pre-vortex field
                psf0 = psf_num[0, 0]
                psf_num = psf_num / psf0 * scale_psf
                perf_num = self.readfield(coron_temp, 'zz_perf_' + calib)  # read the perfect-result vortex field
                perf_num = perf_num / psf0 * scale_psf
                wf._wfarr = (
                             wf._wfarr - psf_num) * vvc + perf_num  # the wavefront takes into account the real pupil with the perfect-result vortex field

        return wf

    def lyotstop(self, wf,  RAVC=None, APP=None, get_pupil='no', dnpup=50):
        """Add a Lyot stop, or an APP."""

        # load parameters
        npupil = 1#conf['NPUPIL']
        pad = int((210 - npupil) / 2)

        # get LS misalignments
        LS_misalignment = (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * npupil).astype(int)
        dx_amp, dy_amp, dz_amp = LS_misalignment[0:3]
        dx_phase, dy_phase, dz_phase = LS_misalignment[3:6]

        # case 1: Lyot stop (no APP)
        if APP is not True:

            # Lyot stop parameters: R_out, dR_in, spi_width
            # outer radius (absolute %), inner radius (relative %), spider width (m)
            (R_out, dR_in, spi_width) = [0.98, 0.03, 0]

            # Lyot stop inner radius at least as large as obstruction radius
            R_in = 0.15

            # case of a ring apodizer
            if RAVC is True:
                # define the apodizer transmission and apodizer radius [Mawet2013]
                # apodizer radius at least as large as obstruction radius
                T_ravc = 1 - (R_in ** 2 + R_in * np.sqrt(R_in ** 2 + 8)) / 4
                R_in /= np.sqrt(1 - T_ravc)

            # oversize Lyot stop inner radius
            R_in += dR_in

            # create Lyot stop
            proper.prop_circular_aperture(wf, R_out, dx_amp, dy_amp, NORM=True)
            if R_in > 0:
                proper.prop_circular_obscuration(wf, R_in, dx_amp, dy_amp, NORM=True)
            if spi_width > 0:
                for angle in [10]:
                    proper.prop_rectangular_obscuration(wf, 0.05 * 8, 8 * 1.3, ROTATION=20)
                    proper.prop_rectangular_obscuration(wf, 8 * 1.3, 0.05 * 8, ROTATION=20)
                    # proper.prop_rectangular_obscuration(wf, spi_width, 2 * 8, \
                    #                                     dx_amp, dy_amp, ROTATION=angle)

        # case 2: APP (no Lyot stop)
        else:
            # get amplitude and phase files
            APP_amp_file = os.path.join(conf['INPUT_DIR'], conf['APP_AMP_FILE'])
            APP_phase_file = os.path.join(conf['INPUT_DIR'], conf['APP_PHASE_FILE'])
            # get amplitude and phase data
            APP_amp = getdata(APP_amp_file) if os.path.isfile(APP_amp_file) \
                else np.ones((npupil, npupil))
            APP_phase = getdata(APP_phase_file) if os.path.isfile(APP_phase_file) \
                else np.zeros((npupil, npupil))
            # resize to npupil
            APP_amp = resize(APP_amp, (npupil, npupil), preserve_range=True, mode='reflect')
            APP_phase = resize(APP_phase, (npupil, npupil), preserve_range=True, mode='reflect')
            # pad with zeros to match PROPER gridsize
            APP_amp = np.pad(APP_amp, [(pad + 1 + dx_amp, pad - dx_amp), \
                                       (pad + 1 + dy_amp, pad - dy_amp)], mode='constant')
            APP_phase = np.pad(APP_phase, [(pad + 1 + dx_phase, pad - dx_phase), \
                                           (pad + 1 + dy_phase, pad - dy_phase)], mode='constant')
            # multiply the loaded APP
            proper.prop_multiply(wf, APP_amp * np.exp(1j * APP_phase))

        # get the pupil amplitude or phase for output
        if get_pupil.lower() in 'amplitude':
            return wf, proper.prop_get_amplitude(wf)[pad + 1 - dnpup:-pad + dnpup, pad + 1 - dnpup:-pad + dnpup]
        elif get_pupil.lower() in 'phase':
            return wf, proper.prop_get_phase(wf)[pad + 1 - dnpup:-pad + dnpup, pad + 1 - dnpup:-pad + dnpup]
        else:
            return wf

    def writefield(self, path, filename, field):
        writeto(path + filename + '_r.fits', field.real, header=None, overwrite=True)
        writeto(path + filename + '_i.fits', field.imag, header=None, overwrite=True)

    def readfield(self, path, filename):

        try:
            data_r, hdr = getdata(path + filename + '_r.fits', header=True)
            data_i = getdata(path + filename + '_i.fits')
        except:
            dprint('FileNotFoundError. Waiting...')
            import time
            time.sleep(10)
            data_r, hdr = getdata(path + filename + '_r.fits', header=True)
            data_i = getdata(path + filename + '_i.fits')

        field = np.array(data_r, dtype=complex)
        field.imag = data_i

        return (field)
