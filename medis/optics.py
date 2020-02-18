"""
optics.py

contains functionality related to the optical components needed to build up a proper prescription. This is a generic
module containing functions that don't have a home elsewhere. It contains the class structure to make the wavefront
object used in most prescriptions.
"""
import numpy as np
import proper
import copy
from scipy.interpolate import interp1d
from inspect import getframeinfo, stack

from medis.params import ap, tp, sp
from medis.utils import dprint


############################
# Create Wavefront Array
############################
class Wavefronts():
    """
    An object containing all of the complex E fields for each sampled wavelength and astronomical object at this timestep

    :params

    :returns
    self.wf_collection: a 2D array, where each element is its own 2D proper wavefront object structured as
     self.wf_collection[iw,io]. Here, iw, io is each wavefront and object, respectively
        ...meaning its an array of arrays.
        thus, self.wf_collection[iw,io] is itself a 2D array of complex data. Its size is [sp.grid_size, sp.grid_size]
        we will call each instance of the collection a single wavefront wf
    self.save_E_fields: a matrix of E fields (proper.WaveFront.wfarr) at specified locations in the chain
    """
    def __init__(self):

        # Using Proper to propagate wavefront from primary through optical system, loop over wavelength
        self.wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)  # units set in params (should be m)

        # wf_collection is an array of arrays; the wf_collection is (number_wavelengths x number_astro_objects)
        # each 2D field in the wf_collection is the 2D array of complex E-field values at that wavelength, per object
        # the E-field size is given by (sp.grid_size x sp.grid_size)
        if ap.companion:
            self.wf_collection = np.empty((ap.n_wvl_init, 1 + len(ap.contrast)), dtype=object)
        else:
            self.wf_collection = np.empty((ap.n_wvl_init, 1), dtype=object)

        # Init Beam Ratios
        self.beam_ratios = np.zeros_like(self.wsamples)

        # Init Locations of saved E-field
        self.saved_planes = []  # string of locations where fields have been saved (should match sp.save_list after run is completed)
        self.Efield_planes = np.empty((0, np.shape(self.wf_collection)[0],  # array of saved complex field data at
                                           np.shape(self.wf_collection)[1],    # specified locations of the optical train
                                           sp.grid_size,
                                           sp.grid_size), dtype=np.complex64)
        # self.plane_sampling = np.empty((len(sp.save_list), ap.n_wvl_init))
        self.plane_sampling = []

    def initialize_proper(self):
        # Initialize the Wavefront in Proper
        for iw, w in enumerate(self.wsamples):
            # Scale beam ratio by wavelength for polychromatic imaging
            # see Proper manual pg 37
            # Proper is devised such that you get a Nyquist sampled image in the focal plane. If you optical system
            #  goes directly from pupil plane to focal plane, then you need to scale the beam ratio such that sampling
            #  in the focal plane is constant. You can check this with check_sampling, which returns the value from
            #  prop_get_sampling. If the optical system does not go directly from pupil-to-object plane at each optical
            #  plane, the beam ratio does not need to be scaled by wavelength, because of some optics wizardry that
            #  I don't fully understand. KD 2019
            if sp.focused_sys:
                self.beam_ratios[iw] = sp.beam_ratio
            else:
                self.beam_ratios[iw] = sp.beam_ratio * ap.wvl_range[0] / w
                # dprint(f"iw={iw}, w={w}, beam ratio is {self.beam_ratios[iw]}")

            # Initialize the wavefront at entrance pupil
            wfp = proper.prop_begin(tp.entrance_d, w, sp.grid_size, self.beam_ratios[iw])

            wfs = [wfp]
            names = ['star']

            # Initiate wavefronts for companion(s)
            if ap.companion:
                for ix in range(len(ap.contrast)):
                    wfc = proper.prop_begin(tp.entrance_d, w, sp.grid_size, self.beam_ratios[iw])
                    wfs.append(wfc)
                    names.append('companion_%i' % ix)

            for io, (iwf, wf) in enumerate(zip(names, wfs)):
                self.wf_collection[iw, io] = wf

    def loop_collection(self, func, *args, **kwargs):
        """
        For each wavelength and astronomical object apply a function to the wavefront.

        The wavefront object has dimensions of shape=(n_wavelengths, n_astro_objects, grid_sz, grid_sz)

        To save, you must pass in the keyword argument plane_name when you call this function from the perscription.
        This function does not have a keyword argument for plane_name specifically, since you need to distinguish
        it from the **kwargs you want to pass to the function that you are looping over.
        If you are saving the plane at this location, keep in mind it is saved AFTER the function is applied. This
        is desirable for most functions but be careful when using it for prop_lens, etc

        :param func: function to be applied e.g. ap.add_aber()
        :param plane_name: name the plane where this is called if you want to save the complex field data via save_plane
        :param args: args to be passed to the function
        :param kwargs: kwargs to be passed to the function
        :return: everything is just applied to the wfo, so nothing is returned in the traditional sense
        """
        if 'plane_name' in kwargs:
            plane_name = kwargs.pop('plane_name')  # remove plane_name from **kwargs
            if plane_name in sp.save_list:
                pass
            else:
                plane_name = None
        elif func.__name__ in sp.save_list:
            plane_name = func.__name__
        else:
            plane_name = None

        shape = self.wf_collection.shape
        for iw in range(shape[0]):
            for io in range(shape[1]):
                func(self.wf_collection[iw, io], *args, **kwargs)

        # Saving complex field data after function is applied
        if sp.save_fields and plane_name is not None:
            self.save_plane(location=plane_name)

    def save_plane(self, location=None):
        """
        Saves the complex field at a specified location in the optical system. If the function is called by
        wfo.loop_collection, the plane is saved AFTER the function is applied

        Note that the complex planes saved are not summed by object, interpolated over wavelength, nor masked down
        to the sp.maskd_size.

        :param location: name of plane where field is being saved
        :return: self.save_E_fields
        """
        if sp.verbose:
            dprint(f"saving plane at {location}")

        if location is not None and location in sp.save_list:
            shape = self.wf_collection.shape
            E_field = np.zeros((1, np.shape(self.wf_collection)[0],
                                       np.shape(self.wf_collection)[1],
                                       sp.grid_size,
                                       sp.grid_size), dtype=np.complex64)
            samp_lambda = np.zeros(ap.n_wvl_init)
            for iw in range(shape[0]):
                for io in range(shape[1]):
                    wf = proper.prop_shift_center(self.wf_collection[iw, io].wfarr)
                    E_field[0, iw, io] = copy.copy(wf)
                    samp_lambda[iw] = proper.prop_get_sampling(self.wf_collection[iw, 0])
                    # self.plane_sampling.append(proper.prop_get_sampling(self.wf_collection[iw,0]))

            self.Efield_planes = np.vstack((self.Efield_planes, E_field))
            self.saved_planes.append(location)
            self.plane_sampling.append(samp_lambda)

    def focal_plane(self):
        """
        ends the proper prescription and return sampling. most functionality involving image processing now in utils

        :return:
        """
        # Saving Complex Data via save_plane
        if sp.save_fields:                                 # save detector by default unless sp.save_fields is false
            self.save_plane(location='detector')           # shifting, etc already done in save_plane function

        cpx_planes = np.array(self.Efield_planes)
        sampling = np.array(self.plane_sampling)

        # Conex Mirror-- cirshift array for off-axis observing
        # if tp.pix_shift is not [0, 0]:
        #     datacube = np.roll(np.roll(datacube, tp.pix_shift[0], 1), tp.pix_shift[1], 2)

        return cpx_planes, sampling


####################################################################################################
# Functions Relating to Processing Complex Cubes
####################################################################################################
def interp_wavelength(data_in, ax):
    """
    Interpolating spectral cube from ap.n_wvl_init discreet wavelengths to ap.n_wvl_final

    :param data_in array where one axis contains the wavelength of the data
    :param ax  axis of wavelength
    :return data_out array that has been interpolated over axis=ax
    """
    # Interpolating spectral cube from ap.n_wvl_init discreet wavelengths to ap.n_wvl_final
    if ap.interp_wvl and 1 < ap.n_wvl_init < ap.n_wvl_final:
        wave_samps = np.linspace(0, 1, ap.n_wvl_init)
        f_out = interp1d(wave_samps, data_in, axis=ax)
        new_heights = np.linspace(0, 1, ap.n_wvl_final)
        data_out = f_out(new_heights)
    else:
        data_out = data_in

    return data_out


def extract_plane(data_in, plane_name):
    """
    pull out the specified plane of the detector from complex array

    here we assume that the data_in has the specific format of:
    [timestep, plane, object, wavelength, x, y]
      meaning that no dimensions have been removed from the original obs sequence
    Code will return invalid results if data_in is not in this format

    :param data_in: the 5D wavefront array of shape  [timestep, plane, wavelength, x, y]
    :param plane_name: the name of a plane you want to pull out, must match the plane name given in sp.plane_list

    :return sequence of data with format [tstep, obj, wavelength, x, y] (remove plane dimension)
    """
    ip = sp.save_list.index(plane_name)
    return data_in[:, ip, :, :, :, :]  # [tsteps, #wavelengths, x, y]--it automatically squeezes the plane axis


def cpx_to_intensity(data_in):
    """
    converts complex data to units of intensity

    WARNING: if you sum the data sequence over object or wavelength with simple case of np.sum(), must be done AFTER
    converting to intensity, else results are invalid
    """
    return np.abs(data_in)**2


def extract_center(wf):
    """
    extracts [sp.maskd_size, sp.maskd_size] from [sp.grid_size, sp.grid_size] data
    fp~focal plane
    code modified from the EXTRACT flag in prop_end

    :param wf: [sp.grid_size, sp.grid_size] array
    :returns: array with size [sp.maskd_size, sp.maskd_size]
    """
    smaller_wf = np.zeros((sp.maskd_size, sp.maskd_size))
    EXTRACT = sp.maskd_size
    nx,ny = wf.shape
    smaller_wf = wf[int(ny/2-EXTRACT/2):int(ny/2+EXTRACT/2),
                    int(nx/2-EXTRACT/2):int(nx/2+EXTRACT/2)]
    return smaller_wf


################################################################################################################
# Optics in Proper
################################################################################################################
def add_obscurations(wf, M2_frac=0, d_primary=0, d_secondary=0, legs_frac=0.05, plane_name=None):
    """
    adds central obscuration (secondary shadow) and/or spider legs as spatial mask to the wavefront

    :param wf: 2D proper wavefront
    :param M2_frac: ratio of tp.diam the M2 occupies
    :param d_primary: diameter of the primary mirror
    :param d_secondary: diameter of the secondary mirror
    :param legs_frac: fractional size of spider legs relative to d_primary
    :return: acts upon wfo, applies a spatial mask of s=circular secondary obscuration and possibly spider legs
    """
    if tp.obscure is False:
        pass
    else:
        # dprint('Including Obscurations')
        if M2_frac > 0 and d_primary > 0:
            proper.prop_circular_obscuration(wf, M2_frac * d_primary)
        elif d_secondary > 0:
            proper.prop_circular_obscuration(wf, d_secondary)
        else:
            raise ValueError('must either specify M2_frac and d_primary or d_secondary')
        if legs_frac > 0:
            proper.prop_rectangular_obscuration(wf, legs_frac*d_primary, d_primary*1.3, ROTATION=20)
            proper.prop_rectangular_obscuration(wf, d_primary*1.3, legs_frac * d_primary, ROTATION=20)


def prop_pass_lens(wf, fl_lens, dist):
    """
    pass the wavefront through a lens then propagate to the next surface

    :param wf: single wavefront of shape=(sp.grid_sz, sp.grid_sz)
    :param fl_lens: focal length in m
    :param dist: distance in m
    """
    proper.prop_lens(wf, fl_lens)
    proper.prop_propagate(wf, dist)


def abs_zeros(wf):
    """
    zeros everything outside the pupil

    This function attempts to figure out where the edges of the pupil are by determining if either the real
     or imaginary part of the complex-valued E-field is zero at a gridpoint. If so, it sets the full cpmplex
     value to zero, so 0+0j
    """
    bad_locs = np.logical_or(np.real(wf) == -0, np.imag(wf) == -0)
    wf[bad_locs] = 0 + 0j

    return wf


def rotate_sky(wf, it):
    raise NotImplementedError


def offset_companion(wfo):
    """
    offsets the companion wavefront using the 2nd and 3rd order Zernike Polynomials (X,Y tilt)
    companion(s) contrast and location(s) set in params

    We don't call this function via wfo.loop_collection because we need to know which object (io) we are on, which
    is not supported in the current format. This is the only acception to applying loop_collection

    Important: this function must be called AFTER any calls to proper.prop_define_entrance, which normalizes the
    intensity, because we scale the intensity of the planet relative to the star via the user-parameter ap.contrast.

    If you have a focused system, and do not scale the grid sampling of the system by wavelength, we account
    for that here (thus the if/else statements). This is because we shift the companion's location in the focal plane
    by proper.prop_zernikes, which scales the x/y tilt (zernike orders 2 and 3) by wavelength to account for the
    presumed resampling based on wavelength. We thus counteract that in the case of sp.focused_sys=True

    Wavelength/contrast scaling scales the contrast ratio between the star and planet as a function of wavelength.
    This ratio is given by ap.C_spec, and the scale ranges from 1/ap.C_spec to 1 as a function of ap.n_wvl_init. The
        gradient ap.C_spec should be chosen carefully to consider the number of wavelengths and spectral type of the
        star and planet in the simulation.

    :param wfo: wavefront object, shape=(n_wavelengths, n_astro_objects, grid_sz, grid_sz)
    :return: nothing implicitly returned but the given wfo initiated in Wavefronts class has been altered to give the
        appropriate wavefront for a planet in the focal plane
    """
    if ap.companion is True:
        cont_scaling = np.linspace(1. / ap.C_spec, 1, ap.n_wvl_init)

        for iw in range(wfo.wf_collection.shape[0]):
            for io in range(1, wfo.wf_collection.shape[1]):
                # Shifting the Array
                if sp.focused_sys:
                    # Scaling into lambda/D AND scaling by wavelength
                    xloc = ap.companion_xy[io-1][0] * wfo.wf_collection[iw,io].lamda / tp.entrance_d \
                           * ap.wvl_range[0] / wfo.wf_collection[iw,io].lamda # * (-1)**(iw%2)
                    yloc = ap.companion_xy[io-1][1] * wfo.wf_collection[iw,io].lamda / tp.entrance_d \
                            *  ap.wvl_range[0] / wfo.wf_collection[iw,io].lamda  # / (2*np.pi)   * (-1)**(iw%2)
                else:
                    # Scaling Happens Naturally!
                    xloc = ap.companion_xy[io-1][0]
                    yloc = ap.companion_xy[io-1][1]
                proper.prop_zernikes(wfo.wf_collection[iw, io], [2, 3], np.array([xloc, yloc]))  # zernike[2,3] = x,y tilt

                ##############################################
                # Wavelength/Contrast  Scaling the Companion
                ##############################################
                wfo.wf_collection[iw, io].wfarr *= np.sqrt(ap.contrast[io-1])

                #TODO implement wavelength-dependant scaling
                # Wavelength-dependent scaling by cont_scaling
                # wfo.wf_collection[iw, io].wfarr = wfo.wf_collection[iw, io].wfarr * np.sqrt(ap.contrast[io-1] * cont_scaling[iw])


def check_sampling(tstep, wfo, location, line_info, units=None):
    """
    checks the sampling of the wavefront at the given location and prints to console

    :param tstep: timestep, will only check for first timestep, so when tstep==0
    :param wfo: wavefront object
    :param location: string that identifies where call is being made
    :param line_info: info on the line number and function name from where check_sampling was called
        example: getframeinfo(stack()[0][0])
        via: from inspect import getframeinfo, stack
    :param units: desired units of returned print statement; options are 'mm,'um','nm','arcsec','rad'
    :return: prints sampling to the command line
    """
    if tstep == 0:
        print(f"From {line_info.filename}:{line_info.lineno}\n Sampling at {location}")
        for w in range(wfo.wf_collection.shape[0]):
            check_sampling = proper.prop_get_sampling(wfo.wf_collection[w,0])
            if units == 'mm':
                print(f"sampling at wavelength={wfo.wsamples[w]*1e9:.0f}nm is {check_sampling:.4f} m")
            elif units == 'um':
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e6:.1f} um")
            elif units == 'nm':
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e9:.1f} nm")
            elif units == 'arcsec':
                check_sampling = proper.prop_get_sampling_arcsec(wfo.wf_collection[w, 0])
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e3:.2f} mas")
            elif units == 'rad':
                check_sampling = proper.prop_get_sampling_radians(wfo.wf_collection[w, 0])
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling:.3f} rad")
            else:
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling} m")
