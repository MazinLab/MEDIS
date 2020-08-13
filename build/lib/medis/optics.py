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
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from inspect import getframeinfo, stack

from medis.twilight_colormaps import sunlight
from medis.params import ap, tp, sp
from medis.utils import dprint
from medis.distribution import planck


class Wavefront(proper.WaveFront):
    """ Wrapper for proper.Wavefront that stores source and wavelength info """

    def __init__(self, wavefront, lamda, name, beam_ratio, iw, ib):
        self.lamda = lamda
        self.name = name
        self.beam_ratio = beam_ratio
        self.iw = iw
        self.ib = ib

        for attr in dir(wavefront):
            if not hasattr(self, attr):
                setattr(self, attr, getattr(wavefront, attr))


class Wavefronts():
    """
    An object containing all of the complex E fields for each sampled wavelength and astronomical object at this tstep

    :params

    :returns
    self.wf_collection: a 2D array, where each element is its own 2D proper wavefront object structured as
     self.wf_collection[iw,ib]. Here, iw, ib is each wavefront and object, respectively
        ...meaning its an array of arrays.
        thus, self.wf_collection[iw,ib] is itself a 2D array of complex data. Its size is [sp.grid_size, sp.grid_size]
        we will call each instance of the collection a single wavefront wf
    self.save_E_fields: a matrix of E fields (proper.WaveFront.wfarr) at specified locations in the chain
    """
    def __init__(self, debug=False):

        # Using Proper to propagate wavefront from primary through optical system, loop over wavelength
        self.debug = debug
        self.wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)  # units set in params (should be m)
        self.num_bodies = 1 + len(ap.contrast) if ap.companion else 1
        # wf_collection is an array of arrays; the wf_collection is (number_wavelengths x number_astro_bodies)
        # each 2D field in the wf_collection is the 2D array of complex E-field values at that wavelength, per object
        # the E-field size is given by (sp.grid_size x sp.grid_size)

        ############################
        # Create Wavefront Array
        ############################
        self.wf_collection = np.empty((len(self.wsamples), self.num_bodies), dtype=object)

        # Init Locations of saved E-field
        self.saved_planes = []  # string of locations where fields have been saved (should match sp.save_list after run is completed)
        self.Efield_planes = np.empty((0, np.shape(self.wf_collection)[0],  # array of saved complex field data at
                                           np.shape(self.wf_collection)[1],    # specified locations of the optical train
                                           sp.grid_size,
                                           sp.grid_size), dtype=np.complex64)
        # self.plane_sampling = np.empty((len(sp.save_list), ap.n_wvl_init))
        self.plane_sampling = []

        self.spectra = []
        for object_spectrum in ap.spectra:
            if isinstance(object_spectrum, (int, float)):
                spectrum = planck(object_spectrum, self.wsamples)
                spectrum /= np.sum(spectrum)
            else:
                spectrum = np.ones((ap.n_wvl_init))
            self.spectra.append(spectrum)

    def initialize_proper(self, set_up_beam=False):
        """

        Initialize the Wavefronts in Proper

        :param set_up_beam: bool applies prop_circular_aperture and prop_define_entrance before spectral scaling
            instead of during prescription wher it normally goes

        returns wf_colllection attribute array of wavefronts
        """
        for iw, wavelength in enumerate(self.wsamples):
            # Scale beam ratio by wavelength for polychromatic imaging
            # see Proper manual pg 37
            # Proper is devised such that you get a Nyquist sampled image in the focal plane. If you optical system
            #  goes directly from pupil plane to focal plane, then you need to scale the beam ratio such that sampling
            #  in the focal plane is constant. You can check this with check_sampling, which returns the value from
            #  prop_get_sampling. If the optical system does not go directly from pupil-to-object plane at each optical
            #  plane, the beam ratio does not need to be scaled by wavelength, because of some optics wizardry that
            #  I don't fully understand. KD 2019
            if sp.focused_sys:
                beam_ratio = sp.beam_ratio
            else:
                beam_ratio = sp.beam_ratio * ap.wvl_range[0] / wavelength
                # dprint(f"iw={iw}, w={w}, beam ratio is {self.beam_ratios[iw]}")

            # Initialize the wavefront at entrance pupil
            wfp = proper.prop_begin(tp.entrance_d, wavelength, sp.grid_size, beam_ratio)
            if set_up_beam:
                proper.prop_circular_aperture(wfp, radius = tp.entrance_d / 2)
                proper.prop_define_entrance(wfp)  # normalizes the intensity
            wfp.wfarr = np.multiply(wfp.wfarr, np.sqrt(self.spectra[0][iw]), out=wfp.wfarr, casting='unsafe')

            wfs = [wfp]
            names = ['star']

            # Initiate wavefronts for companion(s)
            if ap.companion:
                for ix in range(len(ap.contrast)):
                    wfc = proper.prop_begin(tp.entrance_d, wavelength, sp.grid_size, beam_ratio)
                    if set_up_beam:
                        proper.prop_circular_aperture(wfc, radius=tp.entrance_d / 2)
                        proper.prop_define_entrance(wfc)  # normalizes the intensity
                    wfc.wfarr = np.multiply(wfc.wfarr, np.sqrt(self.spectra[ix][iw]), out=wfc.wfarr, casting='unsafe')
                    wfs.append(wfc)
                    names.append('companion_%i' % ix)

            for io, (name, wf) in enumerate(zip(names, wfs)):
                self.wf_collection[iw, io] = Wavefront(wf, wavelength, name, beam_ratio, iw, io)

    def loop_collection(self, func, *args, **kwargs):
        """
        For each wavelength and astronomical object apply a function to the wavefront.

        The wavefront object has dimensions of shape=(n_wavelengths, n_astro_bodies, grid_sz, grid_sz)

        To save, you must pass in the keyword argument plane_name when you call this function from the prescription.
        Specifically for the AO/DM system, there is different behavior in the function depending on the plane name.
        Therefore, we need to pass the plane_name through to ao.deformable_mirror. Otherwise, we just use it for saving
        and we can pop it out of the kwargs list because no other functions use it for dependant functionality.

        If you are saving the plane at this location, keep in mind it is saved AFTER the function is applied. This
        is desirable for most functions but be careful when using it for prop_lens, etc

        :param func: function to be applied e.g. ap.add_aber()
        :param args: args to be passed to the function
        :param kwargs: kwargs to be passed to the function
        :return: everything is just applied to the wfo, so nothing is returned in the traditional sense
        """
        if 'plane_name' in kwargs:
            if kwargs['plane_name'] == 'woofer' or kwargs['plane_name'] == 'tweeter':
                plane_name = kwargs['plane_name']
                pass  # keep the plane_name in kwargs.
            else:
                plane_name = kwargs.pop('plane_name')  # remove plane_name from **kwargs
        elif 'lens_name' in kwargs:
            plane_name = kwargs['lens_name']
        else:
            plane_name = func.__name__

        if 'zero_outside' in kwargs:
            raise DeprecationWarning('Zero outside issues fixed by pupil masking in adaptive.py')

        # manipulator_output is a way to store the values of a function passed into loop_collections.
        # EG you could use this as a way to save your WFS map if desired. The WFS map would be returned
        # as a 2D array by a WFS function (eg ao.closed_loop_wfs()) and then we could call it later and use it
        # or  just save it and plot it
        # also if we don't use it we can get rid of some of it
        # manipulator_output = np.empty(self.wf_collection.shape)
        manipulator_output = [[[] for _ in range(len(self.wsamples))] for _ in range(self.num_bodies)]
        for iw, sources in enumerate(self.wf_collection):
            for io, wavefront in enumerate(sources):
                if not func.__name__ in sp.skip_functions:
                    manipulator_output[io][iw] = func(wavefront, *args, **kwargs)

        # Show phase and amplitude of the plane during debugging
        if self.debug and not func.__name__ in sp.skip_functions:
            self.quicklook(title=plane_name)

        manipulator_output = np.array(manipulator_output)
        # if there's at least one not np.nan element add this array to the Wavefronts obj
        if np.any(manipulator_output != None):
            setattr(self, plane_name, manipulator_output)

        # Saving complex field data after function is applied
        if plane_name in sp.save_list:
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
            E_field = np.zeros((1, np.shape(self.wf_collection)[0],
                                       np.shape(self.wf_collection)[1],
                                       sp.grid_size,
                                       sp.grid_size), dtype=np.complex64)
            samp_lambda = np.zeros(ap.n_wvl_init)

            for iw, sources in enumerate(self.wf_collection):
                samp_lambda[iw] = proper.prop_get_sampling(self.wf_collection[iw,0])
                for io, wavefront in enumerate(sources):
                    wf = proper.prop_shift_center(wavefront.wfarr)
                    E_field[0, iw, io] = copy.copy(wf)

            self.Efield_planes = np.vstack((self.Efield_planes, E_field))
            self.saved_planes.append(location)
            self.plane_sampling.append(samp_lambda)

    def focal_plane(self):
        """
        ends the proper prescription and return sampling. most functionality involving image processing now in utils

        :return:
        """
        # Saving Complex Data via save_plane
        self.save_plane(location='detector')           # shifting, etc already done in save_plane function

        cpx_planes = np.array(self.Efield_planes)
        sampling = np.array(self.plane_sampling)

        # Conex Mirror-- cirshift array for off-axis observing
        # if tp.pix_shift is not [0, 0]:
        #     datacube = np.roll(np.roll(datacube, tp.pix_shift[0], 1), tp.pix_shift[1], 2)

        return cpx_planes, sampling

    def quicklook(self, wf=None, logZ=True, show=True, title=None):
        """
        Produces a figure with an image of amplitude and one of phase as well as 1D slices through these images

        :param wf: optics.Wavefront
        :param logZ: bool logarithmic Z scaling
        :param show: bool display figure now or leave show() to be called by user later on
        :param title: str
        :return:
        """
        if wf == None:
            wf = self.wf_collection[0,0]

        amp_map = proper.prop_get_amplitude(wf)
        phase_map = proper.prop_get_phase(wf)

        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=2)  # ((shape of grid to place axis x,y),(location x,y))
        ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=2)
        # ax3 = plt.subplot2grid((3, 2), (2, 0))
        # ax4 = plt.subplot2grid((3, 2), (2, 1))
        if logZ:
            im1 = ax1.imshow(amp_map, origin='lower', cmap="YlGnBu_r", norm=LogNorm())
        else:
            im1 = ax1.imshow(amp_map, origin='lower', cmap="YlGnBu_r")
        ax1.set_title('Amplitude Map')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(phase_map, origin='lower', cmap=sunlight)  # , vmin=-0.5, vmax=0.5)
        ax2.set_title('Phase Map')
        plt.colorbar(im2, ax=ax2)

        # ax3.plot(after_dm[int(sp.grid_size / 2)])
        # ax3.plot(np.sum(np.eye(sp.grid_size) * after_dm, axis=1), label=f'row {int(sp.grid_size / 2)}')
        # ax3.legend()
        #
        # # plt.plot(np.sum(after_dm,axis=1)/after_dm[128,128])
        #
        # ax4.plot(phase_afterdm[int(sp.grid_size / 2)], label=f'row {int(sp.grid_size / 2)}')
        # ax4.legend()
        # # ax4.plot(np.sum(np.eye(ap.grid_size)*phase_afterdm,axis=1))
        # plt.xlim([0, proper.prop_get_gridsize(wf)])
        if not title:
            title = input(f"Always Add A Title\n "
                          f"Please Enter Plane Name:")
            fig.suptitle(f"plane: {title}, lambda: {wf.lamda} m, body: {wf.name}", fontsize=18)
        else:
            fig.suptitle(f"plane: {title}, lambda: {wf.lamda} m, body: {wf.name}", fontsize=18)

        plt.subplots_adjust(top=0.9)

        if show:
            plt.show(block=True)


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


def interp_sampling(sampling):
    if ap.interp_wvl and 1 < ap.n_wvl_init < ap.n_wvl_final:
        wave_samps = np.linspace(0, 1, ap.n_wvl_init)
        f_out = interp1d(wave_samps, sampling, axis=1)  # axis 1 because sa
        new_heights = np.linspace(0, 1, ap.n_wvl_final)
        data_out = f_out(new_heights)
    else:
        data_out = sampling

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


def extract_center(wf, new_size=sp.maskd_size):
    """
    extracts [new_size, new_size] from [sp.grid_size, sp.grid_size] data
    fp~focal plane
    code modified from the EXTRACT flag in prop_end

    :param wf: [sp.grid_size, sp.grid_size] array
    :returns: array with size [new_size, new_size]
    """
    smaller_wf = np.zeros((new_size, new_size))
    EXTRACT = new_size
    nx,ny = wf.shape
    smaller_wf = wf[int(ny/2-EXTRACT/2):int(ny/2+EXTRACT/2),
                    int(nx/2-EXTRACT/2):int(nx/2+EXTRACT/2)]
    return smaller_wf


####################################################################################################
# Functions Relating to Masking the Pupil Plane
####################################################################################################
def circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    # Y, X = np.mgrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return mask


def apodize_pupil(wf):
    phase_map = proper.prop_get_phase(wf)
    amp_map = proper.prop_get_amplitude(wf)
    h, w = wf.wfarr.shape[:2]
    wavelength = wf.lamda
    scale = ap.wvl_range[0] / wavelength
    inds = circular_mask(h, w, radius=scale * 0.95 * h * sp.beam_ratio / 2)  #TO DO remove hardcoded sizing, especially if use is default
    mask = np.zeros_like(phase_map)
    mask[inds] = 1
    smooth_mask = gaussian_filter(mask, 1.5, mode='nearest')  #TO DO remove hardcoded sizing, especially if use is default
    smoothed = phase_map * smooth_mask
    wf.wfarr = proper.prop_shift_center(amp_map * np.cos(smoothed) + 1j * amp_map * np.sin(smoothed))


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


################################################################################################################
# Optics in Proper
################################################################################################################
def add_obscurations(wf, M2_frac=0, d_primary=0, d_secondary=0, legs_frac=0.05, plane_name=None):
    """
    adds central obscuration (secondary shadow) and/or spider legs as spatial mask to the wavefront

    :param wf: 2D proper wavefront
    :param M2_frac: ratio of tp.entrance_d the M2 occupies
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
            proper.prop_rectangular_obscuration(wf, legs_frac, 1.3, ROTATION=-20, NORM=True)
            proper.prop_rectangular_obscuration(wf, 1.3, legs_frac, ROTATION=-20, NORM=True)


def prop_pass_lens(wf, fl_lens, dist):
    """
    pass the wavefront through a lens then propagate to the next surface

    :param wf: single wavefront of shape=(sp.grid_sz, sp.grid_sz)
    :param fl_lens: focal length in m
    :param dist: distance in m
    """
    proper.prop_lens(wf, fl_lens)
    proper.prop_propagate(wf, dist)


def offset_companion(wf):
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

    :param wf: singe wavefront object from wfo.wf_collection, shape=(grid_sz, grid_sz)
    :return: nothing implicitly returned but the given wfo initiated in Wavefronts class has been altered to give the
        appropriate wavefront for a planet in the focal plane
    """
    if ap.companion is True and wf.name != 'star':
        cont_scaling = np.linspace(1. / ap.C_spec, 1, ap.n_wvl_init)

        # Shifting the Array
        if sp.focused_sys:
            # Scaling into lambda/D AND scaling by wavelength
            xloc = ap.companion_xy[wf.ib-1][0] * wf.lamda / tp.entrance_d \
                   * ap.wvl_range[0] / wf.lamda # * (-1)**(iw%2)
            yloc = ap.companion_xy[wf.ib-1][1] * wf.lamda / tp.entrance_d \
                    *  ap.wvl_range[0] / wf.lamda  # / (2*np.pi)   * (-1)**(iw%2)
        else:
            # Scaling Happens Naturally!
            xloc = ap.companion_xy[wf.ib-1][0]*1e-6
            yloc = ap.companion_xy[wf.ib-1][1]*1e-6
        proper.prop_zernikes(wf, [2, 3], np.array([xloc, yloc]))  # zernike[2,3] = x,y tilt

        ##############################################
        # Wavelength/Contrast  Scaling the Companion
        ##############################################
        wf.wfarr *= np.sqrt(ap.contrast[wf.ib-1])

        #TODO implement wavelength-dependant scaling
        # Wavelength-dependent scaling by cont_scaling
        # wf = wf * np.sqrt(ap.contrast[wf.ib-1] * cont_scaling[wf.iw])


####################################################################################################
# Check Sampling & Misc Tools
####################################################################################################
def check_sampling(wf, tstep, location, line_info, units=None):
    """
    checks the sampling of the wavefront at the given location and prints to console

    mostly reformats the output of proper.prop_get_sampling into human-readable format

    :param wf: wavefront object
    :param tstep: timestep, will only check for first timestep, so when tstep==0
    :param location: string that identifies where call is being made
    :param line_info: info on the line number and function name from where check_sampling was called
        example: getframeinfo(stack()[0][0])
        via: from inspect import getframeinfo, stack
    :param units: desired units of returned print statement; options are 'mm,'um','nm','arcsec','rad'
    :return: prints sampling to the command line
    """
    if tstep == 0 and wf.ib == 0:
        if wf.iw == 0:
            print(f"\nFrom {line_info.filename}:{line_info.lineno}\n Sampling at {location}")

        check_sampling = proper.prop_get_sampling(wf)
        if units == 'mm':
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling*1e3:.4f} mm")
        elif units == 'um':
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling*1e6:.1f} um")
        elif units == 'nm':
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling*1e9:.1f} nm")
        elif units == 'arcsec':
            check_sampling = proper.prop_get_sampling_arcsec(wf)
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling*1e3:.2f} mas")
        elif units == 'rad':
            check_sampling = proper.prop_get_sampling_radians(wf)
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling:.3f} rad")
        else:
            print(f"sampling at wavelength={wf.lamda * 1e9:.0f}nm is {check_sampling} m")


def unwrap_phase_zeros(phasemap):
    """ combination of abs_zeros and masking allows phase unwrap to work without discontiuities sometimes occur """
    masked_phase = np.ma.masked_equal(phasemap, 0)
    unwrap = unwrap_phase(masked_phase, wrap_around=[False, False])
    unwrap[phasemap == 0] = 0
    return unwrap
