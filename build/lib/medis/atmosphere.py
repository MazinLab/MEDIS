"""
atmosphere.py

For mini-medis, code copied over from atmos.py and aberrations.py from Rupert's original MEDIS

Contains functions that will create, load, save, and apply atmospheric maps
"""
import numpy as np
import os
import astropy.io.fits as fits
import hcipy
import proper
from skimage.restoration import unwrap_phase
# from mkidpipeline.speckle.genphotonlist_IcIsIr import corrsequence
from scipy import special

from medis.params import iop, ap, tp, sp, atmp
from medis.utils import dprint, clipped_zoom
from medis.optics import circular_mask

def recursion(r,g, f, sqrt1mf2, n):
    for i in range(1, n):
        r[i] = r[i - 1]*f + g[i]*sqrt1mf2
    return r

def corrsequence(Ttot, tau):
    """
    Generate a sequence of correlated Gaussian noise, correlation time
    tau.  Algorithm is recursive and from Markus Deserno.  The
    recursion is implemented as an explicit for loop but has a lower
    computational cost than converting uniform random variables to
    modified-Rician random variables.

    Arguments:
    Ttot: int, the total integration time in microseconds.
    tau: float, the correlation time in microseconds.

    Returns:
    t: a list of integers np.arange(0, Ttot)
    r: a correlated Gaussian random variable, zero mean and unit variance, array of length Ttot

    """

    t = np.arange(Ttot)
    g = np.random.normal(0, 1, Ttot)
    r = np.zeros(g.shape)
    f = np.exp(-1. / tau)
    sqrt1mf2 = np.sqrt(1 - f ** 2)
    r = recursion(r, g, f, sqrt1mf2, g.shape[0])

    return t, r

def gen_atmos(plot=False, debug=True):
    """
    generates atmospheric phase distortions using hcipy (updated from original using CAOS)

    read more on HCIpy here: https://hcipy.readthedocs.io/en/latest/index.html
    In hcipy, the atmosphere evolves as a function of time, specified by the user. User can thus specify the
    timescale of evolution through both velocity of layer and time per step in the obs_sequence, in loop for
    medis_main.gen_timeseries().

    :param plot: turn plotting on or off
    :return:

    todo add simple mp.Pool.map code to make maps in parrallel
    """
    if tp.use_atmos is False:
        pass  # only make new atmosphere map if using the atmosphere
    else:

        if sp.verbose: dprint("Making New Atmosphere Model")
        # Saving Parameters
        # np.savetxt(iop.atmosconfig, ['Grid Size', 'Wvl Range', 'Number of Frames', 'Layer Strength', 'Outer Scale', 'Velocity', 'Scale Height', cp.model])
        # np.savetxt(iop.atmosconfig, ['ap.grid_size', 'ap.wvl_range', 'ap.numframes', 'atmp.cn_sq', 'atmp.L0', 'atmp.vel', 'atmp.h', 'cp.model'])
        # np.savetxt(iop.atmosconfig, [ap.grid_size, ap.wvl_range, ap.numframes, atmp.cn_sq, atmp.L0, atmp.vel, atmp.h, cp.model], fmt='%s')

        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)
        wavefronts = []

        ##################################
        # Initiate HCIpy Atmosphere Type
        ##################################
        pupil_grid = hcipy.make_pupil_grid(sp.grid_size, tp.entrance_d)
        if atmp.model == 'single':
            layers = [hcipy.InfiniteAtmosphericLayer(pupil_grid, atmp.cn_sq, atmp.L0, atmp.vel, atmp.h, 2)]
        elif atmp.model == 'hcipy_standard':
            # Make multi-layer atmosphere
            # layers = hcipy.make_standard_atmospheric_layers(pupil_grid, atmp.L0)
            heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
            velocities = np.array([10, 10, 10, 10, 10, 10])
            Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350]) * 3.5e-12

            layers = []
            for h, v, cn in zip(heights, velocities, Cn_squared):
                layers.append(hcipy.InfiniteAtmosphericLayer(pupil_grid, cn, atmp.L0, v, h, 2))

        elif atmp.model == 'evolving':
            raise NotImplementedError
        atmos = hcipy.MultiLayerAtmosphere(layers, scintilation=False)

        for wavelength in wsamples:
            wavefronts.append(hcipy.Wavefront(hcipy.Field(np.ones(pupil_grid.size), pupil_grid), wavelength))

        if atmp.correlated_sampling:
            # Damage Detection and Localization from Dense Network of Strain Sensors

            # fancy sampling goes here
            normal = corrsequence(sp.numframes, atmp.tau/sp.sample_time)[1] * atmp.std
            uniform = (special.erf(normal / np.sqrt(2)) + 1)

            times = np.cumsum(uniform) * sp.sample_time

            if debug:
                import matplotlib.pylab as plt
                plt.plot(normal)
                plt.figure()
                plt.plot(uniform)
                plt.figure()
                plt.hist(uniform)
                plt.figure()
                plt.plot(np.arange(0, sp.numframes * sp.sample_time, sp.sample_time))
                plt.plot(times)
                plt.show()
        else:
            times = np.arange(0, sp.numframes * sp.sample_time, sp.sample_time)

        ###########################################
        # Evolving Wavefront using HCIpy tools
        ###########################################
        for it, t in enumerate(times):
            atmos.evolve_until(t)
            for iw, wf in enumerate(wavefronts):
                wf2 = atmos.forward(wf)

                filename = get_filename(it, wsamples[iw], (iop.atmosdir, sp.sample_time,
                                                           atmp.model))
                if sp.verbose: dprint(f"atmos file = {filename}")
                hdu = fits.ImageHDU(wf2.phase.reshape(sp.grid_size, sp.grid_size))
                hdu.header['PIXSIZE'] = tp.entrance_d/sp.grid_size
                hdu.writeto(filename, overwrite=True)

                if plot and iw == 0:
                    import matplotlib.pyplot as plt
                    from medis.twilight_colormaps import sunlight
                    plt.figure()
                    plt.title(f"Atmosphere Phase Map t={t} lambda={eformat(wsamples[iw], 3, 2)}")
                    hcipy.imshow_field(wf2.phase, cmap=sunlight)
                    plt.colorbar()
                    plt.show(block=True)


def add_atmos(wf, it, param_tup=None, spatial_zoom=False):
    """
    creates a phase offset matrix for each wavelength at each time step,
    sampled from the atmosphere generated by hcipy

    HCIpy generates an atmosphere with given parameters. The returned field is in units of phase delay for each
    wavelength. prop_add_phase wants the phase delay to be in units of meters for each wavelength, so we convert to
    meters via the wavelength/np.pi

    :param wf: a single (2D) wfo.wf_collection[iw,ib] at one wavelength and object
    :param it: timestep# in obs_sequence. Comes from medis_main.gen_timeseries()
    :return: nothing returned, wfo is modified with proper.prop_add_phase
    """
    if tp.use_atmos is False:
        pass  # don't do anything. Putting this type of check here allows universal toggling on/off rather than
                # commenting/uncommenting in the proper perscription
    else:
        wavelength = wf.lamda  # the .lamda comes from proper, not from Wavefronts class

        # Check for Existing File)
        atmos_map = get_filename(it, wavelength, param_tup)
        if not os.path.exists(atmos_map):
            # todo remove when all test scripts use the new format
            # TODO check for new file name convention in addition to directory name
            print('atmospheres should be created at the beginng, not on the fly')
            raise NotImplementedError

        atm_map = fits.open(atmos_map)[1].data
        atm_map = unwrap_phase(atm_map)
        atm_map *= wavelength/(2*np.pi)  # converts atmosphere in units of phase delay (rad) into distance (m)

        if spatial_zoom:
            scale = ap.wvl_range[0] / wavelength
            atm_map = clipped_zoom(atm_map, scale)
            h, w = atm_map.shape[:2]
            mask = circular_mask(h, w, radius=scale * h * sp.beam_ratio / 2)
            atm_map[~mask] = 0
            atm_map[mask] -= np.mean(atm_map[mask])  # remove bias from spatial stretching

        proper.prop_add_phase(wf, atm_map)


def rotate_atmos(wf, it):
    time = it * ap.sample_time
    rotate_angle = tp.rot_rate * time
    wf.wfarr = proper.prop_shift_center(wf.wfarr)
    wf.wfarr = proper.prop_rotate(wf.wfarr, rotate_angle)
    wf.wfarr = proper.prop_shift_center(wf.wfarr)


def eformat(wvl, prec, exp_digits):
    """
    reformats wavelength into scientific notation in meters

    :param wvl: wavelength float
    :param prec: precision
    :param exp_digits: number of digits in the exponent
    :return:
    """
    s = "%.*e" % (prec, wvl)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def get_filename(it, lamda, param_tup=None):
    """
    returns the atmosphere map names in the format location/atmos_t<time>_<model>_wvl<wavelength>
    example output:

    :param it: time index
    :param lamda: wavelength, comes from metadata in proper wavefront object (ie wfo.wf_collection.lamda)
    :return:
    """
    wave = eformat(lamda, 3, 2)
    if param_tup:
        atmosdir, sample_time, model = param_tup
    else:
        atmosdir, sample_time, model = iop.atmosdir, sp.sample_time, atmp.model

    sigfig = int(np.abs(np.floor(np.log10(sp.sample_time))))

    return f'{atmosdir}/atmos_t{sample_time*it:.6f}_{model}_wvl{wave}.fits'