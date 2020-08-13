"""
adaptive.py

functions relating to simulating an AO system with Proper
mostly code copied from the original MEDIS from Rupert

Generally, the optical prescription will call the deformable_mirror function, which will compile all information and
run sequentially all functions related to creating the adaptive optic correction (main AO functionality, relating to
atmospheric and common-path aberrations) as well as using or not CDI probes, DM corrections or errors, etc

TODO
    Add astrogrid pattern functionality from MEDIS0

"""

import numpy as np
from scipy import interpolate, ndimage
from inspect import getframeinfo, stack
import matplotlib.pylab as plt
import proper

from medis.params import sp, tp
from medis.CDI import cdi, config_probe
from medis.optics import check_sampling, apodize_pupil, unwrap_phase_zeros as unwrap_phase
from medis.utils import dprint
# from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes


################################################################################
# Deformable Mirror
################################################################################
def deformable_mirror(wf, WFS_map, iter, previous_output=None, apodize=False, plane_name='', debug=False):
    """
    combine different DM actuator commands into single map to send to prop_dm

    prop_dm needs an input map of n_actuators x n_actuators in units of actuator command height. quick_ao will handle
    the conversion to actuator command height, and the CDI probe must be scaled in cdi.probe_amp in params in
    units of m. Each subroutine is also responsible for creating a map of n_actuators x n_actuators spacing. prop_dm
    handles the resampling of this map onto the wavefront, including the influence function. Its some wizardry that
    happens in c, and presumably it is taken care of so you don't have to worry about it.

    In the call to proper.prop_dm, we apply the flag tp.fit_dm, which switches between two 'modes' of proper's DM
    surface fitting. If FALSE, the DM is driven to the heights specified by dm_map, and the influence function will
    act on these heights to define the final surface shape applied to the DM, which may differ substantially from
    the initial heights specified by dm_map. If TRUE, proper will iterate applying the influence function to the
    input heights, and adjust the heights until the difference between the influenced-map and input map meets some
    proper-defined convergence criterea. Setting tp.fit_dm=TRUE will obviously slow down the code, but will (likely)
    more accurately represent a well-calibrated DM response function.

    much of this code copied over from example from Proper manual on pg 94

    :param wf: single wavefront
    :param WFS_map: wavefront sensor map, should be in units of phase delay
    :param previous_output:
    :param iter: the current index of iteration (which timestep this is)
    :param plane_name: name of plane (should be 'woofer' or 'tweeter' for best functionality)
    :return: nothing is returned, but the probe map has been applied to the DM via proper.prop_dm. DM plane post DM
        application can be saved via the sp.save_list functionality
    """
    assert np.logical_xor(WFS_map is None, previous_output is None)

    # AO Actuator Count from DM Type
    if plane_name == 'tweeter' and hasattr(tp,'act_tweeter'):
        nact = tp.act_tweeter
    elif plane_name == 'woofer' and hasattr(tp,'act_woofer'):
        nact = tp.act_woofer
    else:
        nact = tp.ao_act

    # DM Coordinates
    nact_across_pupil = nact - 2  # number of full DM actuators across pupil (oversizing DM extent)
    dm_xc = (nact / 2)   # The location of the optical axis (center of the wavefront) on the DM in
    dm_yc = (nact / 2)   # actuator units. First actuator is centered on (0.0, 0.0). The 0.5 is a
    #  parameter introduced/tuned by Rupert to remove weird errors (address this).
    # KD verified this needs to be here or else suffer weird errors 9/19
    # TODO address/remove the 0.5 in DM x,y coordinates

    ############################
    # Creating DM Surface Map
    ############################
    d_beam = 2 * proper.prop_get_beamradius(wf)  # beam diameter
    act_spacing = d_beam / nact_across_pupil  # actuator spacing [m]

    #######
    # AO
    #######
    if previous_output is not None and WFS_map is None:
        dm_map = update_dm(previous_output)
    else:
        dm_map = quick_ao(wf, nact, WFS_map[wf.iw])

    #########
    # Waffle
    #########
    if tp.satelite_speck['apply'] and plane_name is not 'woofer':
        waffle = make_speckle_kxy(tp.satelite_speck['xloc'], tp.satelite_speck['yloc'],
                                  tp.satelite_speck['amp'], tp.satelite_speck['phase'])
        waffle += make_speckle_kxy(tp.satelite_speck['xloc'], -tp.satelite_speck['yloc'],
                                   tp.satelite_speck['amp'], tp.satelite_speck['phase'])
        dm_map += waffle

    #######
    # CDI
    ######
    if cdi.use_cdi and plane_name == cdi.which_DM:
        theta = cdi.phase_series[iter]
        if not np.isnan(theta):
            # dprint(f"Applying CDI probe, lambda = {wfo.wsamples[iw]*1e9:.2f} nm")
            probe = config_probe(theta, nact, iw=wf.iw, ib=wf.ib)  # iw and ib only used for plotting, and only if sp.verbose=true
            dm_map = dm_map + probe  # Add Probe to DM map

    #########################
    # Applying Piston Error
    #########################
    if tp.piston_error:
        mean_dm_map = np.mean(np.abs(dm_map))
        var = 1e-4  # 1e-11
        dm_map = dm_map + np.random.normal(0, var, (dm_map.shape[0], dm_map.shape[1]))

    #########################
    # proper.prop_dm
    #########################
    dmap = proper.prop_dm(wf, dm_map, dm_xc, dm_yc, act_spacing, FIT=tp.fit_dm)  #

    if debug and wf.iw == 0 and wf.ib == 0:
        dprint(plane_name)
        check_sampling(wf, iter, plane_name+' DM pupil plane', getframeinfo(stack()[0][0]), units='mm')

        # plt.ion()
        fig, ax = plt.subplots(1,1)
        cax = ax.imshow(dm_map*1e9, interpolation='none', origin='lower')
        plt.title(f'{plane_name} dm_map (actuator coordinates)')
        cb = plt.colorbar(cax)
        cb.set_label('nm')
        plt.show(block=True)

        post_ao = unwrap_phase(proper.prop_get_phase(wf)) * wf.lamda / (2 * np.pi)
        # quick2D(pre_ao_dist*1e9, title='unwrapped wavefront before DM', zlabel='nm', show=False)  # , vlim=(-0.5e-7,0.5e-7))
        # quick2D(np.abs(pre_ao_amp)**2, title='Pre-AO Intensity', show=False)#, vlim=(-0.5e-7,0.5e-7))
        # quick2D(dmap, title='the phase map prop_dm is applying', zlabel='distance (m)', show=False)#, vlim=(-0.5e-7,0.5e-7))
        # plt.figure()
        # plt.plot(pre_ao_dist[len(pre_ao_dist)//2], label=f'pre_ao 1D cut, row {len(pre_ao_dist)//2}')
        # plt.plot(2*dmap[len(dmap)//2], label=f'dmap 1D cut (x2), row {len(dmap)//2}')
        # plt.plot((pre_ao_dist + (2*dmap))[len(dmap)//2], label='difference')
        # plt.legend()
        # plt.xlim(sp.grid_size//2*np.array([1-sp.beam_ratio*1.1, 1+sp.beam_ratio*1.1]))
        # quick2D(pre_ao + (2*dmap), title='diff', zlabel='m', show=False, vlim=(-0.5e-7,0.5e-7))
        # quick2D(post_ao, title='unwrapped wavefront after DM', zlabel='m', show=False, vlim=(-0.5e-7,0.5e-7))
        # quick2D(np.abs(proper.prop_get_amplitude(wf))**2, title='wavefront after DM intensity', show=False)
        # quick2D(proper.prop_get_phase(wf), title='wavefront after DM in phase units', zlabel='Phase',
        #          show=True)  # colormap='sunlight',

    if apodize:
        hardmask_pupil(wf)

    return dmap


################################################################################
# Ideal AO
################################################################################
def quick_ao(wf, nact, WFS_map):
    """
    calculate the offset map to send to the DM from the WFS map

    The main idea is to apply the DM only to the region of the wavefront that contains the beam. The phase map from
    the wfs saved the whole wavefront, so that must be cropped. During the wavefront initialization in
    wavefront.initialize_proper, the beam ratio set in sp.beam_ratio is scaled per wavelength (to achieve constant
    sampling sto create white light images), so the cropped value must also be scaled by wavelength. Note, beam ratio
    is scaled differently depending on if sp.focused_sys is True or not. See params-->sp.focused_sys and Proper
    manual pg 36 for more info.

    Then, we interpolate the cropped beam onto a grid of (n_actuators,n_actuators), such that the DM can apply a
    actuator height to each represented actuator, not a over or sub-sampled form. If the number of actuators is low
    compared to the number of samples on the beam, you should anti-alias the WFS map via a lowpass filter before
    interpolating. There is a discrepancy between the sampling of the wavefront at this location (the size you cropped)
    vs the size of the DM. proper.prop_dm handles this, so just plug in the n_actuator sized DM map with specified
    parameters, and assume that prop_dm handles the resampling correctly via the spacing or n_act_across_pupil flag.
    FYI the resampling is done via a c library you installed/compiled when installing proper.

    The WFS map is a map of real values in units of phase delay in radians. However, the AO map that gets passed to
    proper.prop_dm wants input in nm height of each actuator. Therefore, you need to convert the phase delay to
    a DM height. For the ideal AO, you would do this individually for each wavelength. However, for a 'real' AO system
    you do this for the median wavelength. You also need to account for a factor of 2, since the DM is modeled as
    a mirror so it travels the length of the phase delay twice.

    much of this code copied over from example from Proper manual on pg 94

    :param wfo: wavefront object created by optics.Wavefronts() [n_wavelengths, n_objects] of tp.gridsize x tp.gridsize
    :param WFS_map: returned from quick_wfs (as of Aug 2019, its an idealized image)
    :return: ao_map: map of DM actuator command heights in units of m
    """

    nact_across_pupil = nact-2          # number of full DM actuators across pupil (oversizing DM extent)
                                        # Note: oversample by 2 actuators hardcoded here, check if this is appropriate 
    
    ############################
    # Creating AO Surface Map
    ############################
    d_beam = 2 * proper.prop_get_beamradius(wf)  # beam diameter
    act_spacing = d_beam / nact_across_pupil  # actuator spacing [m]

    ###################################
    # Cropping the Beam from WFS map
    ###################################
    # cropping here by beam_ratio rather than d_beam is valid since the beam size was initialized
    #  using the scaled beam_ratios when the wfo was created
    ao_map = WFS_map[
             # sp.grid_size//2 - np.int_(wf.beam_ratio*sp.grid_size//2)+1:
             # sp.grid_size//2 + np.int_(wf.beam_ratio*sp.grid_size//2)+2,
             # sp.grid_size//2 - np.int_(wf.beam_ratio*sp.grid_size//2)+1:
             # sp.grid_size//2 + np.int_(wf.beam_ratio*sp.grid_size//2)+2]
             sp.grid_size//2 - np.int_(wf.beam_ratio*sp.grid_size//2)-1:
             sp.grid_size//2 + np.int_(wf.beam_ratio*sp.grid_size//2)+2,
             sp.grid_size//2 - np.int_(wf.beam_ratio*sp.grid_size//2)-1:
             sp.grid_size//2 + np.int_(wf.beam_ratio*sp.grid_size//2)+2]
    # dprint(f"WFS map coordinates are {sp.grid_size//2 - np.int_(wf.beam_ratio*sp.grid_size//2)-1},"
    #        f"{sp.grid_size//2 + np.int_(wf.beam_ratio*sp.grid_size//2)+1}")

    ########################################################
    # Interpolating the WFS map onto the actuator spacing
    # (tp.nact,tp.nact)
    ########################################################
    # Lowpass Filter- prevents aliasing; uses Gaussian filter
    nyquist_dm = tp.ao_act/2 * act_spacing  # [m]
    sigma = [nyquist_dm/2.355, nyquist_dm/2.355]  # assume we want sigma to be twice the HWHM
    ao_map = ndimage.gaussian_filter(ao_map, sigma=sigma, mode='nearest')

    f = interpolate.interp2d(range(ao_map.shape[0]), range(ao_map.shape[0]), ao_map, kind='cubic')
    ao_map = f(np.linspace(0,ao_map.shape[0],nact), np.linspace(0,ao_map.shape[0], nact))
    # map_spacing = proper.prop_get_sampling(wf)
    # ao_map = proper.prop_magnify(ao_map, map_spacing / act_spacing, nact, QUICK=True)

    ################################################
    # Converting phase delay to DM actuator height
    ################################################
    # Apply the inverse of the WFS image to the DM, so use -dm_map (dm_map is in phase units, divide by k=2pi/lambda)
    surf_height = proper.prop_get_wavelength(wf) / (4 * np.pi)  # [m/rad]
    ao_map = -ao_map * surf_height  # Converts DM map to units of [m] of actuator heights

    return ao_map


def retro_wfs(star_fields, wfo, plane_name='wfs'):
    """
    Retrospective wfs (measure an old field)

    :param star_fields:
    :param wfo:
    :param plane_name:
    :return:
    """
    WFS_map = np.zeros((len(star_fields), sp.grid_size, sp.grid_size))
    from skimage.restoration import unwrap_phase
    for iw in range(len(star_fields)):
        quick2D(np.angle(star_fields), title='before mask', colormap='sunlight')
        phasemap = np.angle(star_fields[iw])
        masked_phase = np.ma.masked_equal(phasemap, 0)
        quick2D(masked_phase, title='before unwrap', colormap='sunlight')
        WFS_map[iw] = unwrap_phase(masked_phase, wrap_around=[False, False])
        WFS_map[iw][phasemap == 0] = 0
        quick2D(WFS_map[iw], title='after')
    if 'retro_closed_wfs' in sp.save_list:
        wfo.save_plane(location='WFS_map')

    return WFS_map


def open_loop_wfs(wfo, plane_name='wfs'):
    """
    saves the unwrapped phase [arctan2(imag/real)] of the wfo.wf_collection at each wavelength

    It is an idealized image (exact copy) of the wavefront phase per wavelength. Only the map for the first object
    (the star) is saved. We have initialized

    Here we hardmask on the WFS map to be a circle around the beam in the pupil plane. This hard masking prevents the
     DM from acting on non-beam signal, since the DM modelled by proper is a nxn square array, but the beam is nominally
     circular for circular apertures.

    #TODO the way this is saved for naming the WFS_map is going to break if you want to do closed loop WFS on a
    #TODO woofer-tweeter system

    :param wfo: wavefront object
    :param plane_name: name of the plane to enable or disable saving the WFS map
    :return: array containing only the unwrapped phase delay of the wavefront; shape=[n_wavelengths], units=radians
    """
    from skimage.restoration import unwrap_phase

    star_wf = wfo.wf_collection[:, 0]
    WFS_map = np.zeros((len(star_wf), sp.grid_size, sp.grid_size))

    for iw in range(len(star_wf)):
        hardmask_pupil(star_wf[iw])
        phasemap = proper.prop_get_phase(star_wf[iw])
        WFS_map[iw] = unwrap_phase(phasemap, wrap_around=[False, False])
        WFS_map[iw][phasemap==0] = 0

    if 'open_loop_wfs' in sp.save_list or sp.closed_loop:
        wfo.save_plane(location='WFS_map')

    return WFS_map


def hardmask_pupil(wf):
    """
    hard-edged circular mask of the pupil plane.

    Masks out the WFS map outside of the beam since the DM modeled by proper can only be a square nxn grid of actuators,
    and thus the influence function surrounding each DM actuator could  be affecting on-beam pixels, even if the
    actuator is acting on off-beam signal. In other words, even if a cornerDM actuator doesn't actuate on the beam,
    if there was non-zero signal in the WFS map, it will try to act on it, and it could 'influence' nearby DM actuators
    that are acting on the beam.

    This hard-edged mask is different from prop_circular_aperture in that it does not anti-alias the edges of the mask
    based on the 'fill factor' of the edge pixels. Instead, it has a boolean mask to zero everything > a fixed radius,
    in this case determined by the grid size and beam ratio of each wavefront passed into it.

    :param wf: a single wavefront
    :return: nothing is returned but the wf passed into it has been masked

    """
    phase_map = proper.prop_get_phase(wf)
    amp_map = proper.prop_get_amplitude(wf)

    # Sizing the Mask
    h, w = wf.wfarr.shape[:2]
    center = (int(w / 2), int(h / 2))
    radius = np.floor(sp.grid_size * wf.beam_ratio / 2)  # Should scale with wavelength if sp.focused_system=False,
                                                        # np.ceil used to oversize map so don't clip the beam
    # Making the Circular Boolean Mask
    Y, X = np.mgrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    inds = dist_from_center <= radius

    # Applying the Mask to the Complex Array
    mask = np.zeros_like(phase_map)
    mask[inds] = 1
    masked = phase_map * mask
    wf.wfarr = proper.prop_shift_center(amp_map * np.cos(masked) + 1j * amp_map * np.sin(masked))

    if sp.debug and wf.ib == 0:
        dprint(f"Radius of hard-edge pupil mask is {radius} pixels")
        from medis.plot_tools import quick2D
        quick2D(masked, title=f"Masked phase map in hardmask_pupil, lambda={wf.lamda*1e9} nm", zlabel='phase?')


def make_speckle_kxy(kx, ky, amp, dm_phase):
    """given an kx and ky wavevector,
    generates a NxN flatmap that has
    a speckle at that position"""
    N = tp.ao_act
    dmx, dmy   = np.meshgrid(
                    np.linspace(-0.5, 0.5, N),
                    np.linspace(-0.5, 0.5, N))

    xm=dmx*kx*2.0*np.pi
    ym=dmy*ky*2.0*np.pi
    # print 'DM phase', dm_phase
    ret = amp*np.cos(xm + ym +  dm_phase)
    return ret


# def ao(wf, WFS_map, theta):
#     if sp.closed_loop:
#         deformable_mirror(wf, WFS_map, theta)
#     else:
#         WFS_map = open_loop_wfs(wf)  # overwrite WFS_map
#         # dprint(f"WFS_ma.shape = {WFS_map.shape}")
#         deformable_mirror(wf, WFS_map, theta)
#
################################################################################
# Full AO
################################################################################
# not implemented. Full AO implies a time delay, and maybe non-ideal WFS
