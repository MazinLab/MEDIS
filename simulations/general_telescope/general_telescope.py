"""
Prescription containing a full system that can be fully toggled using switches in the parent module

*** This module is currently unverified ***

TODO
    * verify this script using all the modes
    * reimplement tiptilt again?

"""

import numpy as np
import proper

from medis.params import ap, tp, iop, sp
import medis.atmosphere as atmos
import medis.adaptive as ao
import medis.aberrations as aber
import medis.optics as opx
from medis.coronagraphy import coronagraph, vortex, apodization


def general_telescope(empty_lamda, grid_size, PASSVALUE):
    """
    #TODO pass complex datacube for photon phases

    propagates instantaneous complex E-field through the optical system in loop over wavelength range

    this function is called as a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere, then telescope, to the focal plane
    currently: optics system "hard coded" as single aperture and lens
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time
    """
    # print("Propagating Broadband Wavefront Through Telescope")

    # datacube = []
    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    ###################################################
    # Aperture, Atmosphere, and Secondary Obscuration
    ###################################################
    # Defines aperture (baffle-before primary)
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': tp.entrance_d/2})
    # wfo.loop_collection(proper.prop_define_entrance)  # normalizes the intensity

    # Obscure Baffle
    if tp.obscure:
        wfo.loop_collection(opx.add_obscurations, M2_frac=1/8, d_primary=tp.entrance_d, legs_frac=tp.legs_frac)

    # Pass through a mini-atmosphere inside the telescope baffle
    #  The atmospheric model used here (as of 3/5/19) uses different scale heights,
    #  wind speeds, etc to generate an atmosphere, but then flattens it all into
    #  a single phase mask. The phase mask is a real-valued delay lengths across
    #  the array from infinity. The delay length thus corresponds to a different
    #  phase offset at a particular frequency.
    # quicklook_wf(wfo.wf_collection[0,0])
    if tp.use_atmos:
        wfo.loop_collection(atmos.add_atmos, PASSVALUE['iter'], plane_name='atmosphere')
        # quicklook_wf(wfo.wf_collection[0, 0])

    # quicklook_wf(wfo.wf_collection[0,0])
    #TODO rotate atmos not yet implementid in 2.0
    # if tp.rotate_atmos:
    #     wfo.loop_collection(aber.rotate_atmos, *(PASSVALUE['iter']))

    # Both offsets and scales the companion wavefront
    if wfo.wf_collection.shape[1] > 1:
        wfo.loop_collection(opx.offset_companion)
        wfo.loop_collection(proper.prop_circular_aperture,
                               **{'radius': tp.entrance_d / 2})  # clear inside, dark outside

    # TODO rotate atmos not yet implementid in 2.0
    # if tp.rotate_sky:
    #     wfo.loop_collection(opx.rotate_sky, *PASSVALUE['iter'])

    ########################################
    # Telescope Primary-ish Aberrations
    #######################################
    # Abberations before AO
    if tp.use_CPA:
        wfo.loop_collection(aber.add_aber, tp.entrance_d, tp.aber_params, tp.aber_vals,
                               step=PASSVALUE['iter'], lens_name='CPA')
    # wfo.loop_collection(proper.prop_circular_aperture, **{'radius': tp.entrance_d / 2})
        # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)

    #######################################
    # AO
    #######################################
    if tp.use_ao:
        if not sp.closed_loop:
            WFS_map = ao.open_loop_wfs(wfo)
            # tiptilt = np.zeros((sp.grid_size,sp.grid_size))
        else:
            #TODO Rupert-CPA maps are no longer generated. Not sure what you want to do here. The CPA map is stored
            # as a fits file at f"{iop.aberdir}/t{iter}_CPA.fits"
            #
            WFS_map = PASSVALUE['CPA_maps']
            # tiptilt = PASSVALUE['tiptilt']

        # if tp.include_tiptilt:
        #     CPA_maps, PASSVALUE['tiptilt'] = ao.tiptilt(wfo, CPA_maps, tiptilt)

        if tp.include_dm:
            wfo.loop_collection(ao.deformable_mirror, WFS_map, PASSVALUE['iter'])

   ########################################
    # Post-AO Telescope Distortions
    # #######################################
    # Abberations after the AO Loop
    if tp.use_NCPA:
        aber.add_aber(wfo, tp.f_lens, tp.aber_params, tp.aber_vals, PASSVALUE['iter'], lens_name='NCPA')
        wfo.loop_collection(proper.prop_circular_aperture, **{'radius': tp.entrance_d / 2})
        # TODO does this need to be here?
        # wfo.loop_collection(opx.add_obscurations, tp.entrance_d/4, legs=False)
        # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)

    # Low-order aberrations
    if tp.use_zern_ab:
        wfo.loop_collection(aber.add_zern_ab)

    if tp.use_apod:
        wfo.loop_collection(apodization, True)

    # First Focusing Optics
    wfo.loop_collection(opx.prop_pass_lens, tp.f_lens, tp.f_lens)

    ########################################
    # Coronagraph
    ########################################
    # there are additional un-aberated optics in the coronagraph module
    if tp.use_coronagraph:
        wfo.loop_collection(coronagraph, *(tp.f_lens, tp.occulter_type, tp.occult_loc, tp.entrance_d))

    ########################################
    # Focal Plane
    ########################################
    cpx_planes, sampling = wfo.focal_plane()

    print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    return cpx_planes, wfo.plane_sampling
