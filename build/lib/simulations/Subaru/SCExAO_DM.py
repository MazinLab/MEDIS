"""
model the SXExAO optical system starting with the DM and ending with the focal plane. This script is primarily used
to model the idealized focal plane response to probe patterns presented on the DM in the absence of other aberrations
or atmospheric effects on the telescope or optical train.

"""

import numpy as np
from inspect import getframeinfo, stack
import proper

from medis.params import sp, ap, tp
from medis.utils import dprint
import medis.optics as opx
import medis.aberrations as aber
import medis.adaptive as ao
import medis.coronagraphy as cg

#################################################################################################
#################################################################################################
#################################################################################################

# ------------------------------
# SCExAO
tp.entrance_d = 0.051  # diameter of optics in SCExAO train are 2 inches=0.051 m

tp.d_tweeter = 0.051  # diameter of optics in SCExAO train are 2 inches=0.051 m
tp.act_tweeter = 50  # approx a 2000 actuator DM, (45x45=2025)
tp.fl_SxOAPG = 0.255  # m focal length of Genera SCExAO lens (OAP1,3,4,5)
tp.fl_SxOAP2 = 0.519  # m focal length of SCExAO OAP 2
tp.d_SxOAPG = 0.051  # diameter of SCExAO OAP's

tp.dist_SxOAP1_scexao = 0.1345  # m
tp.dist_scexao_sl2 = 0.2511 - tp.dist_SxOAP1_scexao  # m
tp.dist_sl2_focus = 0.1261  # m

tp.lens_params = [{'aber_vals': [7.2e-17, 0.8, 3.1],
                   'diam': tp.d_SxOAPG,
                   'fl': tp.fl_SxOAPG,
                   'dist': tp.fl_SxOAPG,
                   'name': 'SxOAPG'},

                 {'aber_vals': [7.2e-17, 0.8, 3.1],
                   'diam': tp.d_SxOAPG,
                   'fl': tp.fl_SxOAP2,
                   'dist': tp.fl_SxOAP2,
                   'name': 'SxOAP2'}
                 ]
# ------------------------------
# Coronagraph
tp.cg_type = 'Gaussian'
# tp.cg_size = 3  # physical size or lambda/D size
tp.cg_size = 1.5  # physical size or lambda/D size
tp.cg_size_units = "l/D"  # "m" or "l/D"
tp.fl_cg_lens = tp.fl_SxOAPG  # m
tp.lyot_size = 0.95  # units are in fraction of surface un-blocked

#################################################################################################
#################################################################################################
#################################################################################################

def SCExAO_DM(empty_lamda, grid_size, PASSVALUE):
    """
    propagates instantaneous complex E-field thru Subaru from the primary through SCExAO

    this function is called a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere,
        then telescope, to the focal plane
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time in the focal_plane()
    """
    # print("Propagating Broadband Wavefront Through Subaru")

    # Initialize the Wavefront in Proper
    wfo = opx.Wavefronts(sp.debug)
    wfo.initialize_proper()


    # Defines aperture (baffle-before primary)
    wfo.loop_collection(proper.prop_circular_aperture,
                        **{'radius': tp.entrance_d / 2})  # clear inside, dark outside
    wfo.loop_collection(proper.prop_define_entrance, plane_name='entrance_pupil')  # normalizes abs intensity

    # Test Sampling
    if sp.verbose:
        wfo.loop_collection(opx.check_sampling, PASSVALUE['iter'], "Telescope Aperture",
                            getframeinfo(stack()[0][0]), units='mm')
    # Testing Primary Focus (instead of propagating to focal plane)
    # wfo.loop_collection(opx.prop_pass_lens, tp.flen_nsmyth, tp.flen_nsmyth)  # test only going to prime focus

    ########################################
    # SCExAO
    # #######################################
    # AO System
    if tp.use_ao:
        WFS_map = ao.open_loop_wfs(wfo)
        wfo.loop_collection(ao.deformable_mirror, WFS_map, PASSVALUE['iter'], apodize=False,
                            plane_name='tweeter', debug=sp.verbose)
    # ------------------------------------------------
    wfo.loop_collection(proper.prop_propagate, tp.fl_SxOAPG)  # from tweeter-DM to OAP2

    # SXExAO Reimaging 2
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='SxOAP2')
    wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders)/2)
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_SxOAP2, tp.fl_SxOAP2, plane_name='post-DM-focus')  #tp.dist_sl2_focus

    # wfo.loop_collection(opx.check_sampling, PASSVALUE['iter'], "post-DM-focus",
    #                     getframeinfo(stack()[0][0]), units='nm')

    # Coronagraph
    # settings should be put into tp, and are not implicitly passed here
    wfo.loop_collection(cg.coronagraph, occulter_mode=tp.cg_type, plane_name='coronagraph')

    ########################################
    # Focal Plane
    # #######################################
    # Check Sampling in focal plane
    # wfo.focal_plane fft-shifts wfo from Fourier Space (origin==lower left corner) to object space (origin==center)
    cpx_planes, sampling = wfo.focal_plane()

    if sp.verbose:
        wfo.loop_collection(opx.check_sampling, PASSVALUE['iter'], "focal plane",
                            getframeinfo(stack()[0][0]), units='nm')
        # opx.check_sampling(PASSVALUE['iter'], wfo, "focal plane", getframeinfo(stack()[0][0]), units='arcsec')

    if sp.verbose:
        print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    return cpx_planes, sampling

