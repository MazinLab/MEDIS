"""
model the end-to-end Subaru-SCExAO optics system

Here, we will add the basic functionality of the Subaru+SCExAO Telescope, including the primary, secondary, and AO188.
The SCExAO system sits behind the AO188 instrument of Subaru, which is a 188-element AO system located at the
Nasmyth focus (IR) of the telescope.

AO188 is a basic 4f-type optical system with a 188 element circular DM. Proper will only simulate 2D square DM's, so we
use a 14x14 square inserted in the middle of the collimated beam. We also artifically oversize the number of AO188
actuators to be a 16x16 array since we oversize the DM to the beam by one actuator on either side of the beam.

This script is meant to override any Subaru/SCExAO-specific parameters specified in the user's params.py
"""

import numpy as np
from inspect import getframeinfo, stack
import proper

from medis.params import iop, sp, ap, tp
from medis.utils import dprint
import medis.optics as opx
import medis.aberrations as aber
import medis.adaptive as ao
import medis.atmosphere as atmos
import medis.coronagraphy as cg

#################################################################################################
#################################################################################################
#################################################################################################
# iop.update_testname('SCExAO-dummy-save')

# Defining Subaru parameters
# ----------------------------
# According to Iye-et.al.2004-Optical_Performance_of_Subaru:AstronSocJapan, the AO188 uses the IR-Cass secondary,
# but then feeds it to the IR Nasmyth f/13.6 focusing arrangement. So instead of simulating the full Subaru system,
# we can use the effective focal length at the Nasmyth focus, and simulate it as a single lens.
tp.d_nsmyth = 7.9716  # m pupil diameter
tp.fn_nsmyth = 13.612  # f# Nasmyth focus
tp.flen_nsmyth = tp.d_nsmyth * tp.fn_nsmyth  # m focal length
tp.dist_nsmyth_ao1 = tp.flen_nsmyth + 1.14  # m distance secondary to M1 of AO188 (hand-tuned, could update with
                                            # data from literature)

#  Below are the actual dimenstions of the Subaru telescope.
# --------------------------------
# tp.enterence_d = 8.2  # m diameter of primary
# tp.flen_primary = 15  # m focal length of primary
# tp.dist_pri_second = 12.652  # m distance primary -> secondary
# Secondary
tp.d_secondary = 1.265  # m diameter secondary, used for central obscuration
# tp.fn_secondary = 12.6

# Re-writing params terms in Subaru-units
# need this to accurately make atmospheric and aberration maps
tp.entrance_d = tp.d_nsmyth
tp.flen_primary = tp.flen_nsmyth

# ----------------------------
# AO188 DM
tp.act_woofer = 15  # approximately a 188 DM (14*14=169) but then we include +2 pixels because the dm map is oversized
                    # by 2 pixels around the edge of the array

# ----------------------------
# AO188 OAP1
# Paramaters taken from "Design of the Subaru laser guide star adaptive optics module"
#  Makoto Watanabe et. al. SPIE doi: 10.1117/12.551032
tp.d_ao1 = 0.20  # m  diamater of AO1
tp.fl_ao1 = 1.201  # m  focal length OAP1
tp.dist_ao1_dm = 1.345  # m distance OAP1 to DM

# ----------------------------
# AO188 OAP2
tp.dist_dm_ao2 = 2.511-tp.dist_ao1_dm  # m distance DM to OAP2
tp.d_ao2 = 0.2  # m  diamater of AO2
tp.fl_ao2 = 1.201  # m  focal length AO2
tp.dist_oap2_focus = 1.261

# ------------------------------
# SCExAO
# These params aren't actually working, so just doing very basic, 4F optical systems until further notice

tp.d_tweeter = 0.051  # diameter of optics in SCExAO train are 2 inches=0.051 m
tp.act_tweeter = 50  # SCExAO actuators are 50x50=2500 actuators
tp.fl_SxOAPG = 0.255  # m focal length of Genera SCExAO lens (OAP1,3,4,5)
tp.fl_SxOAP2 = 0.519  # m focal length of SCExAO OAP 2
tp.d_SxOAPG = 0.051  # diameter of SCExAO OAP's
# tp.dist_cg_sl1 = tp.fl_SxOAPG + .000001  # m distance between AO188 focus and scexao lens1

tp.dist_SxOAP1_scexao = 0.1345  # m
tp.dist_scexao_sl2 = 0.2511 - tp.dist_SxOAP1_scexao  # m
tp.dist_sl2_focus = 0.1261  # m

tp.lens_params = [{'aber_vals': [7.2e-17, 0.8, 3.1],
                   'diam': tp.entrance_d,
                   'fl': tp.flen_nsmyth,
                   'dist': tp.dist_nsmyth_ao1,
                   'name': 'effective-primary'},

                  {'aber_vals': [7.2e-17, 0.8, 3.1],
                   'diam': tp.d_ao1,
                   'fl': tp.fl_ao1,
                   'dist': tp.dist_ao1_dm,
                   'name': 'ao188-OAP1'},

                  {'aber_vals': [7.2e-17, 0.8, 3.1],
                   'diam': tp.d_ao2,
                   'fl': tp.fl_ao2,
                   'dist': tp.dist_oap2_focus,
                   'name': 'ao188-OAP2'},

                 {'aber_vals': [7.2e-17, 0.8, 3.1],
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
tp.cg_size = 2  # physical size or lambda/D size
tp.cg_size_units = "l/D"  # "m" or "l/D"
# tp.fl_cg_lens = 0.1021  # m
tp.fl_cg_lens = tp.fl_SxOAPG
tp.lyot_size = 0.9  # units are in fraction of surface blocked

#################################################################################################
#################################################################################################
#################################################################################################

def Subaru_SCExAO(empty_lamda, grid_size, PASSVALUE):
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

    # Atmosphere
    # atmos has only effect on phase delay, not intensity
    wfo.loop_collection(atmos.add_atmos, PASSVALUE['iter'], plane_name='atmosphere')

    # Defines aperture (baffle-before primary)
    wfo.loop_collection(opx.add_obscurations, d_primary=tp.entrance_d, d_secondary=tp.d_secondary, legs_frac=0.05)
    wfo.loop_collection(proper.prop_circular_aperture,
                        **{'radius': tp.entrance_d / 2})  # clear inside, dark outside
    wfo.loop_collection(proper.prop_define_entrance, plane_name='entrance_pupil')  # normalizes abs intensity

    if ap.companion:
        # Must do this after all calls to prop_define_entrance
        wfo.loop_collection(opx.offset_companion)
        wfo.loop_collection(proper.prop_circular_aperture,
                            **{'radius': tp.entrance_d / 2})  # clear inside, dark outside

    # Test Sampling
    if sp.verbose:
        wfo.loop_collection(opx.check_sampling, PASSVALUE['iter'], "Telescope Aperture",
                            getframeinfo(stack()[0][0]), units='mm')
    # Testing Primary Focus (instead of propagating to focal plane)
    # wfo.loop_collection(opx.prop_pass_lens, tp.flen_nsmyth, tp.flen_nsmyth)  # test only going to prime focus

    ########################################
    # Subaru Propagation
    #######################################
    # Effective Primary
    # CPA from Effective Primary
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='effective-primary')  # high order
    wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders))  # low order
    wfo.loop_collection(opx.prop_pass_lens, tp.flen_nsmyth, tp.dist_nsmyth_ao1)
    ########################################
    # AO188 Propagation
    ########################################
    # # AO188-OAP1
    wfo.loop_collection(aber.add_aber,  step=PASSVALUE['iter'], lens_name='ao188-OAP1')  # high order
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_ao1, tp.dist_ao1_dm)

    # AO System
    if tp.use_ao:
        WFS_map = ao.open_loop_wfs(wfo)
        wfo.loop_collection(ao.deformable_mirror, WFS_map, PASSVALUE['iter'], apodize=False,
                            plane_name='woofer', debug=sp.verbose)  # don't use PASSVALUE['WFS_map'] here because open loop
    # ------------------------------------------------
    wfo.loop_collection(proper.prop_propagate, tp.dist_dm_ao2)

    # AO188-OAP2
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='ao188-OAP2')  # high order CPA
    wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders)/2)  # low order CPA
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_ao2, tp.dist_oap2_focus)

    ########################################
    # SCExAO
    # #######################################
    # SXExAO Reimaging 1
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='SxOAPG')  # high order CPA
    wfo.loop_collection(proper.prop_propagate, tp.fl_SxOAPG)  # from AO188 focus to S-OAP1
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_SxOAPG, tp.fl_SxOAPG)  # from SxOAP1 to tweeter-DM
    #
    # AO System
    if tp.use_ao:
        WFS_map = ao.open_loop_wfs(wfo)
        wfo.loop_collection(ao.deformable_mirror, WFS_map, PASSVALUE['iter'], apodize=False,
                            plane_name='tweeter', debug=sp.verbose)
    # ------------------------------------------------
    wfo.loop_collection(proper.prop_propagate, tp.fl_SxOAPG)  # from tweeter-DM to OAP2

    # SXExAO Reimaging 2
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='SxOAP2')  # high order NCPA
    wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders)/2)  # low order NCPA
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

