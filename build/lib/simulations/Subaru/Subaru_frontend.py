"""
model the Subaru optics system

This is a code modified from Rupert's original optics_propagate.py. This code adds more optics to the system,
as well as puts the AO, etc in order for Subaru.

Here, we will add the basic functionality of the Subaru Telescope, including the primary, secondary, and AO188.
The SCExAO system sits behind the AO188 instrument of Subaru, which is a 188-element AO system located at the
Nasmyth focus (IR) of the telescope. AO188 is a basic 4f-type optical system with a 188 element DM (not perfectly
represented here. Proper will only simulate 2D square DM's, so we use a 14x14 square) inserted in the middle of the
collimated beam. This routine simulates an idealized detector at the A0188 focus.

If you want to scale the intensity of the companion, you can't then normalize the intensity
(prop_define_entrance) since the normalization is per wavefront, so the companion will normalize to itself
rather than to the star. Doing it this way may cause confusion, but better than re-writing things in proper

AO188 uses laser guide-star technology. More info here:
https://subarutelescope.org/Introduction/instrument/AO188.html

A more detailed model of SCExAO will be modelled in a SCExAO_optics.py code. However, this routine is designed for
simple simulations that need to optimize runtime but still have relevance to the Subaru Telescope.

This script is meant to override any Subaru/SCExAO-specific parameters specified in the user's params.py
"""

import numpy as np
from inspect import getframeinfo, stack
import proper

from medis.params import ap, tp, sp
from medis.utils import dprint
import medis.optics as opx
import medis.aberrations as aber
import medis.adaptive as ao
import medis.atmosphere as atmos


#################################################################################################
#################################################################################################
#################################################################################################
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

# Effective Primary Aberrations
# primary_aber_vals = {'a': [7.2e-17, 3e-17],  # power at low spatial frequencies (m4)
#                      'b': [0.8, 0.2],  # correlation length (b/2pi defines knee)
#                      'c': [3.1, 0.5],  #
#                      'a_amp': [0.05, 0.01]}
# ----------------------------
# AO188 OAP1
# Paramaters taken from "Design of the Subaru laser guide star adaptive optics module"
#  Makoto Watanabe et. al. SPIE doi: 10.1117/12.551032
tp.d_ao1 = 0.20  # m  diamater of AO1
tp.fl_ao1 = 1.201  # m  focal length OAP1
tp.dist_ao1_dm = 1.345  # m distance OAP1 to DM
# OAP1_aber_vals = {'a': [7.2e-17, 3e-17],  # power at low spatial frequencies (m4)
#                   'b': [0.8, 0.2],  # correlation length (b/2pi defines knee)
#                   'c': [3.1, 0.5],  #
#                   'a_amp': [0.05, 0.01]}

# ----------------------------
# AO188 OAP2
tp.dist_dm_ao2 = 2.511-tp.dist_ao1_dm  # m distance DM to OAP2
tp.d_ao2 = 0.2  # m  diamater of AO2
tp.fl_ao2 = 1.201  # m  focal length AO2
tp.dist_oap2_focus = 1.261
# OAP2_aber_vals = {'a': [7.2e-17, 3e-17],  # power at low spatial frequencies (m4)
#                   'b': [0.8, 0.2],  # correlation length (b/2pi defines knee)
#                   'c': [3.1, 0.5],  #
#                   'a_amp': [0.05, 0.01]}

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
                   'name': 'ao188-OAP2'}
                    ]
#################################################################################################
#################################################################################################
#################################################################################################

def Subaru_frontend(empty_lamda, grid_size, PASSVALUE):
    """
    propagates instantaneous complex E-field thru Subaru from the primary through the AO188
        AO system in loop over wavelength range

    this function is called a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere,
        then telescope, to the focal plane
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time in the focal_plane()
    """
    # print("Propagating Broadband Wavefront Through Subaru")

    # Initialize the Wavefront in Proper
    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    # Atmosphere
    # atmos has only effect on phase delay, not intensity
    wfo.loop_collection(atmos.add_atmos, PASSVALUE['iter'], plane_name='atmosphere')

    # Defines aperture (baffle-before primary)
    # Obscurations (Secondary and Spiders)
    wfo.loop_collection(opx.add_obscurations, d_primary=tp.d_nsmyth, d_secondary=tp.d_secondary, legs_frac=0.05)
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
        opx.check_sampling(PASSVALUE['iter'], wfo, "initial", getframeinfo(stack()[0][0]), units='mm')
    # Testing Primary Focus (instead of propagating to focal plane)
    # wfo.loop_collection(opx.prop_pass_lens, tp.flen_nsmyth, tp.flen_nsmyth)  # test only going to prime focus

    ########################################
    # Subaru Propagation
    #######################################
    # Effective Primary
    # CPA from Effective Primary
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='ao188-OAP1')
    # Zernike Aberrations- Low Order
    # wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders))
    wfo.loop_collection(opx.prop_pass_lens, tp.flen_nsmyth, tp.dist_nsmyth_ao1)

    ########################################
    # AO188 Propagation
    ########################################
    # # AO188-OAP1
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='ao188-OAP1')
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_ao1, tp.dist_ao1_dm)

    # AO System
    if tp.use_ao:
        WFS_map = ao.open_loop_wfs(wfo)
        wfo.loop_collection(ao.deformable_mirror, WFS_map, PASSVALUE['iter'], plane_name='woofer',
                            debug=sp.debug)  # don't use PASSVALUE['WFS_map'] here because open loop
    # ------------------------------------------------
    wfo.loop_collection(proper.prop_propagate, tp.dist_dm_ao2)

    # AO188-OAP2
    wfo.loop_collection(aber.add_aber, step=PASSVALUE['iter'], lens_name='ao188-OAP2')
    # wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders)/2)
    wfo.loop_collection(opx.prop_pass_lens, tp.fl_ao2, tp.dist_oap2_focus)

    ########################################
    # Focal Plane
    # #######################################
    # Check Sampling in focal plane
    if sp.verbose:
        wfo.loop_collection(opx.check_sampling, PASSVALUE['iter'], "focal plane",
                            getframeinfo(stack()[0][0]), units='nm')

    # wfo.focal_plane fft-shifts wfo from Fourier Space (origin==lower left corner) to object space (origin==center)
    cpx_planes, sampling = wfo.focal_plane()

    print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    return cpx_planes, sampling

