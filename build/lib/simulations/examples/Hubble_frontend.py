"""
model the Hubble optics system

This is a code modified from Rupert's original optics_propagate.py.

This script is meant to override any Hubble-specific parameters specified in the user's params.py
"""

import numpy as np
import proper
from scipy.interpolate import interp1d

from medis.params import iop, ap, tp, sp
from medis.utils import dprint
import medis.optics as opx
import medis.aberrations as aber
import medis.adaptive as ao
import medis.atmosphere as atmos

#################################################################################################
#################################################################################################
#################################################################################################
#iop.update_testname(Hubble-basic-test1')

# Defining Hubble parameters
# ----------------------------
tp.entrance_d = 2.4  # m
tp.flen_primary = 5.52085  # m
tp.dist_pri_second = 4.907028205  # m distance primary -> secondary

# --------------------------------
# Secondary
tp.d_secondary = 0.396  # m diameter secondary, used for central obscuration
tp.flen_secondary = -0.6790325  # m focal length of secondary
tp.dist_second_focus = 6.3919974  # m distance secondary to M1 of AO188

# --------------------------------
# Wavelength
ap.wvl_range = np.array([450, 580]) / 1e9
ap.n_wvl_init = 3
ap.n_wvl_final = 6
sp.subplt_cols = 3

tp.obscure = True
tp.use_ao = False

#################################################################################################
#################################################################################################
#################################################################################################

def Hubble_frontend(empty_lamda, grid_size, PASSVALUE):
    """
    propagates instantaneous complex E-field tSubaru from the primary through the AO188
        AO system in loop over wavelength range

    this function is called a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere,
        then telescope, to the focal plane
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time, sampling of the wavefront at final location

    """
    # print("Propagating Broadband Wavefront Through Hubble Telescope")

    # Getting Parameters-import statements weren't working--RD
    passpara = PASSVALUE['params']
    ap.__dict__ = passpara[0].__dict__
    tp.__dict__ = passpara[1].__dict__
    iop.__dict__ = passpara[2].__dict__
    sp.__dict__ = passpara[3].__dict__

    datacube = []

    # Initialize the Wavefront in Proper
    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    ########################################
    # Hubble Propagation
    #######################################
    # Defines aperture (baffle-before primary)
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': tp.entrance_d / 2})  # clear inside, dark outside
    # Obscurations
    wfo.loop_collection(opx.add_obscurations, d_primary=tp.entrance_d, d_secondary=tp.d_secondary, legs_frac=0.01)
    wfo.loop_collection(proper.prop_define_entrance)  # normalizes the intensity

    # Test Sampling
    if PASSVALUE['iter'] == 1:
        initial_sampling = proper.prop_get_sampling(wfo.wf_collection[0,0])
        dprint(f"initial sampling is {initial_sampling:.4f}")

    # Primary
    # CPA from Effective Primary
    # aber.add_aber(wfo.wf_collection, tp.entrance_d, step=PASSVALUE['iter'], lens_name='primary')

    # wfo.loop_collection(opx.prop_pass_lens, tp.flen_primary, tp.flen_primary)
    wfo.loop_collection(opx.prop_pass_lens, tp.flen_primary, tp.dist_pri_second)

    # Secondary
    # aber.add_aber(wfo.wf_collection, tp.d_secondary, step=PASSVALUE['iter'], lens_name='second')
    # # Zernike Aberrations- Low Order
    # wfo.loop_collection(aber.add_zern_ab, tp.zernike_orders, tp.zernike_vals)
    wfo.loop_collection(opx.prop_pass_lens, tp.flen_secondary, tp.dist_second_focus)

    ########################################
    # Focal Plane
    # #######################################
    # Converting Array of Arrays (wfo) into 3D array
    #  wavefront array is now (number_wavelengths x number_astro_bodies x sp.grid_size x sp.grid_size)
    #  prop_end moves center of the wavefront from lower left corner (Fourier space) back to the center
    #    ^      also takes square modulus of complex values, so gives units as intensity not field
    shape = wfo.wf_collection.shape
    for iw in range(shape[0]):
        for io in range(shape[1]):
            if sp.maskd_size != sp.grid_size:
                wframes = np.zeros((sp.maskd_size, sp.maskd_size))
                (wframe, sampling) = proper.prop_end(wfo.wf_collection[iw, io], EXTRACT=np.int(sp.maskd_size))
            else:
                wframes = np.zeros((sp.grid_size, sp.grid_size))
                (wframe, sampling) = proper.prop_end(wfo.wf_collection[iw, io])  # Sampling returned by proper is in m
            wframes += wframe  # adds 2D wavefront from all astro_bodies together into single wavefront, per wavelength
        # dprint(f"sampling in focal plane at wavelength={iw} is {sampling} m")
        datacube.append(wframes)  # puts each wavlength's wavefront into an array
                                  # (number_wavelengths x sp.grid_size x sp.grid_size)

    datacube = np.array(datacube)
    datacube = np.roll(np.roll(datacube, tp.pix_shift[0], 1), tp.pix_shift[1],
                       2)  # cirshift array for off-axis observing
    datacube = np.abs(datacube)  # get intensity from datacube

    # Interpolating spectral cube from ap.n_wvl_init discreet wavelengths to ap.n_wvl_final
    if ap.interp_wvl and ap.n_wvl_init > 1 and ap.n_wvl_init < ap.n_wvl_final:
        wave_samps = np.linspace(0, 1, ap.n_wvl_init)
        f_out = interp1d(wave_samps, datacube, axis=0)
        new_heights = np.linspace(0, 1, ap.n_wvl_final)
        datacube = f_out(new_heights)

    print('Finished datacube at single timestep')

    return datacube, sampling

