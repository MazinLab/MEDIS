"""
check the params entered to params for given telescope settings for correct sampling
KD 8/13/19

According to the Proper manual:
    sp.beam_ratio = tp.entrance_d/ init_grid_width
    where init_grid_width is the initial grid width (diameter) in meters
Thus
    init_grid_width = tp.enterence_d / sp.beam_ratio

The sampling of the beam is defined by
    sampling = grid_width / sp.grid_size
    :keyword
    sampling = tp.enterence_d * sp.grid_size / sp.beam_ratio
    where grid_width here is the initial grid width

"""
import numpy as np
from medis.params import tp, sp

def check_sampling():
    #######################################
    # Definitions from inside mini-medis
    #######################################
    # Basics-not imported
    # tp.entrance_d = 0.1  # [m]
    # sp.beam_ratio = .3  # [unitless]
    # sp.grid_size = 512  # [pix]

    # From Params Buried in mini-medis
    legs_frac = 0.1  # [m]

    #######################
    # Initial Sampling
    ######################
    init_grid_width = tp.entrance_d / sp.beam_ratio  # [m]
    init_sampling = init_grid_width / sp.grid_size  # [m/pix]
    init_npix_on_beam = tp.entrance_d / init_sampling  # [pix]
    print(f"Initial Grid Width = {init_grid_width:.2f} m")
    print(f"Initial number of pixels across beam = {sp.beam_ratio*sp.grid_size}")

    #################################
    # Checking Sampling of Spiders
    ################################
    spider_width = tp.entrance_d * legs_frac  # [m]
    npix_on_spiders = spider_width / init_sampling  # [pix]
    print(f"Number of pixels across the spiders = {npix_on_spiders}")
    print("********************************")

###################
# Ideal Sampling
###################
def ideal_sampling():
    """
    see proper manual pg 36 for definitions

    init_grid_width is dependant on chosen/user-defined beam ratio. In order to determine what a "good" one would be,
    first choose a grid_width that is significantly larger than the entrance diameter, such that the beam is only
    resolved by a few pixels. This ensures there is sufficient zero-padding of the pupil to avoid FFT effects, and you
    can check the sampling of the spiders with the code above. There is a trade-off between sampling in the pupil and
    the image planes, so it is best to only choose a beam ratio that barely resolves the smallest features of the aperture
    (spiders), in order to get the best resolution of the image plane.
    """
    # Hard-Coded Parameters
    Fnum = 12.6  # focal ratio of the beam as it approaches the focus
    lmbda = 800e-9  # [m]  wavelength of interest
    grid_width = tp.entrance_d/sp.beam_ratio  # [m]

    # Sampling
    samp_good = Fnum*lmbda*tp.entrance_d / grid_width
    samp_nyq = Fnum*lmbda/2
    print(f"In the pupil plane at lambda={lmbda*1e6:.3f} um:\n"
          f" Ideal Sampling is {samp_good:.6f} m/pix \n"
          f"Nyquist Sampling is {samp_nyq:.6f} m/pix ")

    # selecting beam ratio
    br_good = tp.entrance_d/(samp_good*sp.grid_size)
    br_nyq = tp.entrance_d/samp_nyq/sp.grid_size
    print(f"Good starter beam ratio is {br_good} \n"
          f"Nyquist beam ratio is {br_nyq}")

    ###########################################
    # Beam Size and Grid Width at other Plane
    ###########################################
    final_sampling = 0.212727  # m/pix
    final_grid_width = final_sampling * sp.grid_size  # m
    final_beam_size = final_grid_width * sp.beam_ratio  # m
    final_npix_on_beam = final_beam_size / final_sampling  # pixels


if __name__ == '__main__':
    check_sampling()
