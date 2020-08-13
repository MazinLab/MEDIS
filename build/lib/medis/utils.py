import numpy as np
from inspect import getframeinfo, stack
import pickle
import tables as pt
import astropy.io.fits as afits

from medis.params import sp, ap, tp, iop


def dprint(*message, path_display=-3):
    """
    prints location of code where message is printed from

    >>> dprint('foo', 5000, (), np.arange(9).reshape(3,3))
    MEDIS++/medis/optics.py:173 - lol, 5000, (), [[0 1 2]
    [3 4 5]
    [6 7 8]]

    path_to_display : integer number of folders back from the module location to display in printed statement
    """
    caller = getframeinfo(stack()[1][0])
    message_str = ''
    for mess in message:
        message_str += f'{mess}, '
    message_str = message_str[:-2]

    reduced_filename = '/'.join(caller.filename.split('/')[path_display:])
    print("%s:%d - %s" % (reduced_filename, caller.lineno, message_str))


def phase_cal(wavelengths):
    """Wavelength in nm"""
    phase = tp.wavecal_coeffs[0] * wavelengths + tp.wavecal_coeffs[1]
    return phase


####################################################################################################
# Functions Relating to Reading, Loading, and Saving Data #
####################################################################################################
def save_to_disk_sequence(obs_sequence, obs_seq_file='obs_seq.pkl'):
    """saves obs sequence as a .pkl file

    :param obs_sequence- Observation sequence, 6D data structure
    :param obs_seq_file- filename for saving, including directory tree
    """
    #dprint((obs_seq_file, obs_seq_file[-3:], obs_seq_file[-3:] == '.h5'))
    if obs_seq_file[-3:] == 'pkl':
        with open(obs_seq_file, 'wb') as handle:
            pickle.dump(obs_sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif obs_seq_file[-3:] == 'hdf' or obs_seq_file[-3:] == '.h5':
        f = pt.open_file(obs_seq_file, 'w')
        ds = f.create_array(f.root, 'data', obs_sequence)
        f.close()
    else:
        dprint('Extension not recognised')


def check_exists_obs_sequence(plot=False):
    """
    This code checks to see if there is already
    an observation sequence saved with the output of the run in the
    location specified by the iop.

    :return: boolean flag if it can find a file or not
    """
    import os
    if os.path.isfile(iop.obs_seq):
        dprint(f"File already exists at {iop.obs_seq}")
        return True
    else:
        return False


def open_obs_sequence(obs_seq_file='hyper.pkl'):
    """opens existing obs sequence .pkl file and returns it"""
    with open(obs_seq_file, 'rb') as handle:
        obs_sequence =pickle.load(handle)
    return obs_sequence


def open_obs_sequence_hdf5(obs_seq_file='hyper.h5'):
    """opens existing obs sequence .h5 file and returns it"""
    # hdf5_path = "my_data.hdf5"
    read_hdf5_file = pt.open_file(obs_seq_file, mode='r')
    # Here we slice [:] all the data back into memory, then operate on it
    obs_sequence = read_hdf5_file.root.data[:]
    # hdf5_clusters = read_hdf5_file.root.clusters[:]
    read_hdf5_file.close()
    return obs_sequence


####################################################################################################
# Functions Relating to Reading, Loading, and Saving Images #
####################################################################################################
def saveFITS(image, name='test.fit'):
    header = afits.Header()
    header["PIXSIZE"] = (0.16, " spacing in meters")

    hdu = afits.PrimaryHDU(image, header=header)
    hdu.writeto(name)


def readFITS(filename):
    """
    reads a fits file and returns data fields only

    :param filename: must specify full filepath
    """
    hdulist = afits.open(filename)
    header = hdulist[0].header
    scidata = hdulist[0].data

    return scidata


def clipped_zoom(img, zoom_factor, **kwargs):
    from scipy.ndimage import zoom
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        from medis.Utils.plot_tools import quicklook_im
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # quicklook_im(out, logZ=True)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        # print top, zh, left, zw
        # print out.shape[0], trim_top, h, trim_left, w
        if trim_top < 0 or trim_left < 0:
            temp = np.zeros_like(img)
            temp[:out.shape[0],:out.shape[1]] = out
            out = temp
        else:
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]
        # quicklook_im(out, logZ=False)
    # If zoom_factor == 1, just return the input array
    else:
        out = img

    # import matplotlib.pyplot as plt
    # plt.hist(out.flatten(), bins =100, alpha =0.5)
    # plt.hist(img.flatten(), bins =100, alpha=0.5)
    # plt.show()

    # print(np.sum(img), np.sum(out))
    # out = out*np.sum(img)/np.sum(out)
    # out = out*4
    return out
