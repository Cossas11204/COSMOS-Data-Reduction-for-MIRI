# Last Modified on 2023-10-13 by Cossas

import os, re

import numpy as np

import matplotlib.pyplot as plt

import math

from photutils import detect_sources, detect_threshold

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel, Box2DKernel

from matplotlib.patches import Rectangle

import datetime

def fits_reader(filename, header=False, debug=True):
    """
    Read a FITS file.

    Args:
        filename (str): The name of the FITS file.
        debug (bool, optional): Print the details for debugging. Defaults to False.
    
    Return:
        Data (dictionary): A dictionary that contains every image in the FITS file.
    """
    data = {}
    detector_number = filename.split('_')[3].split('/')[0][3:]
    if debug:
        print(f"Detector reading: {detector_number}")
    data[f'{detector_number}'] = {}
    
    with fits.open(filename) as hdul: 
        for hdu in hdul:
            extname = hdu.header.get('EXTNAME', None)
            if not extname:
                if debug:
                    print("No EXTNAME found for this HDU")
            else:
                if debug:
                    print(f"HDU extension name: {extname}")
            
            # check if the hud object is an image
            if type(hdu) == fits.hdu.image.ImageHDU:
                data[f'{detector_number}'][f'{extname}'] = hdu.data

    hdul.close()
    return data

def image_visualization(data, title=None, auto_color=True, show=True,
                        color_style='jet', save=False, scale_data=None,
                        img_dpi=150, vmin_value=20, vmax_value=97,
                        output_path=None, share_scale=False):
    """
    Function to visualize given ndarray

    Args:
        data(list): The 2D image.
        title(list of str): The titles of the ndarrays.
        auto_color(bool): If set True, use 'jet' as the colorbar, else, gray scale.
        color_style(str): Any color bar args of plt.
        save(bool): If set True, svae the plot into png file.
        output_path(str): Output path for the png files.
        share_scale(bool): If set True, share the same scale for all images.

    Returns:
        None
    """
    if not isinstance(data, list):
        data = [data]
    
    if title:
        if not isinstance(title, list):
            title = [title]

    num_images = len(data)
    
    # Determine the subplot grid dimensions
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.floor(num_images / num_rows))

    fig = plt.figure(figsize=(12,8), dpi=img_dpi)
    

    if auto_color:
        if share_scale:
            if scale_data is not None:
                vmin=np.nanpercentile(scale_data.flatten(), vmin_value)
                vmax=np.nanpercentile(scale_data.flatten(), vmax_value)
            else:
                vmin=np.nanpercentile(data[0].flatten(), vmin_value)
                vmax=np.nanpercentile(data[0].flatten(), vmax_value)
            
        for i, d in enumerate(data, start=1):
            ax = fig.add_subplot(num_rows, num_cols, i)
            d = np.copy(d)
            if not share_scale:
                vmin=np.nanpercentile(d.flatten(), vmin_value)
                vmax=np.nanpercentile(d.flatten(), vmax_value)
            im = ax.imshow(d, cmap=color_style, vmin=vmin, vmax=vmax)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            if title:
                ax.set_title(title[i-1])
                ax.title.set_size(24)

            # set the font size of the plot title
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])        
    
    if show:
        plt.show()

    if save:
        if output_path:
            print(f"Saving output png to: {output_path}")
            plt.savefig(f"{output_path}")
        else:
            print(f"Saving output png to: /mnt/C/JWST/COSMOS/NIRCAM/PNG/")
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/PNG/Output_{timestamp}.png")

def image_histogram(data, title=None, bins=np.logspace(-5, 2, 100),
                    share=False, alpha=0.8,
                    x_log=False, y_log=False):
    """
    Function to plot the histogram as the function of flux of given ndarray

    Args:
        data (numpy.array): The 2D image.
        title (list of str): Titles for each image.
        share (boolean): Set to True if you want to overplot each histogram.
        alpha (float): alpha for histogram.
        
    Returns:
        None
    """
    if not isinstance(data, list):
        data = [data]
    
    if title:
        if not isinstance(title, list):
            title = [title]

    num_images = len(data)
    
    # Determine the subplot grid dimensions
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.floor(num_images / num_rows))

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot()
    
    for i, d in enumerate(data, start=1):
        if not share:
            ax = fig.add_subplot(num_rows, num_cols, i)

        else:
            ax.hist(d.flatten(), bins=bins,
                    alpha=alpha, histtype='step', lw=1.5,
                    label=title[i-1])
            
            if x_log:
                ax.set_xscale('log')
            
            if y_log:
                ax.set_yscale('log')

        ax.legend()

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def load_fits(path):
    with fits.open(path) as hdul: 
        image = hdul[1].data
        error = hdul[2].data
        wcs = WCS(hdul[1].header)
    hdul.close()
    return image, error, wcs 

def tiered_source_detection(image, sigma_values, snr=3.5):
    """
    Source Extraction method for tired sigma values, SNR is set to 3.5 by default.
    Modified algorithm adapted from CEERS paper.
    The Gaussian kernel size is determined by the 4*sigma_values + 1.

    Args:
        image(2d array): The 2D image contain source information.
        sigma_values(list, float): sigma values threshold for detection.
        snr (float): SNR threshold for detection.
    
    Returns:
        detected_sources(2D array): Masked array containing source information.
    """
    detected_sources = np.zeros_like(image).astype(bool)
    
    for sigma in sigma_values:
        kernel = Gaussian2DKernel(x_stddev=sigma,
                                  x_size=sigma*4 + 1,
                                  y_size=sigma*4 + 1
                                 )
        # modified, reshape the sigma = 25 gaussian into the same shape as sigma = 15 gaussian
        smoothed_image = convolve(image, kernel)
        threshold = detect_threshold(smoothed_image, nsigma=snr)
        sources = detect_sources(smoothed_image, threshold, npixels=5)
        
        if sources:
            detected_sources = np.logical_or(detected_sources, sources.data.astype(bool))

    return detected_sources

def perform_fft(data):
    # Perform FFT
    fft_result = np.fft.fft(data)
    
    return fft_result

def plot_fft(data, label=None, normalized=True, 
             alpha=0.5, xlim=None, ylim=None):

    if not isinstance(data, list):
        data = [data]
    
    fig, ax = plt.subplots(figsize=(16,8))

    for d in range(len(data)):
        # Perform FFT
        result = perform_fft(data[d])
        if normalized :
            magnitude = np.abs(result)/np.max(np.abs(result))
        else:
            magnitude = np.abs(result)
        # Generate frequency bins
        freq = np.fft.fftfreq(len(result))
        # Plot the magnitude spectrum
        if label:
            ax.plot(freq, magnitude, label=label[d], alpha=alpha)
        else:
            ax.plot(freq, magnitude, alpha=alpha)

    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid(True)
    ax.legend(loc=8)
    fig.show()

def gaussian(x, mu, sigma):
    # Define the Gaussian function
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def remove_file_suffix(filename) -> str:
    """
    Remove the file suffix for a given filename.
    The suffix naming rules are:
        Uncalibrated raw input                          uncal
        Corrected ramp data                             ramp
        Corrected countrate image                       rate
        Corrected countrate per integration             rateints
        Background-subtracted image                     bsub
        Per integration background-subtracted image     bsubints
        Calibrated image                                cal
        Calibrated per integration images               calints
        CR-flagged image                                crf
        CR-flagged per integration images               crfints
        Resampled 2D image                              i2d
        Source catalog                                  cat
        Segmentation map                                segm
    """

    suffix = ['uncal', 'ramp', 'rate', 'rateints', 'bsub', 'bsubints', 'cor_wsp', 'cor_cal',
              'cal', 'calints', 'crf', 'crfints', 'i2d', 'cat', 'segm']
    
    for string in suffix:
        pattern = re.compile(rf'_{string}\.fits$')
        new_filename, n = pattern.subn('', filename)
        if n > 0:
            # print(f'Suffix found and removed: {string}')
            # print('New filename:', new_filename)
            break

    return new_filename

def mask_sources(img_data, sigma_values=[5, 2], snr=3, nan=False):
    sources = tiered_source_detection(img_data, sigma_values=sigma_values, snr=snr)
    mask = np.logical_not(sources).astype(float)
    if nan:
        mask = np.where(mask==0, np.nan, mask) 
    masked_image = np.multiply(mask, img_data)

    return masked_image

def record_and_save_data(path, fitsname, corrected_image, pedestal, suffix=""):
    if suffix in ['bkg_sub', 'bri_sub']:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cor_wsp.fits")
        
    elif suffix in ['cor']:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_rate.fits")

    elif suffix in ['cor_cal']:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cor.fits")
    
    else:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cor_cal.fits")

    if pedestal is not None:
        hdul[1].data = np.array(corrected_image-pedestal)

    else:
        hdul[1].data = np.array(corrected_image)
        
    hdul[1].header['PED_VAL'] = pedestal
    hdul.writeto(f"{path}/{remove_file_suffix(fitsname)}_{suffix}.fits", overwrite=True)
    hdul.close()

def calculate_pedestal(image):
    med = np.nanmedian(image.flatten())
    return med

def sigma_clip_replace_with_median(arr, sigma=3):
    """
    Perform sigma clipping on a 2D array, replacing clipped data with the median,
    and leaving NaN values unchanged.

    :param arr: 2D NumPy array
    :param sigma: Number of standard deviations to use for clipping
    :return: 2D NumPy array with clipped values replaced
    """
    # Making a copy of the array to avoid changing the original data
    arr_copy = np.copy(arr)
    
    # Calculating the median and standard deviation, ignoring NaN values
    median = np.nanmedian(arr_copy)
    std = np.nanstd(arr_copy)
    
    # Identifying outliers
    lower_bound = median - sigma * std
    upper_bound = median + sigma * std
    outliers = np.logical_or(arr_copy < lower_bound, arr_copy > upper_bound)
    
    # Replacing outliers with the median
    arr_copy[outliers] = median
    
    return arr_copy

def extract_miri_effective_area():
    img_data = fits_reader("/mnt/C/JWST/COSMOS/MIRI/F770W/jw01727043001_02201_00001_mirimage/jw01727043001_02201_00001_mirimage_cor_cal.fits")['image']['SCI']
    final_cor_image = np.zeros_like(img_data)
    eff_FULL = np.ones_like(img_data)

    # cutout of different parts of miri image detector
    miri_corona_lyot = img_data[745:    ,    :279]
    miri_corona_4qpm = img_data[    :682,    :232]
    final_cor_image[745:   ,    :279] = miri_corona_lyot
    final_cor_image[   :682,    :232] = miri_corona_4qpm

    # empty space between each parts of miri image detector
    miri_empty_1 = img_data[682:745,    :279]
    miri_empty_2 = img_data[   :   , 279:362]
    miri_empty_3 = img_data[   :682, 232:279]
    final_cor_image[682:745,    :279] = miri_empty_1 
    final_cor_image[   :   , 279:362] = miri_empty_2
    final_cor_image[   :682, 232:279] = miri_empty_3

    # detail cuts for miri imaging FULL detector
    final_cor_image[435:   , 362:   ] = img_data[435:   , 362:   ]
    final_cor_image[379:435, 376:   ] = img_data[379:435, 376:   ]
    final_cor_image[375:379, 388:   ] = img_data[375:379, 388:   ]
    final_cor_image[336:375, 417:   ] = img_data[336:375, 417:   ]
    final_cor_image[318:336, 388:   ] = img_data[318:336, 388:   ]
    final_cor_image[165:318, 376:   ] = img_data[165:318, 376:   ]
    final_cor_image[   :165, 362:   ] = img_data[   :165, 362:   ]

    # image_visualization([eff_FULL])
    if os.path.exists("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_nan.fits"):
        eff_FULL[   :682,    :232] = np.nan
        eff_FULL[682:745,    :279] = np.nan
        eff_FULL[   :   , 279:362] = np.nan
        eff_FULL[   :682, 232:279] = np.nan

        eff_FULL[165:435, 362:376] = np.nan
        eff_FULL[318:379, 376:388] = np.nan
        eff_FULL[336:375, 388:417] = np.nan

        # Rectangle properties
        rect_center = (145.5, 882.5)  # Center of the rectangle
        rect_width, rect_height = 18, 300  # Width and height of the rectangle
        rotation_angle = 176  # Rotation angle in degrees

        # Generate mask
        y, x = np.ogrid[:eff_FULL.shape[0], :eff_FULL.shape[1]]
        rec_mask = rotated_rect_mask(rect_center[0], rect_center[1], rect_width, rect_height, rotation_angle, x, y)

        # Parameters for the circle
        c_x, c_y = 145.5, 882.5  # Center of the circle [pix]
        r = 23                   # Radius of the circle [pix]
        
        # Calculate the distance of each element from the center
        distance_from_center = np.sqrt((x - c_x)**2 + (y - c_y)**2)
        cir_mask = distance_from_center <= r
        
        # combine the mask
        mask = rec_mask | cir_mask
        eff_FULL = eff_FULL * mask
        eff_FULL = np.where(eff_FULL == 1.0, np.nan, eff_FULL)
        
        image_visualization(eff_FULL)
        
        # Create a FITS PrimaryHDU object
        primaryhdu = fits.PrimaryHDU()

        # Create a FITS ImageHDU object from the data
        miri_eff_area_hdu = fits.ImageHDU(eff_FULL)
        miri_eff_area_hdu.header['EXTNAME'] = ('EFF_AREA', 'Effective Area of MIRI for COSMOS-Webb Field')
        miri_eff_area_hdu.header['MASK'] = ('np.nan', 'Mask Type (0.0 or np.nan)')

        # Create a FITS HDU list and save it to a file
        hdul = fits.HDUList([primaryhdu, miri_eff_area_hdu])
        hdul.writeto("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_nan.fits", overwrite=True)

    if not os.path.exists("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_zeros.fits"):
        eff_FULL[745:   ,    :279] = 1.0
        eff_FULL[   :682,    :232] = 0.0
        eff_FULL[682:745,    :279] = 0.0
        eff_FULL[   :   , 279:362] = 0.0
        eff_FULL[   :682, 232:279] = 0.0

        eff_FULL[165:435, 362:376] = 0.0
        eff_FULL[318:379, 376:388] = 0.0
        eff_FULL[336:375, 388:417] = 0.0

        # Rectangle properties
        rect_center = (145.5, 882.5)  # Center of the rectangle
        rect_width, rect_height = 18, 300  # Width and height of the rectangle
        rotation_angle = 176  # Rotation angle in degrees

        # Generate mask
        y, x = np.ogrid[:eff_FULL.shape[0], :eff_FULL.shape[1]]
        mask = rotated_rect_mask(rect_center[0], rect_center[1], rect_width, rect_height, rotation_angle, x, y)
        mask = ~mask
        eff_FULL = eff_FULL * mask.astype(float)

        # Parameters for the circle
        c_x, c_y = 145.5, 882.5  # Center of the circle [pix]
        r = 23                   # Radius of the circle [pix]
        
        # Calculate the distance of each element from the center
        distance_from_center = np.sqrt((x - c_x)**2 + (y - c_y)**2)
        mask = distance_from_center <= r
        mask = ~mask
        eff_FULL = eff_FULL * mask.astype(float)
    
        # Create a FITS PrimaryHDU object
        primaryhdu = fits.PrimaryHDU()

        # Create a FITS ImageHDU object from the data
        miri_eff_area_hdu = fits.ImageHDU(eff_FULL)
        miri_eff_area_hdu.header['EXTNAME'] = ('EFF_AREA', 'Effective Area of MIRI for COSMOS-Webb Field')
        miri_eff_area_hdu.header['MASK'] = ('0.0', 'Mask Type (0.0 or np.nan)')

        # Create a FITS HDU list and save it to a file
        hdul = fits.HDUList([primaryhdu, miri_eff_area_hdu])
        hdul.writeto("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_zeros.fits", overwrite=True)

# Function to check if a point is inside a rotated rectangle
def rotated_rect_mask(cx, cy, width, height, angle, x, y):
    # Translate point to origin
    x_translated = x - float(cx)
    y_translated = y - float(cy)
    
    # Rotate point
    angle = math.radians(angle)  # Convert to radians
    x_rot = x_translated * math.cos(angle) + y_translated * math.sin(angle)
    y_rot = -x_translated * math.sin(angle) + y_translated * math.cos(angle)
    
    # Check if point is inside rectangle
    mask = (np.abs(x_rot) <= width / 2) & (np.abs(y_rot) <= height / 2)
    
    return mask

def multiply_by_miri_effective_area(img_data, nan=True):
    if nan:
        eff_area = fits.open("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_nan.fits")[1].data

    else:
        eff_area = fits.open("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area_zeros.fits")[1].data

    return np.multiply(eff_area, img_data)

def convert_MJYSR_to_JYPIX(input_data):
    convert_coefficient = 10**6 * 10**-14 
    return input_data * convert_coefficient

def genPSF(filters):
    import webbpsf

    if filters in ["F560W", "F770W", "F1000W", "F1280W", "F1500W", "F1800W", "F2100W"]:
        instrument = webbpsf.MIRI()
        case = "MIRI"
    else:
        instrument = webbpsf.NIRCam()
        case = "NIRCAM"
    
    for band in filters:
        if not os.path.exists(f"/mnt/c/JWST/COSMOS/{case}/{band}_psf.fits"):
            instrument.filter = band
            instrument.calc_psf(oversample=1, fov_arcsec=4).writeto(
                f"{band}_psf.fits", overwrite=True
            )
