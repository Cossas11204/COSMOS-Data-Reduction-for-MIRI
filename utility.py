# Last Modified on 2023-10-13 by Cossas

import os, re

import numpy as np

import matplotlib.pyplot as plt

from photutils import detect_sources, detect_threshold

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

import datetime

def fits_reader(filename, header=False, debug=False):
    """
    Read a FITS file.

    Args:
        filename (str): The name of the FITS file.
        index (int, optional): The index of the FITS file. Defaults to 1.
    
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

def image_visualization(data, title=None, auto_color=True, 
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
    
    for i, d in enumerate(data, start=1):
        if not share:
            ax = fig.add_subplot(num_rows, num_cols, i)

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

    suffix = ['uncal', 'ramp', 'rate', 'rateints', 'bsub', 'bsubints',
              'cal', 'calints', 'crf', 'crfints', 'i2d', 'cat', 'segm']
    
    for string in suffix:
        result = re.sub(rf'_{string}.fits', '', filename)
        if result:
            break

    return result

def mask_sources(img_data, sigma_values=[5, 2], snr=3):
    sources = tiered_source_detection(img_data, sigma_values=sigma_values, snr=snr)
    mask = np.logical_not(sources).astype(float)
    masked_image = np.multiply(mask, img_data)

    return masked_image

def record_and_save_data(path, fitsname, corrected_image, pedestal, suffix='cal'):
    hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cal.fits")
    hdul[1].data = np.array(corrected_image-pedestal)
    hdul[1].header['PED_VAL'] = pedestal
    hdul.writeto(f"{path}/{remove_file_suffix(fitsname)}_cor_{suffix}.fits", overwrite=True)
    hdul.close()

def calculate_pedestal(image):
    # unmasked_values = image[np.logical_not(np.isnan(image))]
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

def extract_miri_effective_area(img_data):
    final_cor_image = np.zeros_like(img_data)
    eff_FULL = np.ones_like(img_data)

    # cutout of different parts of miri image detector
    miri_corona_lyot = img_data[745:    ,    :279]
    miri_corona_4qpm = img_data[    :682,    :232]
    final_cor_image[745:   ,    :279] = miri_corona_lyot
    final_cor_image[   :682,    :232] = miri_corona_4qpm

    # empty space between each parts of miri image detector
    miri_empty_1 = img_data[682:745,    :279]
    miri_empty_2 = img_data[   :   , 279:355]
    miri_empty_3 = img_data[   :682, 232:279]
    final_cor_image[682:745,    :279] = miri_empty_1 
    final_cor_image[   :   , 279:355] = miri_empty_2
    final_cor_image[   :682, 232:279] = miri_empty_3

    # detail cuts for miri imaging FULL detector
    final_cor_image[435:   , 355:   ] = img_data[435:   , 355:   ]
    final_cor_image[379:435, 376:   ] = img_data[379:435, 376:   ]
    final_cor_image[375:379, 388:   ] = img_data[375:379, 388:   ]
    final_cor_image[336:375, 417:   ] = img_data[336:375, 417:   ]
    final_cor_image[318:336, 388:   ] = img_data[318:336, 388:   ]
    final_cor_image[165:318, 376:   ] = img_data[165:318, 376:   ]
    final_cor_image[   :165, 355:   ] = img_data[   :165, 355:   ]

    eff_FULL[745:   ,    :279] = np.nan
    eff_FULL[   :682,    :232] = np.nan
    eff_FULL[682:745,    :279] = np.nan
    eff_FULL[   :   , 279:355] = np.nan
    eff_FULL[   :682, 232:279] = np.nan

    eff_FULL[165:435, 355:376] = np.nan
    eff_FULL[318:379, 376:388] = np.nan
    eff_FULL[336:375, 388:417] = np.nan

    image_visualization([eff_FULL])

    return eff_FULL

def multiply_by_miri_effective_area(img_data):
    eff_area = fits.open("/mnt/C/JWST/COSMOS/MIRI/MIRI_eff_area.fits")[1].data
    return np.multiply(img_data, eff_area)