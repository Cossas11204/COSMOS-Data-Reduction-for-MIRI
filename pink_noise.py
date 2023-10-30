# Last Modified on 2023-08-15 by Cossas

import astropy.io.fits as fits
from astropy.stats import sigma_clip

import numpy as np

from utility import image_visualization, remove_file_suffix, calculate_pedestal

# @dace
def mask_bad_pixels(DQ_map, path, detected_sources, image_data, visualize=False):
    """
    Mask the bad pixels in a given NIR image with provided sources and Data Quality Map.
    Accelerated/Parallelized with DACE module.
    
    Args:
        DQ_map (np.array): Data Quality Map.
        path (str): Path to the NIR image.
        detected_sources (2D array): Sources found in the image.
        image_data (np.array): NIR image that needs to be masked.
        
    Returns:
        masked_image (np.array): Masked NIR image.
    """
    
    if visualize:
        bad_pixels_mask = np.where(DQ_map > 0, 0, 1)
        image_visualization(bad_pixels_mask, save=True, output_path=f"{path}/Pixel_mask.png")
        mask = np.logical_and(bad_pixels_mask, np.logical_not(detected_sources)).astype(float)
        image_visualization(mask, save=True, output_path=f"{path}/mask.png")
        masked_image = np.multiply(mask, image_data)
        image_visualization(masked_image, save=True, output_path=f"{path}/masked_image_1.png")
        masked_image = np.where(masked_image==0, np.nan, masked_image)
        image_visualization(masked_image, save=True, output_path=f"{path}/masked_image_2.png")
        
    else:
        bad_pixels_mask = np.where(DQ_map > 0, 0, 1)
        mask = np.logical_and(bad_pixels_mask, np.logical_not(detected_sources)).astype(float)
        masked_image = np.multiply(mask, image_data)
        masked_image = np.where(masked_image==0, np.nan, masked_image)
        
    return masked_image

def remove_striping(masked_image, real_image, sigma=2, axis=1, SWC=True, instrument=None):
    """
    SWC (Short Wavelengths Channel)
    """
    sectors = []
    x = real_image.shape[0]
    y = real_image.shape[1]

    if SWC:
        # divide the horizontal striping pattern into 4 groups of amplifiers
        for rows in masked_image:
            sectors.append([rows[0:int(np.floor(x*1/4))],
                        rows[int(np.floor(x*1/4)):int(np.floor(x*2/4))],
                        rows[int(np.floor(x*2/4)):int(np.floor(x*3/4))], 
                        rows[int(np.floor(x*3/4)):int(np.floor(x))]]
                        )
        
        for i, sector in enumerate(sectors):
            for j, patch in enumerate(sector):
                # first clip bright pixels then 2-sigma clip
                cliped_patch = sigma_clip(patch, sigma=2)

                # Check if the length of patch is less than 52
                if len(cliped_patch) > 52:
                    # Replace the original patch with the sigma clipped patch
                    sectors[i][j] = list(cliped_patch)
                else:
                    sectors[i][j] = list(patch)

        striping_pattern = np.nanmedian(sectors, axis=2)
        reconstructed_striping_pattern = np.zeros(masked_image.shape)

        # Reconstruct the striping pattern
        for i in range(4):
            reconstructed_striping_pattern[:, i*512:(i+1)*512] = striping_pattern[:, i][:, np.newaxis]

        # Remove the striping pattern from the input image
        corrected_image = real_image - reconstructed_striping_pattern + calculate_pedestal(masked_image)

    else:
        # Calculate the median along the specified axis using sigma clipping
        striping_pattern = np.nanmedian(masked_image, axis=1)

        # Reshape the striping pattern into a 2D array with shape (1, 2048)
        striping_pattern_reshaped = striping_pattern.reshape(1, masked_image.shape[0])

        # Create the striping pattern image by repeating the reshaped striping pattern along the x-axis
        striping_pattern_image = np.repeat(striping_pattern_reshaped, real_image.shape[1], axis=0).T
        
        # Remove the striping pattern from the input image
        corrected_image = real_image - striping_pattern_image + calculate_pedestal(masked_image)

    if instrument == 'lyot':
        # Calculate the median along the specified axis using sigma clipping
        striping_pattern = np.nanmedian(corrected_image, axis=0)

        # Reshape the striping pattern into a 2D array with shape (2048, 1)
        striping_pattern_reshaped = striping_pattern.reshape(masked_image.shape[1], 1)

        # Create the striping pattern image by repeating the reshaped striping pattern along the y-axis
        striping_pattern_image = np.repeat(striping_pattern_reshaped, real_image.shape[0], axis=1).T

        # Remove the striping pattern from the input image
        corrected_image = corrected_image - striping_pattern_image + calculate_pedestal(masked_image)

    elif instrument == '4qpm':
        # Calculate the median along the specified axis using sigma clipping
        striping_pattern = np.nanmedian(corrected_image, axis=0)

        # Reshape the striping pattern into a 2D array with shape (2048, 1)
        striping_pattern_reshaped = striping_pattern.reshape(masked_image.shape[1], 1)

        # Create the striping pattern image by repeating the reshaped striping pattern along the y-axis
        striping_pattern_image = np.repeat(striping_pattern_reshaped, real_image.shape[0], axis=1).T

        # Remove the striping pattern from the input image
        corrected_image = corrected_image - striping_pattern_image + calculate_pedestal(masked_image)
        
    else:
        # Calculate the median along the specified axis using sigma clipping
        striping_pattern = np.nanmedian(corrected_image, axis=0)

        # Reshape the striping pattern into a 2D array with shape (2048, 1)
        striping_pattern_reshaped = striping_pattern.reshape(masked_image.shape[1], 1)

        # Create the striping pattern image by repeating the reshaped striping pattern along the y-axis
        striping_pattern_image = np.repeat(striping_pattern_reshaped, real_image.shape[0], axis=1).T

        # Remove the striping pattern from the input image
        corrected_image = corrected_image - striping_pattern_image + calculate_pedestal(masked_image)

    return corrected_image
