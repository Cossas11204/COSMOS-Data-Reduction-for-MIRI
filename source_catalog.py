# Last Modified on 2023-08-15 by Cossas

from photutils import detect_sources, detect_threshold
from photutils.segmentation import deblend_sources, SourceCatalog

from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

import numpy as np

from utility import load_fits, image_visualization

def make_photutil_catalog(img_path, save_cat=r'destination', autobkg=True, npixel=5, SNR=1.5, kernel_sigma=[9]):
    
    """
    Modified algorithm for detecting sources in NIRCam images.
    The Gaussian kernel size is determined by the 4*sigma_values + 1.

    Args:
        img_path(str): The path of the 2D image that needs source extraction.
        save_cat(str): The path of the output source catalog, if set as ``None``, no output catalog will be saved.
        autobkg(bool): Whether to subtract the background from the input image.
        npixel(int): The least number of pixels to be identified as one source.
        SNR(float): The least signal-to-noise ratio of detected sources.

    Returns:
        detected_sources(pd.DataFrame): Source catalog of input image.
    """
    image, error, wcs = load_fits(path=img_path)

    if autobkg:
        from photutils.background import Background2D, MedianBackground
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (50, 50), filter_size=(3, 3),
                        bkg_estimator=bkg_estimator)
        image -= bkg.background  # subtract the background

    for sigma in kernel_sigma:
        kernel = Gaussian2DKernel(x_stddev=sigma,
                                  x_size=sigma*4 + 1,
                                  y_size=sigma*4 + 1
                                 )
        
        detection_image = bilinear_interpolation(np.copy(image))
        image_visualization(detection_image)
    
        smoothed_image = convolve(image, kernel)

        threshold = detect_threshold(smoothed_image, nsigma=SNR)

        sources = detect_sources(
                                 smoothed_image, threshold, 
                                 npixels=npixel, connectivity=4
                                 )
        
        segm_deblend = deblend_sources(
                                       smoothed_image, sources,
                                       npixels=npixel, nlevels=32, contrast=0.001,
                                       progress_bar=False
                                       )
        
        cat = SourceCatalog(
                            image, segm_deblend, convolved_data=smoothed_image, 
                            wcs=wcs, error=error, 
                            kron_params=[1.2, 2.5],
                            )
        cat_sc = cat
        cat_qt = cat.to_table()

    if save_cat:
        if save_cat.endswith('.csv'):
            cat_df = cat_qt.to_pandas()
            cat_df.to_csv(rf"{save_cat}", index=False)
        else:
            raise ValueError("Make sure your file destination is a path of csv file")
    else:
        cat_df = cat_qt.to_pandas()
    
    return cat_sc, cat_df

def source_cutout_imgs(catalog, save_dir, length=f'ALL'):
    """
    Make cutout arrays for each source.
    The cutout for each source will be centered at its centroid position. 

    Args:
        catalog(photutils.segmentation.SourceCatalog): Catalog of your interest.
        save_dir(str): The path of the output images directory.
        length(int, default=str): The number of the cutout images. Default is 'ALL'.

    Returns:
        cutouts: list of CutoutImage. Defalt shape is 91x91 pixels across.
    """
    cutouts = catalog.make_cutouts(shape=(91, 91), mode='partial', fill_value=np.nan)

    if not isinstance(cutouts, list):
        cutouts = [cutouts]

    if length == 'ALL':
        for i in cutouts:
            image_visualization(i.data, title=f'{i.xyorigin[0]}_{i.xyorigin[1]}', 
                                save=True,
                                output_path=f'{save_dir}/{i.xyorigin[0]}_{i.xyorigin[1]}.png'
                                )
    else:
        for i in cutouts[:length]:
            image_visualization(i.data, title=f'{i.xyorigin[0]}_{i.xyorigin[1]}', 
                                save=True,
                                output_path=f'{save_dir}/{i.xyorigin[0]}_{i.xyorigin[1]}.png'
                                )


    return cutouts

def bilinear_interpolation(image_with_nans):
    from scipy.interpolate import griddata
    # Creating a mask of the NaN values
    nan_mask = np.isnan(image_with_nans)

    # Creating coordinate arrays for the input array
    x, y = np.indices(image_with_nans.shape)

    # Flattening the arrays and removing NaN values
    x = x[~nan_mask]
    y = y[~nan_mask]
    values = image_with_nans[~nan_mask]

    # Creating coordinate arrays for the output interpolated array
    xi, yi = np.indices(image_with_nans.shape)

    # Performing the bilinear interpolation
    interpolated_values = griddata((x, y), values, (xi, yi), method='linear')

    # Replacing the original NaN values with the interpolated values
    result = np.where(nan_mask, interpolated_values, image_with_nans)

    return result
