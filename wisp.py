import os, glob, json

import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import sigma_clip

import astropy.io.fits as fits

from utility import *

class Wisp():
    def __init__(self, filter, detector, debug=False):
        # Create the basic structure for storing the discarded frames for each filter & detector.
        self.discarded_slices = {}
        self.discarded_slices['NIRCAM'] = {}

        self.discarded_slices['NIRCAM']['F115W'] = {}
        self.discarded_slices['NIRCAM']['F150W'] = {}

        for d in ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4']:
            self.discarded_slices['NIRCAM']['F115W'][d] = []
            self.discarded_slices['NIRCAM']['F150W'][d] = []

        self.discarded_slices['NIRCAM']['F277W'] = {}
        self.discarded_slices['NIRCAM']['F444W'] = {}

        for d in ['along', 'blong']:
            self.discarded_slices['NIRCAM']['F277W'][d] = []
            self.discarded_slices['NIRCAM']['F444W'][d] = []

        self.discarded_slices['MIRI'] = {}
        self.discarded_slices['MIRI']['F770W'] = {}
        self.discarded_slices['MIRI']['F770W']['image'] = []

        self.detector_frames = []
        self.filter = filter
        self.detector = detector

        if self.filter in ["F770W"]:
            self.instrument = "MIRI"
        else:
            self.instrument = "NIRCAM"

        self.default_storage_path = f"/mnt/C/JWST/COSMOS/{self.instrument}/Wisp_Templates"

        if debug:
            print(self.discarded_slices)

    @staticmethod
    def load_existing_wisp(file_path, debug=False):
        """
        Read a FITS file of wisp template.

        Args:
            filename (str): The name of the FITS file.
            debug (bool, optional): Print the details for debugging. Defaults to False.

        Return:
            Data (dictionary): A dictionary that contains every image in the FITS file.
        """
        data = {}
        detector_number = file_path.split('/')[-1].split('_')[-1][:-5]

        if debug:
            print(f"Detector reading: {detector_number}")

        data[f'{detector_number}'] = {}
        
        with fits.open(file_path) as hdul: 
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

    @staticmethod
    def load_frames(file_path, debug=False):
        """
        Read a FITS file of wisp template.

        Args:
            filename (str): The name of the FITS file.
            debug (bool, optional): Print the details for debugging. Defaults to False.

        Return:
            Data (dictionary): A dictionary that contains every image in the FITS file.
        """
        data = {}
        detector_number = file_path.split('/')[-1].split('_')[-1][:-5]
        if debug:
            print(f"Detector reading: {detector_number}")
        data[f'{detector_number}'] = {}
        
        with fits.open(file_path) as hdul: 
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

    def add_discarded_slices(self, slice_string):
        # read the detector name
        self.detector = slice_string.split('_')[3][3:]
        print(self.detector)
        # set the instruments
        if 'nrc' in slice_string:
            instrument = "NIRCAM"
        else:
            instrument = "MIRI"
            _filter = "F770W"
        
        # set the filters
        if 'along' in slice_string or 'blong' in slice_string:
            if '_02101_' in slice_string:
                _filter = "F277W"
            else:
                _filter = "F444W"
        else:
            if '_02101_' in slice_string:
                _filter = "F115W"
            else:
                _filter = "F150W"
        
        self.discarded_slices[f'{instrument}'][f'{_filter}'][f'{self.detector}'].append(slice_string)

    def write_json(self, output_path):
        # Write the discarded_slices dictionary to a JSON file
        with open(output_path, 'w+') as file:
            json.dump(self.discarded_slices, file, indent=4)

    def load_json(self, file_path):
        # Load the JSON file into a Python dictionary
        try:
            with open(file_path, 'r+') as file:
                self.discarded_slices = json.load(file)
        except Exception as e:
            with open(file_path, 'w+') as file:
                json.dump({}, file, indent=4)
            with open(file_path, 'r+') as file:
                self.discarded_slices = json.load(file)

    def visualize_template(self,) -> None:
        # load the wisp template
        wisp_path = f"{self.default_storage_path}/Wisp_{self.filter}_{self.detector}.fits"
        if os.path.exists(wisp_path):
            # print(wisp_path, Wisp.load_existing_wisp(wisp_path)[self.detector].keys())
            template = Wisp.load_existing_wisp(wisp_path)[self.detector]['WISP']
            image_visualization(template,
                    title=['Median wisp template'],
                    auto_color=True, save=True, 
                    output_path=f"/mnt/C/JWST/COSMOS/{self.instrument}/PNG/{self.filter}/{self.detector}_test_wisp.png")
            
        else:
            print("The wisp template you are trying to visualize is not available!!")

    def visualize_frames(self,) -> None:
        # load frames that build up the template
        if os.path.exists(frame_path):
            frame_path = f"{self.default_storage_path}/Frames_{self.filter}_{self.detector}.fits"
            frames = Wisp.load_frames(frame_path)[self.detector]['WISP']
            image_visualization(frames, auto_color=True, save=False)
            
        else:
            print("The frames you are trying to visualize is not available!!")

    def make_new_wisp_template(self, number_of_frames, include_stars) -> np.array:
        image_pool = glob.glob(f"/mnt/C/JWST/COSMOS/{self.instrument}/{self.filter}/o*/*{self.detector}*cor_cal.fits")
        # print(image_pool)
        # discard the frames that are already flagged as bright-star-contaminated 
        image_pool = [img for img in image_pool if img.split("/")[-1] not in self.discarded_slices[self.instrument][f'{self.filter}'][f'{self.detector}']]
        total_number = len(image_pool)
        print(f"{total_number} files to be used in evaluating the wisp.")
        
        current_step = 0
        # Iterate over the image pool
        for ind in range(len(image_pool)):
            image = fits_reader(image_pool[ind])[self.detector]['SCI']

            try:
                # Mask the sources, nan values, and hot pixels
                masked_image = mask_nan_sources(image, include_stars=include_stars)
                
                if type(masked_image)== bool:
                    total_number -= 1
                    current_step -= 1
                    print(f"[Warning] Slice {image_pool[ind].split('/')[-1]} has extremely bright stars in view.")
                    print(f"Exclude this slice as one of the wisp collection, if you want to include it, please set the parameter 'include_stars'=True")
                    print(f"{total_number} files remained to be used in evaluating the wisp.")
                    self.add_discarded_slices(image_pool[ind].split('/')[-1])

                else:
                    self.detector_frames.append(masked_image-calculate_pedestal(masked_image))

            except Exception as e:
                total_number -= 1
                current_step -= 1
                print(f"Error occurred when including {image_pool[ind].split('/')[-1]} as one of the wisp collection:")
                print(str(e))
                print(f"{total_number} files remained to be used in evaluating the wisp.")
                continue

            current_step += 1
            try:
                print(f"{np.round((current_step)/total_number*100, 2)}% Finished.")
            except Exception:
                pass
                 
        # modelling wisp by taking median 
        self.median_wisp_template_frame = np.nanmedian(self.detector_frames, axis=0)

        # Write discarded slices to a json file
        self.write_json(f'{self.default_storage_path}/discarded_frames.json')

    def save_new_wisp_template(self,):
        # Create a FITS PrimaryHDU object
        primaryhdu = fits.PrimaryHDU()

        # Create a FITS ImageHDU object from the data
        median_wisp_hdu = fits.ImageHDU(self.median_wisp_template_frame)
        median_wisp_hdu.header['EXTNAME'] = ('WISP', 'Wisp template Design for COSMOS-Webb Field')

        # Create a FITS HDU list and save it to a file
        hdul = fits.HDUList([primaryhdu, median_wisp_hdu])
        hdul.writeto(f'{self.default_storage_path}/Wisp_{self.filter}_{self.detector}.fits', overwrite=True)

    def save_frames_for_wisp_template(self,):
        # Create a FITS PrimaryHDU object
        primaryhdu = fits.PrimaryHDU()
    
        # Create a FITS HDU list
        hdul = fits.HDUList([primaryhdu])

        # Create a FITS ImageHDU object from the data
        for frame in self.detector_frames:
            median_wisp_hdu = fits.ImageHDU(frame)
            median_wisp_hdu.header['EXTNAME'] = ('FRAME', 'Frames that are combinded to make wisp template.')
            hdul.append(median_wisp_hdu)
        
        hdul.writeto(f'{self.default_storage_path}/Frame_{self.filter}_{self.detector}.fits', overwrite=True)
    
def mask_nan_sources(image, include_stars):
    # Calclate median and replace nan values with it
    median = np.nanmedian(image)
    masked_image = np.where(np.isnan(image), median, image)

    # Use tired source detection algorithm to find and mask sources
    sources = tiered_source_detection(masked_image, [15, 7, 5], snr=5)
    mask = np.logical_not(sources).astype(float)

    # Check if there are bright stars that contaminate the image
    if not include_stars:
        x, y = mask.shape[0], mask.shape[1]
        if np.count_nonzero(mask)/(x*y) < 0.9:
            return False

    masked_image = np.multiply(mask, masked_image)

    # replace sources with median
    masked_image = np.where(masked_image==0, median, masked_image)
    return masked_image

def combine_wisp_frames(major_frame, minor_frame) -> np.array:
    major_frame += minor_frame
    return major_frame

def smooth_with_gaussian(image, sigma, visualization=False):
    """
    Convolute the image with Gaussian function with specified size.

    Args:
        image(np.ndarray): The image to be convolved. (Here is the wisp template)
        sigma(float): The standard deviation of the gaussian function in the unit of pixels.

    Returns:
        filtered_image(np.ndarray)
    """

    from scipy.ndimage import gaussian_filter

    # Apply Gaussian _filter with sigma = 3
    sigma = 3
    filtered_image = gaussian_filter(image, sigma)

    if visualization:
        # Display the filtered image
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Filtered Image with Gaussian Kernel (sigma=3)')
        plt.colorbar(label='Intensity')
        plt.axis('off')
        plt.show()

    return filtered_image

def minimize_variance(wisp_template, image_containing_wisps, conv=False) -> float:
    """
    Minimize the variance of the entire image that masked out the sources
    by a given number of Wisp multiplier (A).
    """
    # Multiply everything with the miri effective area
    wisp_template = multiply_by_miri_effective_area(wisp_template)
    image_containing_wisps = multiply_by_miri_effective_area(image_containing_wisps)

    # Creating a 2D grid of wisp_multiplier and wisp_pedestal
    wisp_multiplier, wisp_pedestal = np.linspace(0, 3, 180), np.array([0.0])
    wisp_multiplier_grid, wisp_pedestal_grid = np.meshgrid(wisp_multiplier, wisp_pedestal)

    # Initializing a matrix to store the variance values
    variance_matrix = np.zeros_like(wisp_multiplier_grid)

    # Calculating the variance of image that contains wisp
    var_d = np.nanvar(image_containing_wisps.flatten())
    # var_d = np.nanmedian(image_containing_wisps.flatten())
    # print(var_d)

    # clip 5-sigma values
    _wisp_template = sigma_clip(wisp_template, 3, maxiters=5)
    _wisp_template = _wisp_template.filled(fill_value=np.nan)
    if conv:
        conv_wisp = convolve(_wisp_template, 
                            Gaussian2DKernel(x_stddev=7, x_size=15))
        conv_wisp = multiply_by_miri_effective_area(conv_wisp)

        # image_visualization([conv_wisp, _wisp_template], 
        #                     scale_data=_wisp_template,
        #                     share_scale=True)
    
    else:
        pass
        # image_visualization([_wisp_template])
        
    # Calculating the variance for each point in the grid
    print("Calculating the variance for each point in the grid ......")
    for i in range(variance_matrix.shape[0]):
        for j in range(variance_matrix.shape[1]):
            if conv:
                d_rev = image_containing_wisps - (conv_wisp+wisp_pedestal_grid[i, j]) * wisp_multiplier_grid[i, j]

            else:
                d_rev = image_containing_wisps - (_wisp_template+wisp_pedestal_grid[i, j]) * wisp_multiplier_grid[i, j]

            var_d_rev = np.nanvar(d_rev.flatten())
            # var_d_rev = np.nanmedian(d_rev.flatten())
            variance_matrix[i, j] = var_d_rev - var_d

    best_multiplier_ind, best_pedestal_ind = np.unravel_index(np.argmin(variance_matrix, axis=None), 
                                                              variance_matrix.shape)

    # print(variance_matrix[best_multiplier_ind, best_pedestal_ind])
    min_var_multiplier = wisp_multiplier_grid[best_multiplier_ind, best_pedestal_ind]
    min_var_pedestal = wisp_pedestal_grid[best_multiplier_ind, best_pedestal_ind]

    # Creating a contour plot
    # plt.figure(figsize=(10, 8))
    # cp = plt.contourf(wisp_multiplier_grid, wisp_pedestal_grid, variance_matrix, levels=100, cmap='viridis')
    # plt.colorbar(cp, label='Variance of Residuals')
    # plt.scatter(min_var_multiplier, min_var_pedestal, color='red', marker='x', s=100, label='Minimum Variance Point')
    # plt.title('Variance of Residuals for Different Parameter Values')
    # plt.xlabel('A (wisp_multiplier)')
    # plt.ylabel('B (wisp_pedestal)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # fig, ax = plt.subplots(figsize=(4, 3), dpi=250)
    # ax.plot(wisp_multiplier, delta_variance)
    # plt.show()
    
    return min_var_multiplier, min_var_pedestal

