import os
os.environ['CRDS_CONTEXT'] = ''
os.environ['CRDS_PATH'] = '/mnt/D/D220624-data/JWST_data/crds_cache/'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import glob

import shutil

import numpy as np

from jwst.pipeline import Detector1Pipeline, Image2Pipeline

from photutils.background import Background2D, MedianBackground

from utility import *

from wisp import *

from pink_noise import remove_striping, calculate_pedestal


class MIRI_Image():
    def __init__(self, filter, filename=None, restart=False) -> None:
        # Set the basic parameters for file/path naming
        if not filename.endswith('.fits'):
            raise ValueError("Filename should be ended with fits!!!")
        elif not filename:
            raise ValueError("Please enter your filename!!!")
        else:
            self.fitsname = filename
            self.filter = filter 
            self.foldername = remove_file_suffix(filename) # remove filenaming conventions
            self.detector_name = self.foldername.split('_')[3]
            self.mode = self.detector_name[-5:]
            self.obs_num = self.foldername.split('_')[0][7:10]
            self.vis_num = self.foldername.split('_')[0][10:13]
            self.path = f"/mnt/C/JWST/COSMOS/MIRI/{self.filter}/{self.foldername}"

            print(f"Initializing MIRI Obj. for: {self.path}/{self.fitsname}")
            print(f"Observation: o{self.obs_num} Visit: {self.vis_num} Detector: {self.detector_name}")
            
        # Delete all unrevelent files when restart
        if restart:
            # os.system(f"find {self.path} -type f -not -name '*_uncal.fits' -exec rm {{}} \;")
            pass
        
        # Set up the logger for this MIRI Object
        if not os.path.exists(f'./Config_and_Logging'):
            os.mkdir(f'./Config_and_Logging')

        if not os.path.exists(f'suppress_all_{self.detector_name}.cfg'):
            with open("./Config_and_Logging/suppress_all.cfg", "r") as file:
                config = file.read()

            config = config.replace(
                r"%filter%", filter
            )

            config = config.replace(
                r"%detector%", self.detector_name
            )  

            with open(f"./Config_and_Logging/suppress_all_{self.detector_name}.cfg", "w") as file:
                file.write(config)

        print(f"MIRI Obj. Initialized successfully.")

    def run_MIRI_Detector1Pipeline(self) -> None:
        # Run JWST Pipeline Stage 1 using uncalibrated raw data
        result = Detector1Pipeline.call(f"{self.path}/{self.fitsname}",
                                        output_dir=f"{self.path}/", 
                                        save_results=True,
                                        logcfg = f'./Config_and_Logging/suppress_all_{self.detector_name}.cfg' ,
                                        steps={
                                                'jump': {'save_results':True}
                                                }
                                        )
        
    def run_MIRI_Image2Pipeline(self) -> None:
        # Run JWST Pipeline Stage 2 with result from Detector1Pipeline
        result = Image2Pipeline.call(f"{self.path}/{remove_file_suffix(self.fitsname)}_rate.fits",
                                        output_dir=f"{self.path}/", 
                                        save_results=True,
                                        logcfg = f'./Config_and_Logging/suppress_all_{self.detector_name}.cfg' ,
                                        steps={
                                                }
                                        )
        
    def remove_pink_noise(self) -> None:
        dict = fits_reader(f"{self.path}/{remove_file_suffix(self.fitsname)}_cal.fits")
        img_data = dict[f'{self.mode}']['SCI']
        final_cor_image = np.zeros(img_data.shape)
        
        # cutout of different parts of miri image detector
        un_cor_miri_corona_lyot = img_data[745:, :279]
        un_cor_miri_corona_4qpm = img_data[:682, :232]
        un_cor_miri_img = img_data[:, 400:]

        # mask sources for different parts of miri image detector
        masked_corona_lyot_image = mask_sources(un_cor_miri_corona_lyot, sigma_values=[5, 2], snr=3)
        masked_corona_4qpm_image = mask_sources(un_cor_miri_corona_4qpm, sigma_values=[5, 2], snr=3)
        masked_miri_image = mask_sources(un_cor_miri_img, sigma_values=[5, 2], snr=3)

        # empty space between each parts of miri image detector
        miri_empty_1 = img_data[682:745, :279]
        miri_empty_2 = img_data[:, 279:350]
        miri_empty_3 = img_data[:682, 232:279]

        # remove 1/f noise from different parts of miri image detector
        miri_corona_lyot = remove_striping(masked_corona_lyot_image, un_cor_miri_corona_lyot, sigma=2, axis=1, SWC=False, instrument='lyot')
        miri_corona_4qpm = remove_striping(masked_corona_4qpm_image, un_cor_miri_corona_4qpm, sigma=2, axis=1, SWC=False, instrument='4qpm')
        miri_img = remove_striping(masked_miri_image, un_cor_miri_img, sigma=2, axis=1, SWC=False)

        # combine each parts into FULL array
        final_cor_image[682:745, :279] = 0
        final_cor_image[:, 279:350] = 0
        final_cor_image[:682, 232:279] = 0
        final_cor_image[745:, :279] = 0
        final_cor_image[:682, :232] = 0
        final_cor_image[:, 400:] = miri_img

        image_visualization([img_data, final_cor_image], color_style='jet', auto_color=True,
                            title=['Before Stripe Correction','After Stripe Correction'],
                            output_path=f'{self.path}/stripe_corrected_image.png',
                            share_scale=True, save=True, scale_data=miri_img,
                            vmin_value=20, vmax_value=95)
    
        record_and_save_data(self.path, self.fitsname, 
                             final_cor_image, calculate_pedestal(final_cor_image), suffix='cal')

    def load_corrected_data(self) -> dict:
        return fits_reader(f"{self.path}/{remove_file_suffix(self.fitsname)}_cor_cal.fits")

    def wisp_removal(self, debug=False, conv=False,
                    visualize_frames=False, 
                    visualize_template=False,
                    include_stars=False,) -> None:
        """
        Load/create a wisp template based on the given detector or _filter.
        Then Subtract it from the image containing wisps.

        Args:
            visualize_frames(bool): Whether to visualize all frames constructing the wisp template.
            visualize_template(bool): Whether to visualize the wisp template.
            include_stars(bool): We strongly suggest to set this to be False. 
            If an image contains a bright star, the psf of the star will be inprinted in the wisp template.

        Returns:
            None
        """
        # initialize the wisp object
        Wisp_obj = Wisp(self.filter, self.mode)
        if debug:
            print(Wisp_obj.discarded_slices)

        # Reload wisp object if the json file has no length
        if Wisp_obj.discarded_slices is None:
            Wisp_obj.load_json(f'{Wisp_obj.default_storage_path}/discarded_frames.json')

        # Try to load the wisp template
        try:
            wisp = Wisp_obj.load_existing_wisp(f'{Wisp_obj.default_storage_path}/Wisp_{self.filter}_{self.mode}.fits')
        
        except FileNotFoundError:
            print(f"Wisp file for {self.filter}_{self.mode} not found. Creating a new wisp template.")
            Wisp_obj.make_new_wisp_template(15, include_stars) # number_of_frames = 15
            Wisp_obj.save_new_wisp_template() # Save the new wisp template
            wisp = Wisp_obj.load_existing_wisp(f'{Wisp_obj.default_storage_path}/Wisp_{self.filter}_{self.mode}.fits')

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Visualize the frames that make the wisp template
        if visualize_frames:
            print("Visualizing wisp frames ......")
            Wisp_obj.visualize_frames(debug=True)

        # Visualize the wisp template
        if visualize_template:
            print("Visualizing wisp template ......")
            Wisp_obj.visualize_template()

        # Save the new wisp template
        wisp = multiply_by_miri_effective_area(wisp[Wisp_obj.detector]['WISP'], nan=False)
        image_containing_wisps = fits_reader(f"{self.path}/{remove_file_suffix(self.fitsname)}_cor_cal.fits")[Wisp_obj.detector]['SCI']

        print("Scaling wisp and subtract ......")
        wisp_multiplier, wisp_pedestal = minimize_variance(wisp, image_containing_wisps, conv=conv)
        print(f"Wisp multiplier A = {wisp_multiplier}, Wisp pedestal B = {wisp_pedestal}")
        image_without_wisps = image_containing_wisps - wisp_multiplier * (wisp+wisp_pedestal)
        image_without_wisps = multiply_by_miri_effective_area(image_without_wisps, nan=False)
        # image_visualization(wisp)

        # mask sources
        image_without_wisps_for_vis = mask_sources(image_without_wisps)
        image_containing_wisps_for_vis = mask_sources(image_containing_wisps)

        # visualize the original image, wisp template, and result image
        comp_list = [image_containing_wisps_for_vis, wisp, image_without_wisps_for_vis]
        comp_list = [multiply_by_miri_effective_area(data, nan=True) for data in comp_list]
        title = [f"sigma ={np.round(np.nanstd(sigma_clip(data, 3)), 3)} \n median={np.round(np.nanmedian(sigma_clip(data, 3)), 3)}" for data in comp_list]
        image_visualization(comp_list, auto_color=True, share_scale=True, show=False,
                            vmin_value=50, vmax_value=95, img_dpi=300, scale_data=image_containing_wisps_for_vis,
                            save=True, output_path=f'{self.path}/wisp_corrected_image.png',
                            title=title
                            )

        print("Saving data to *_cor_wsp.fits ......")
        record_and_save_data(self.path, remove_file_suffix(self.fitsname), 
                             image_without_wisps, calculate_pedestal(image_without_wisps), suffix='wsp')
        print("*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-.*-")
        
    def subtract_background(self):
        file_name = os.path.join(self.path, f'{self.foldername}_cor_wsp.fits')
        image = fits_reader(file_name)[f'{self.mode}']['SCI']
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (50, 50), filter_size=(3, 3),
                            bkg_estimator=bkg_estimator)
        image -= bkg.background  # subtract the background

        path = "/".join(file_name.split('/')[:-1])
        fits_file = remove_file_suffix(file_name.split('/')[-1])

        bkg_image = multiply_by_miri_effective_area(bkg.background, nan=False)
        bkg_sub_image = multiply_by_miri_effective_area(image, nan=False)
        
        record_and_save_data(path, fits_file, bkg_image, pedestal=None, suffix='bkg')
        record_and_save_data(path, fits_file, bkg_sub_image, pedestal=None, suffix='bkg_sub')
        
    def subtract_brighten_columns(self,):
        data_dict = fits_reader(f"{self.path}/{remove_file_suffix(self.fitsname)}_bkg_sub.fits")
        img_data = data_dict['image']['SCI']
        img = multiply_by_miri_effective_area(img_data, nan=True)

        masked_img = mask_sources(img, nan=True)
        masked_img = sigma_clip(masked_img)
        masked_img = np.copy(masked_img)

        col_median = np.nanmedian(masked_img, axis=1)
        col_median = col_median.reshape(-1, 1)  # Correct the shape to (1024, 1)
        col_median = np.repeat(col_median, masked_img.shape[1], axis=1)  # Repeat along columns

        smoothed_col_med = convolve(col_median, Box2DKernel(width=5))
        sub_img = multiply_by_miri_effective_area(img_data - smoothed_col_med, nan=False)
        record_and_save_data(self.path, 
                             self.fitsname, 
                             img, pedestal=None, 
                             suffix='bri_col_sub')

import jwst.associations
import jwst.associations.mkpool
from jwst.pipeline import Image3Pipeline

def sort_corrected_images(instrument, _filter) -> None:
    all_path = sorted(glob.glob(f"/mnt/C/JWST/COSMOS/{instrument}/F770W/jw*/jw*cor_cal.fits"))

    for file_name in all_path:
        obs_num = file_name.split("/")[-1].split("_")[0].strip("jw")[5:8] 
        dir_path = f"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/o{obs_num}/"

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        shutil.copyfile(file_name, os.path.join(dir_path, file_name.split("/")[-1]))

def sort_images(instrument, _filter, suffix) -> None:
    all_path = sorted(glob.glob(f"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/jw*/jw*_{suffix}.fits"))

    for file_name in all_path:
        obs_num = file_name.split("/")[-1].split("_")[0].strip("jw")[5:8] 
        dir_path = f"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/o{obs_num}/"

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        shutil.copyfile(file_name, os.path.join(dir_path, file_name.split("/")[-1]))

def run_Pipeline_3(instrument, _filter, obs_num, suffix):
    """
    Creates association file and runs JWST Image3Pipeline with the default settings.

    Args:
        obs_num(int): Observation number for which data should be combined. 
        filter(str): Filters avaliable in the field. Could be: 'F115W', 'F150W', 'F277W', 'F444W', or 'F770W'.
    
    Return:	
        None
    """

    if not os.path.exists(f"/mnt/C/JWST/COSMOS/{instrument}/Reduced"):
        os.mkdir(f"/mnt/C/JWST/COSMOS/{instrument}/Reduced")

    # Set the parameters for an Level-3 association 
    association_pool = jwst.associations.mkpool.mkpool(glob.glob(f"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/o{obs_num}/*_{suffix}.fits"))
    association_rules = jwst.associations.registry.AssociationRegistry()
    association = jwst.associations.generate(association_pool, association_rules)[0]
    
    file_name, serialized = association.dump()
    file_name = f"/mnt/C/JWST/COSMOS/NIRCAM/{_filter}/o{obs_num}/{file_name}"

    with open(file_name, 'w') as file_handle:
        file_handle.write(serialized)
    
    with open(file_name, 'r') as file_handle:
        association = jwst.associations.load_asn(file_handle)

    # Run the JWST Image3Pipeline with the association
    results = Image3Pipeline.call(association,
                                  output_dir=f"/mnt/C/JWST/COSMOS/{instrument}/Reduced/{_filter}/",
                                  logcfg = f'/mnt/C/JWST/COSMOS/{instrument}/Config_and_Logging/Pipeline3.cfg' ,
                                  save_results=True,
                                  steps={
                                         'tweakreg': {'fitgeometry': 'rshift',
                                                      'use_custom_catalogs': False,# 'abs_refcat': 'GAIADR2'
                                                      },
                                         'skymatch': {'skymethod':'global+match',
                                                      'subtract': False,
                                                      },
                                         'resample': {'kernel': 'lanczos3', 
                                                      'fillval': f'{0}', 
                                                      'weight_type': 'exptime',
                                                      },
                                         'outlier_detection': {# 'output_dir': f"/mnt/C/JWST/COSMOS/{instrument}/Reduced/midproducts/",
                                                               'in_memory' : True,
                                                               },
                                         'source_catalog': {'skip': True,
                                                            },
                                        }
                                  )