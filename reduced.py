import warnings
warnings.filterwarnings("ignore", module="photutils")
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.pyplot")

from MIRI import MIRI_Image
from MIRI import run_Pipeline_3, sort_corrected_images, sort_images

from utility import *

from pink_noise import *

import glob

files_to_be_reduced = sorted(glob.glob(rf"/mnt/C/JWST/COSMOS/MIRI/F770W/jw*/*_uncal.fits"))

for fits_file in files_to_be_reduced:
    fits_file = fits_file.split('/')[-1]
    folder_name = remove_file_suffix(fits_file)
    # print(folder_name)

    a = MIRI_Image("F770W", filename=fits_file)
    # a.run_MIRI_Detector1Pipeline()
    # a.run_MIRI_Image2Pipeline()
    # a.remove_pink_noise()
    # a.wisp_removal(visualize_frames=False, 
    #                visualize_template=False, 
    #                include_stars=False, conv=True)
    # a.subtract_background()
    if not os.path.exists(os.path.join("/mnt/C/JWST/COSMOS/MIRI/F770W", folder_name, f"{folder_name}_sub_bri_col.fits")):
        a.subtract_brighten_columns()


# all_path = sorted(glob.glob(f"/mnt/C/JWST/COSMOS/MIRI/F770W/jw*/jw*_bkg_sub.fits"))

# for file_name in all_path:
#     obs_num = file_name.split("/")[-1].split("_")[0].strip("jw")[5:8] 
#     run_Pipeline_3("MIRI", "F770W", obs_num)
#     break