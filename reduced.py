import warnings
warnings.filterwarnings("ignore", module="photutils")
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.pyplot")

from MIRI import MIRI_Image

from utility import *

from pink_noise import *

from tqdm import trange

import glob

instrument = "MIRI"
_filter = "F770W"
detector = ""
files_to_be_reduced = sorted(glob.glob(rf"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/jw*/*{detector}*_uncal.fits"))

for fits_file_index in trange(len(files_to_be_reduced)):
    fits_file = files_to_be_reduced[fits_file_index]
    fits_file = fits_file.split('/')[-1]
    folder_name = remove_file_suffix(fits_file)
    main_path = os.path.join(f"/mnt/C/JWST/COSMOS/{instrument}/{_filter}/", folder_name, folder_name)

    
    if not os.path.exists(main_path + "_bri_col_sub.fits"):
        a = MIRI_Image(_filter, filename=fits_file)
        # a.info()

        if not os.path.exists(main_path + "_rate.fits"):
            a.run_MIRI_Detector1Pipeline()

        if not os.path.exists(main_path + "_cor.fits"):
            a.remove_pink_noise()

        if not os.path.exists(main_path + "_cor_cal.fits"):
            a.run_MIRI_Image2Pipeline()

        if not os.path.exists(main_path + "_cor_wsp.fits"):
            a.wisp_removal(visualize_frames=False, 
                            visualize_template=False,
                            include_stars=True, 
                            debug=False,
                            conv=True)
            
        if not os.path.exists(main_path + "_bkg_sub.fits"):
            a.subtract_background()

        a.subtract_brighten_columns()
        a.gather_by_obs('bri_col_sub')


# all_path = sorted(glob.glob(f"/mnt/C/JWST/COSMOS/MIRI/F770W/jw*/jw*_bkg_sub.fits"))

# for file_name in all_path:
#     obs_num = file_name.split("/")[-1].split("_")[0].strip("jw")[5:8] 
#     run_Pipeline_3("MIRI", "F770W", obs_num)
#     break