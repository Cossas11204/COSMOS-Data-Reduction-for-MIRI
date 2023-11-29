import glob, os
from joblib import Parallel, delayed
from astroquery.mast import Observations

proposal_id = 1727
instrument_name = "MIRI/IMAGE"
filters = "F770W"
directory = f"./F770W/"


# Get observations for the given query
obs_list = Observations.query_criteria(proposal_id=proposal_id,
                                        instrument_name=instrument_name,
                                        filters=filters)
obs_list_pd = obs_list.to_pandas().sort_values(by='obs_id').reset_index(drop=True)


total_size = 0
#                 change to ":" retrieve ALL the data
#                            |
#                            V
for name in obs_list_pd.iloc[:]['obs_id']:
    # retrieve the data products for the given observation
    print(f"Gathering data products for: {name}", end=' ')

    mask = (obs_list['obs_id'] == name)
    data_products = Observations.get_product_list(obs_list[mask])
    
    # filter the data products for the uncalibrated images
    filtered_prod = Observations.filter_products(data_products, 
                                            productType="SCIENCE",
                                            extension="fits",
                                            calib_level=[1], 
                                            productSubGroupDescription="UNCAL")
    size = sum(filtered_prod['size'])
    total_size += size
    print(f"...retrieving {len(filtered_prod)} files ({size/1e9:.2f} GB)")

    # download the data products
    manifest = Observations.download_products(filtered_prod, curl_flag=False)

print("Finish fetching data products")
print(f"Total size: {total_size/1e9:.2f} GB")

manifest_list = glob.glob("mastDownload*")
print(f"Total files: {len(manifest_list)}")

def download_data_products(manifest):
    print(f"Downloading: {manifest}")
    os.system(f"chmod +x {manifest}")
    os.system(f"./{manifest}")

Parallel(n_jobs=16)(delayed(download_data_products)(manifest) for manifest in manifest_list)

if not os.path.exists(directory):
    os.system(f"mkdir {directory}")

for manifest in manifest_list:
    os.system(f"mv {manifest}/JWST/* {directory}")
    os.system(f"rm -R {manifest}")
    os.system(f"rm ./{manifest}")
