import create_modis
#from GCP_Files_Tool import GCP_Tools
#import extract_payload


create_modis.run(path="./DATA/modis-l1/2008/2008/056/", save_dir="./DATA/output/")

#extract_payload.random_tile_extract_from_file(file_in="./DATA/output/2008057_0210.pkl", payload_path="./DATA/tiles/payload/", metadata_path="./DATA/tiles/metadata", tile_size=3)