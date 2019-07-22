import create_modis_wrapper
#from GCP_Files_Tool import GCP_Tools
import extract_payload_wrapper
import unsupervised_pipeline

#create_modis_wrapper.create_modis_for_dirs(source_dir="./DATA/modis-l1", save_dir="./DATA/output/")

# extract_payload.random_tile_extract_from_file(file_in="./DATA/output/2008057_0210.pkl", payload_path="./DATA/tiles/payload/", metadata_path="./DATA/tiles/metadata", tile_size=3)

#extract_payload_wrapper.extract_payload_for_dirs(
    # source_dir="./DATA/output",
    # payload_path="./DATA/tiles/payload",
    # metadata_path="./DATA/tiles/metadata",
    # tile_size=10)

unsupervised_pipeline.unsupervised_pipeline_run("./DATA/modis-l1/2008/2008/056/MOD021KM.A2008056.0755.061.2017255070653.hdf", "./DATA/output/pipeline_test")