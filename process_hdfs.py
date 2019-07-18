import create_modis_wrapper

print("source_directory: ../DATA/modis-l1/")

print("save_directory: ../DATA/output/")

create_modis_wrapper.create_modis_for_dirs(source_dir="../DATA/modis-l1", save_dir="../DATA/processed")

print("END OPERATION: files processed")
