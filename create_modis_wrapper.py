import create_modis
import os


def create_modis_for_dirs(source_dir, save_dir):
    modis_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            modis_paths.append("{}/{}".format(root, file))

    source_dirs = set()
    for path in modis_paths:
        head, tail = os.path.split(path)
        source_dirs.add(head)

    for directory in source_dirs:
        print(directory)

        create_modis.run(path="{}/".format(directory), save_dir=save_dir)
