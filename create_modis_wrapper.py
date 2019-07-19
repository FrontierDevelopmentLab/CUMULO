import create_modis
import os


def create_modis_for_dirs(source_dir, save_dir):
    modis_paths = []
    bashed_files = os.popen("find ./DATA/modis-l1 -type f").read().rstrip()
    filepaths = bashed_files.split("\n")
    source_dirs = set()
    for path in filepaths:
        head, tail = os.path.split(path)
        source_dirs.add(head)

    for directory in source_dirs:
        print(directory)

        create_modis.run(path="{}/".format(directory), save_dir=save_dir)
