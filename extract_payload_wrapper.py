import extract_payload
import os


def extract_payload_for_dirs(source_dir, payload_path, metadata_path, tile_size)

    paths = []
    for roots, dirs, files in os.walk(source_dir):
        for file in files:
            paths.append(os.path.join(roots, file))

    for file in paths:
        extract_payload.random_tile_extract_from_file(
            file_in=file, payload_path=payload_path, metadata_path=metadata_path, tile_size=tile_size)
