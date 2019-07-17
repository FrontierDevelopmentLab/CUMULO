# GCP_Tools module

import gcsfs
import os
import json


class GcpDataTools:
    """
    ToDo class docstring
    ToDo !!!!! YAML config
    ToDo move string functions to a different subclass
    ToDo mv / del - perhaps not?
    ToDo file-locks?
    """

    def __init__(self):
        pass

        # ToDo automatic close of gcp connection - context
        # ToDo __all__

    def connect(self, json_path):
        """
        :param json_path: path to json with the project and bucket name
        :return: None (creates self.connection, self.project_name and self.bucket_name)
        expects a "project_name" and "bucket_name" in the json
        """
        with open(json_path, "r") as read_file:
            details = json.load(read_file)

            try:
                self.project_name = details["project_name"]
            except: KeyError("no key \'project_name\' found in json")

            try:
                self.bucket_name = details["bucket_name"]
            except: KeyError("no key \'bucket_name\' found in json")

            self.connection = gcsfs.GCSFileSystem(project=self.project_name)
            print("connection to {} on {} established".format(self.bucket_name, self.project_name))


    @staticmethod
    def path_check(bucketpath):
        """
        :param bucketpath: a gcs path for a directory or file in the bucket
        :return: none
        Checks if the path is a string, raises an exception if not
        """
        if not isinstance(bucketpath, str):
            raise SystemExit('Your path must be a string')

        # ToDo better graceful fail

    def path_generator(self, bucketpath, verbose=False):
        """
        :param bucketpath: a gcs path for a directory or file in the bucket
        :param verbose: boolean as to the verbosity
        :return: an absolute path to the indicated resource
        lazy function to add on the bucket id and in-bucket path together. string validity checked
        """
        self.path_check(bucketpath)

        abs_bucketpath = self.bucket_name + "/" + bucketpath

        if verbose:
            print("Your path is {}".format(abs_bucketpath))

        return abs_bucketpath

    def list_directory(self, bucketpath = ""):
        """
        :param bucketpath: GCS path to a bucket or directory within a bucket
        :return: list of strings of filenames
        Behaves like unix "ls" for querying buckets.
        """
        resource_path = self.path_generator(bucketpath, verbose=True)
        return self.connection.ls(resource_path)

    def get_datafile(self, bucketpath, targetpath, verbose=False):
        """
        :param bucketpath: the path to the datafile
        :param targetpath: the path to save the file
        :param verbose: verbosity switch
        :return: none
        pulls a single file from a bucket path to a specified directory - for public use.
        Uses __get_datafile_if_not_exists
        """
        path_to_remote_object = self.path_generator(bucketpath)

        _, file_name = os.path.split(path_to_remote_object)
        local_path = os.path.join(targetpath, file_name)

        self.get_datafile_if_not_exists(path_to_remote_object, local_path)
        print("file downloaded")

        # ToDo save to the local directory as default

    def push_datafile(self, bucketpath, source_path):
        """
        :param bucketpath:
        :param source_path:
        :return: place a single file at a specified bucket path
        """
        # ToDo push datafile logic
        pass

    def get_data_directory(self, bucketpath, output_dir):
        """
        :param bucketpath: the path of the object/directory of the wanted data
        :param output_dir: the target path to save the files
        :return: none
        The method will download the bucket directory to access the wanted files, recursively. \
        Files will not be renamed.
        """
        pass

        bucket_files_list, bucket_directories_list = self.bucket_walk(bucketpath)

        # bucketpath is the absolute root that is removed
        self.create_local_directories(bucket_directories_list, bucketpath, output_dir)
        # ToDo refactor the variable name "bucketpath"

        # create local file paths
        local_filepath_list = self.absolute_bucket_to_local_path(
                                                                bucket_files_list,
                                                                bucket_directories_list,
                                                                output_dir,
                                                                bucketpath)

        self.check_filelist_length_is_equal(bucket_files_list, local_filepath_list)

        # recreate the bucket paths
        bucket_paths_list = []
        for file_path, directory_path in zip(bucket_files_list, bucket_directories_list):
            bucket_path = directory_path + "/" + file_path
            print(bucket_path)
            bucket_paths_list.append(bucket_path)

        bucket_paths_list = [path for path in bucket_paths_list if path[-1] != "/"]  # filter out blank file(/ at end)
        local_filepath_list = [path for path in local_filepath_list if path[-1] != "/"]

        for bucket_filepath, local_filepath in zip(bucket_paths_list, local_filepath_list):
            self.get_datafile_if_not_exists(bucket_filepath, local_filepath, verbose=False)

        print("\n\nMirror down completed")

        # ToDo elevate verbosity controls class-wide

    @staticmethod
    def check_filelist_length_is_equal(a_list, b_list):
        """to ensure that the bucket and local files are equal in length without a silent zip truncation"""
        if len(a_list) == len(b_list):
            pass
        else:
            print("a_list len: {} \nb_list len: {}".format(len(a_list), len(b_list)))

            raise SystemExit('Error: bucket and local paths must be equal in length')

    def push_data_directory(self, bucketpath, source_dir):
        """
        :param bucketpath: the path to place object/directory of the data
        :param source_dir: the target path to save the files
        :return: none
        Upload a directory of data to the path in the bucket, recursively.
        """
        pass
        # ToDo push directories

    def local_walk(self, local_path):
        """
        :param local_path: path to the local directory
        :return: list of roots, directories and files as a tuple - (root,dirs,files)
        walking through the local directory, for a recursive list of dirs to make and files to upload
        """
        self.path_check(local_path)
        walk_object = os.walk(local_path, topdown=True)

        return walk_object

    def bucket_walk(self, bucketpath):
        """
        :param bucketpath: path to the remote directory
        :return: list of directories to make and files to upload as a nested list - [[dirs], [files]]
        Returns the paths for replicating directories and uploading data ToDo update the bucket_walk docstring
        """
        filepath_list = self.bucket_list_filepaths(bucketpath)

        # initialise lists for appending
        filename_list = []
        directory_list = []
        for filepath in filepath_list:
            head, tail = os.path.split(filepath)
            filename_list.append(tail)
            directory_list.append(head)
        return [filename_list, directory_list]
    # ToDo sets to minimise dir creation attempts
    # ToDo yield to iterator (generator)

    def bucket_list_filepaths(self, bucketpath):
        """
        :param bucketpath: path to the remote directory
        :return: a list of all nested files' paths
        We can only pull the file directories from GCP buckets. Will need further handling to extract the dirs
        #ToDo rewrite references to bucket regex
        """
        # ToDo change all self.path_generator variables to resource_path
        resource_path = self.path_generator(bucketpath)
        filepaths = self.connection.walk(resource_path)

        return filepaths

    def create_local_directories(self, bucket_directories_list, bucket_root_directory, output_dir):
        """
        :param bucket_directories_list: list of the directories in the bucket to recreate locally
        :param bucket_root_directory: the root directory in the bucket
        :param output_dir: the target local directory
        :return: None
        Recreates the directory structure in the bucket, removing the root directory and using relative links.
        If the directories exist, use them
        # ToDo verbosity
        """
        local_directory_path_list = self.absolute_bucket_to_local_path(
            bucket_directories_list,
            bucket_root_directory,
            output_dir)

        for local_directory_path in local_directory_path_list:
            if os.path.exists(local_directory_path):
                print("path: {} already exists, not created".format(local_directory_path))
            else:
                os.makedirs(local_directory_path)

    def get_datafile_if_not_exists(self, bucketpath, targetpath, verbose=False):
        """
        :param bucketpath: path to the datafile in the bucket
        :param targetpath: path to the local target location
        :param verbose: verbosity switch
        :return: none
        checks for existing files, and asks for user input, then downloads a file to the local targetpath
        """
        self.path_check(bucketpath)
        self.path_check(targetpath)
        # if os.path.exists(bucketpath):
        #     user_input = input("File {} already exists. Overwrite? [Y/n]".format(bucketpath))
        #     if user_input == "y" or "Y" or "\n":
        #         self.connection.get(rpath=bucketpath, lpath=targetpath)
        #
        #     else:
        #         print("{} not saved \n".format(bucketpath))
        #         pass

        print("getting file from {} to {}".format(bucketpath, targetpath))
        try:
            self.connection.get(rpath=bucketpath, lpath=targetpath)
        except FileNotFoundError:
            print("\n\n{} file not found - perhaps it is a directory?\n\n".format(bucketpath))
            pass

        if verbose:
            print("{} saved".format(bucketpath))

        # ToDo see if you can check the modified date
        # ToDo wrap user verification in new if statement to  overwrite all

    @staticmethod
    def absolute_bucket_to_local_path(bucket_path_list, bucket_root_directory, output_dir, bucketpath=""):
        """The root of the path is sliced off, based on length, returning a list of relative paths.
        create a list of paths by attaching the output directory to the relative paths, with some slash handling."""
        if isinstance(bucket_root_directory, str):
            slice_position = len(bucket_root_directory) + 1
            relative_path_list = []
            for path in bucket_path_list:
                relative_path_list.append(path[slice_position:])

        elif isinstance(bucket_root_directory, list):
            if not bucketpath:
                raise ValueError("""While creating local path from bucket paths with a directory list, root bucketpath 
                                cannot be empty""")
            slice_position = len(bucketpath) + 1
            relative_path_list = []
            for path, directory in zip(bucket_path_list, bucket_root_directory):
                relative_path_list.append(os.path.join(directory[slice_position:], path))

        else:
            raise TypeError("While creating local path from bucket path, expected static string or list")

        output_path_list = []
        for path in relative_path_list:
            output_path_list.append(os.path.join(output_dir + path))

        return output_path_list


if __name__ == "__main__":
    pass
    # ToDo main logic here
