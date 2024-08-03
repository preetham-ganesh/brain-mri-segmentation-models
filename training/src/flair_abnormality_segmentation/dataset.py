import os
import zipfile
import sys


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)


import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
import skimage
import numpy as np

from src.utils import check_directory_path_existence

from typing import Dict, Any


class Dataset(object):
    """Loads the dataset based on the model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def extract_data_from_zip_file(self) -> None:
        """Extracts files from downloaded data zip file.

        Extracts files from downloaded data zip file.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory path for downloaded data zip file.
        self.home_directory_path = os.getcwd()
        zip_file_path = "{}/data/raw_data/archive.zip".format(self.home_directory_path)

        # Creates the directory path.
        self.extracted_data_directory_path = check_directory_path_existence(
            "data/extracted_data/lgg_mri_segmentation"
        )

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            "{}/kaggle_3m/data.csv".format(self.extracted_data_directory_path)
        ):

            # Extracts files from downloaded data zip file into a directory.
            try:
                with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                    zip_file.extractall(self.extracted_data_directory_path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    "{} does not exist. Download data from ".format(zip_file_path)
                    + "'https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation', and place it in "
                    + "'data/raw_data' as 'archive.zip'."
                )
            print(
                "Finished extracting files from 'archive.zip' to {}.".format(
                    self.extracted_data_directory_path
                )
            )
            print()

    def load_dataset_file_paths(self) -> None:
        """Loads file paths of images & masks in the dataset.

        Loads file paths of images & masks in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Lists patient's directory names from the extracted data directory.
        patients_directory_names = os.listdir(
            "{}/kaggle_3m".format(self.extracted_data_directory_path)
        )

        # Creates empty list to store image & mask file paths as records.
        self.file_paths = list()

        # Iterates across patients directory names.
        for directory_name in patients_directory_names:
            patient_directory_path = "{}/kaggle_3m/{}".format(
                self.extracted_data_directory_path, directory_name
            )

            # If directory name is not a directory or does not start with 'TCGA'.
            if not os.path.isdir(
                patient_directory_path
            ) or not directory_name.startswith("TCGA"):
                continue

            # Lists file names in the patient directory path.
            image_file_names = os.listdir(patient_directory_path)

            # Iterates aross possible image ids.
            n_images = len(image_file_names) // 2
            image_id = 0
            while image_id <= n_images:
                image_file_path = "{}/{}_{}.tif".format(
                    patient_directory_path, directory_name, image_id + 1
                )
                mask_file_path = "{}/{}_{}.tif".format(
                    patient_directory_path, directory_name, image_id + 1
                )

                # Checks if image & mask file paths are valid.
                if os.path.isfile(image_file_path) and os.path.isfile(mask_file_path):
                    self.file_paths.append(
                        {
                            "image_file_path": image_file_path,
                            "mask_file_path": mask_file_path,
                        }
                    )
                image_id += 1

        # Converts list of file paths as records into dataframe.
        self.file_paths = pd.DataFrame.from_records(self.file_paths)
        print(
            "No. of image & mask pair examples in the dataset: {}".format(
                len(self.file_paths)
            )
        )
        print()

    def split_dataset(self):
        """Splits dataset into train & test data splits. Creates object for KFold cross validation.

        Splits dataset into train & test data splits. Creates object for KFold cross validation.

        Args:
            None.

        Returns:
            None.
        """
        # Splits file paths into train & test splits.
        (
            self.train_file_paths,
            self.test_file_paths,
        ) = train_test_split(
            self.file_paths,
            test_size=self.model_configuration["dataset"]["split_percentage"]["test"],
            shuffle=True,
        )

        # Creates object for K-Fold cross validation.
        self.k_fold = KFold(
            n_splits=self.model_configuration["dataset"]["k_fold"]["n_splits"]
        )

    def shuffle_slice_datasets(
        self, train_file_paths: pd.DataFrame, validation_file_paths: pd.DataFrame
    ) -> None:
        """Converts images & masks file paths into tensorflow dataset.

        Converts images & masks file paths into tensorflow dataset & slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Checks type & values of arguments.
        assert isinstance(
            train_file_paths, pd.DataFrame
        ), "Variable train_file_paths should be of type 'pd.DataFrame'."
        assert isinstance(
            validation_file_paths, pd.DataFrame
        ), "Variable validation_file_paths should be of type 'pd.DataFrame'."

        # Zips images & annotations file paths into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(train_file_paths["image_file_path"]),
                list(train_file_paths["mask_file_path"]),
            )
        ).shuffle(len(train_file_paths))
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(validation_file_paths["image_file_path"]),
                list(validation_file_paths["mask_file_path"]),
            )
        ).shuffle(len(validation_file_paths))
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.test_file_paths["image_file_path"]),
                list(self.test_file_paths["mask_file_path"]),
            )
        ).shuffle(len(self.test_file_paths))

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.validation_dataset = self.validation_dataset.batch(self.batch_size)
        self.test_dataset = self.test_dataset.batch(self.batch_size)

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = len(train_file_paths) // self.batch_size
        self.n_validation_steps_per_epoch = (
            len(validation_file_paths) // self.batch_size
        )
        self.n_test_steps_per_epoch = len(self.test_file_paths) // self.batch_size
        print("No. of train steps per epoch: {}".format(self.n_train_steps_per_epoch))
        print(
            "No. of validation steps per epoch: {}".format(
                self.n_validation_steps_per_epoch
            )
        )
        print("No. of test steps per epoch: {}".format(self.n_test_steps_per_epoch))
        print("")

    def load_image(self, image_file_path: str) -> np.ndarray:
        """Loads the image for the current image path.

        Loads the image for the current image path.

        Args:
            image_file_path: A string for the location where the image is located.

        Returns:
            A NumPy array for the image loaded from the file path.
        """
        # Checks type & values of arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path should be of type 'str'."

        # Loads the image for the current image path.
        image = skimage.io.imread(image_file_path)
        return image


dataset = Dataset({})
dataset.extract_data_from_zip_file()
dataset.load_dataset_file_paths()
