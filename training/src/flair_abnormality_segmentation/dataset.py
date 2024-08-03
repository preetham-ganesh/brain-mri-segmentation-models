import os
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.data import from_pandas
import tensorflow as tf
import skimage
import numpy as np

from src.utils import check_directory_path_existence

from typing import Dict, List, Any


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
                            "image_file_paths": image_file_path,
                            "mask_file_paths": mask_file_path,
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
            random_state=0,
        )
        (
            self.train_file_paths,
            self.validation_file_paths,
        ) = train_test_split(
            self.train_file_paths,
            test_size=self.model_configuration["dataset"]["split_percentage"][
                "validation"
            ],
            shuffle=True,
            random_state=0,
        )

        # Logs train, validation & test datasets to mlflow server as inputs.
        mlflow.log_input(
            from_pandas(
                self.train_file_paths,
                name="{}_v{}_train".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )
        mlflow.log_input(
            from_pandas(
                self.validation_file_paths,
                name="{}_v{}_validation".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )
        mlflow.log_input(
            from_pandas(
                self.test_file_paths,
                name="{}_v{}_test".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )

        # Computes no. of examples per data split.
        self.n_train_examples = len(self.train_file_paths)
        self.n_validation_examples = len(self.validation_file_paths)
        self.n_test_examples = len(self.test_file_paths)
        print("No. of examples in training dataset: {}".format(self.n_train_examples))
        print(
            "No. of examples in validation dataset: {}".format(
                self.n_validation_examples
            )
        )
        print("No. of examples in test dataset: {}".format(self.n_test_examples))
        print()

    def shuffle_slice_datasets(self) -> None:
        """Converts images & masks file paths into tensorflow dataset.

        Converts images & masks file paths into tensorflow dataset & slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips images & annotations file paths into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.train_file_paths["image_file_paths"]),
                list(self.train_file_paths["mask_file_paths"]),
            )
        ).shuffle(self.n_train_examples)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.validation_file_paths["image_file_paths"]),
                list(self.validation_file_paths["mask_file_paths"]),
            )
        ).shuffle(self.n_validation_examples)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.test_file_paths["image_file_paths"]),
                list(self.test_file_paths["mask_file_paths"]),
            )
        ).shuffle(self.n_test_examples)

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = self.n_train_examples // self.batch_size
        self.n_validation_steps_per_epoch = (
            self.n_validation_examples // self.batch_size
        )
        self.n_test_steps_per_epoch = self.n_test_examples // self.batch_size
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

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resizes image based on model configuration.

        Resizes image to (final_image_height, final_image_width, n_channels) shape.

        Args:
            image: A NumPy array for the image.

        Returns:
            A NumPy array for the resized version of the image.
        """
        # Checks type & values of arguments.
        assert isinstance(
            image, np.ndarray
        ), "Variable image should be of type 'numpy.ndarray'."

        # Resizes image to (final_image_height, final_image_width, n_channels). n_channels = 1 for mask & for input.
        resized_image = skimage.transform.resize(
            image,
            output_shape=(
                self.model_configuration["model"]["final_image_height"],
                self.model_configuration["model"]["final_image_width"],
            ),
        )
        return resized_image

    def threshold_image(self, image: np.ndarray) -> np.ndarray:
        """Thresholds image to have better distinction of regions in image.

        Thresholds image to have better distinction of regions in image.

        Args:
            image: A NumPy array for the image.

        Returns:
            A NumPy array for the thresholded version of the image.
        """
        # Checks type & values of arguments.
        assert isinstance(
            image, np.ndarray
        ), "Variable image should be of type 'numpy.ndarray'."

        # Thresholds image to have better distinction of regions in image.
        thresholded_image = np.where(
            image > self.model_configuration["model"]["threshold"], 255, 0
        )
        return thresholded_image

    def load_input_target_images(
        self, image_file_paths: List[str], mask_file_paths: List[str]
    ) -> List[tf.Tensor]:
        """Loads input & mask images for current batch as tensors.

        Loads input & mask images for current batch as tensors.

        Args:
            image_file_paths: A list of strings for locations of images in current batch.
            mask_file_paths: A list of strings for locations of masks in current batch.

        Returns:
            A list of tensors for input & target batch of images.
        """
        # Checks types & values of arguments.
        assert isinstance(
            image_file_paths, list
        ), "Variable images_file_paths should be of type 'list'."
        assert isinstance(
            mask_file_paths, list
        ), "Variable mask_file_paths should be of type 'str'."

        # Zero array for input batch of shape (batch, height, width, 3), & target batch of shape (batch, height, width).
        input_batch = np.zeros(
            shape=(
                len(image_file_paths),
                self.model_configuration["model"]["final_image_height"],
                self.model_configuration["model"]["final_image_width"],
                self.model_configuration["model"]["n_channels"],
            ),
            dtype=np.float32,
        )
        target_batch = np.zeros(
            shape=(
                len(image_file_paths),
                self.model_configuration["model"]["final_image_height"],
                self.model_configuration["model"]["final_image_width"],
            )
        )

        # Iterates across images & annotations file paths in current batch.
        for id_0 in range(len(image_file_paths)):
            # Loads the image & mask for the current image paths.
            input_image = self.load_image(str(image_file_paths[id_0], "UTF-8"))
            target_image = self.load_image(str(mask_file_paths[id_0], "UTF-8"))

            # Resizes image & mask based on model configuration.
            input_image = self.resize_image(input_image)
            target_image = self.resize_image(target_image)

            # Thresholds image to have better distinction of regions in image.
            input_image = self.threshold_image(input_image)

            # Adds loaded & preprocessed input & target images to corresponding batch arrays.
            input_batch[id_0, :, :, :] = input_image
            target_batch[id_0, :, :] = target_image

        # Converts input & target batch lists into tensors.
        input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
        target_batch = tf.convert_to_tensor(target_batch, dtype=tf.float32)

        # Normalizes the input batches from [0, 255] to [0, 1] range
        input_batch = input_batch / 255.0
        target_batch = target_batch / 255.0

        # Adds an extra dimension to the target batch.
        target_batch = tf.expand_dims(target_batch, axis=-1)
        return [input_batch, target_batch]
