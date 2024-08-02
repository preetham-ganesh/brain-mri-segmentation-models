import os
import zipfile

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
        extracted_data_directory_path = check_directory_path_existence(
            "data/extracted_data/lgg_mri_segmentation"
        )

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            "{}/kaggle_3m/data.csv".format(extracted_data_directory_path)
        ):

            # Extracts files from downloaded data zip file into a directory.
            try:
                with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                    zip_file.extractall(extracted_data_directory_path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    "{} does not exist. Download data from ".format(zip_file_path)
                    + "'https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation', and place it in "
                    + "'data/raw_data' as 'archive.zip'."
                )
            print(
                "Finished extracting files from 'archive.zip' to {}.".format(
                    extracted_data_directory_path
                )
            )
            print()
