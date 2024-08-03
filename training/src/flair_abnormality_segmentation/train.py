import os

from src.utils import load_json_file
from src.flair_abnormality_segmentation.dataset import Dataset


class Train(object):
    """Trains the segmentation model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = (
            "{}/configs/models/flair_abnormality_segmentation".format(
                self.home_directory_path
            )
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads the dataset based on dataset name and its version.

        Loads the dataset based on dataset name and its version.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Extracts files from downloaded data zip file.
        self.dataset.extract_data_from_zip_file()

        # Loads file paths of images & masks in the dataset.
        self.dataset.load_dataset_file_paths()

        # Splits dataset into train & test data splits. Creates object for KFold cross validation.
        self.dataset.split_dataset()
