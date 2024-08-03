import os

import tensorflow as tf

from src.utils import load_json_file
from src.flair_abnormality_segmentation.dataset import Dataset
from src.flair_abnormality_segmentation.model import UNet


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

    def load_model(self, mode: str) -> None:
        """Loads model & other utilies for training or testing.

        Loads model & other utilies for training or testing.

        Args:
            mode: A string for the mode by which the model should be loaded.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(mode, str), "Variable mode should be of type 'str'."
        assert (
            mode == "train" or mode == "predict"
        ), "Variable mode should 'train' or 'predict' as values."

        # Loads model for current model configuration.
        self.model = UNet(self.model_configuration)

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = "{}/models/{}/v{}/checkpoints".format(
            self.home_directory_path, self.model_name, self.model_version
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["learning_rate"]
        )
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.checkpoint_directory_path, max_to_keep=3
        )

        # If mode is predict, then the trained checkpoint is restored.
        if mode == "predict":
            checkpoint.restore(
                tf.train.latest_checkpoint(self.checkpoint_directory_path)
            )
        print("Finished loading model for {} configuration.".format(mode))
        print()
