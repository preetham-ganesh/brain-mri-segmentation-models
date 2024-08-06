import os
import sys
import logging
import warnings
import argparse


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import tensorflow as tf
import numpy as np

from src.utils import load_json_file
from src.flair_abnormality_segmentation.dataset import Dataset


class PredictMask(object):
    """Predicts mask for FLAIR abnormality in brain MRI images."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the PredictMask class.

        Creates object attributes for the PredictMask class.

        Args:
            model_version: A string for the version of the model should be used for prediction.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version

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
            "{}/configs/flair_abnormality_segmentation".format(self.home_directory_path)
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_model(self) -> None:
        """Loads model & other utilities for prediction.

        Loads model & other utilities for prediction.

        Args:
            None.

        Returns:
            None.
        """
        # Loads the tensorflow serialized model using model name & version.
        self.model = tf.saved_model.load(
            "{}/models/{}/v{}/serialized".format(
                self.home_directory_path, self.model_name, self.model_version
            ),
        )

        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

    def load_preprocess_input_image(self, image_file_path: str) -> np.ndarray:
        """Loads & preprocesses image based on segmentation model requirements.

        Loads & preprocesses image based on segmentation model requirements.

        Args:
            image_file_path: A string for the location of the image.

        Returns:
            A NumPy array for the processed image as input to the model.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path should be of type 'str'."

        # Loads the image for the current image path.
        image = self.dataset.load_image(image_file_path)

        # Thresholds image to have better distinction of regions in image.
        thresholded_image = self.dataset.threshold_image(image)

        # Adds an extra dimension to the image.
        model_input_image = np.expand_dims(thresholded_image, axis=0)

        # Casts input image to float32 and normalizes the image from [0, 255] range to [0, 1] range.
        model_input_image = np.float32(model_input_image)
        model_input_image = model_input_image / 255.0
        return model_input_image

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Converts the prediction from the segmentation model output into an image.

        Converts the prediction from the segmentation model output into an image.

        Args:
            prediction: A NumPy array for the prediction output from the model for the input image.

        Returns:
            A NumPy array for the processed version of prediction output.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            prediction, np.ndarray
        ), "Variable prediction should be of type 'np.ndarray'."

        # Removes 0th and 2nd dimension from the predicted image.
        predicted_image = np.squeeze(prediction, axis=0)
        predicted_image = np.squeeze(predicted_image, axis=-1)

        # De-normalizes predicted image from [0, 1] to [0, 255], and type casts it to uint8.
        predicted_image *= 255.0
        predicted_image = predicted_image.astype(np.uint8)

        # Thresholds the predicted image to convert into black & white image.
        predicted_image = self.dataset.threshold_image(predicted_image)
        return predicted_image
