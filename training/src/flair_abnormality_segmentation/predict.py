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
