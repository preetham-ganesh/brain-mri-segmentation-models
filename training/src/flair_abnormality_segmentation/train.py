import os

import tensorflow as tf
import mlflow

from src.utils import load_json_file, check_directory_path_existence
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

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(plot, bool), "Variable plot should be of type 'bool'."

        # Builds plottable graph for the model.
        model = self.model.build_graph()

        # Compiles the model to log the model summary.
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        mlflow.log_text(
            model_summary,
            "v{}/summary.txt".format(self.model_configuration["version"]),
        )

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/v{}/reports".format(self.model_version)
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                model,
                "{}/model_plot.png".format(self.reports_directory_path),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                "{}/model_plot.png".format(self.reports_directory_path),
                "v{}".format(self.model_configuration["version"]),
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_dice = tf.keras.metrics.Mean(name="train_dice_coefficient")
        self.validation_dice = tf.keras.metrics.Mean(name="validation_dice_coefficient")
        self.train_iou = tf.keras.metrics.Mean(name="train_iou")
        self.validation_iou = tf.keras.metrics.Mean(name="validation_iou")

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the loss computed on comparing target & predicted batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for current target & predicted batches.
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        current_loss = loss_object(target_batch, predicted_batch)
        return current_loss

    def compute_dice_coefficient(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes dice coefficient for the current batch using actual & predicted values.

        Computes dice coefficient for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the dice coefficient computed on comparing target & predicted batch.
        """
        # Flattens the target and predicted batches.
        target_batch = tf.keras.layers.Flatten()(target_batch)
        predicted_batch = tf.keras.layers.Flatten()(predicted_batch)

        # Computes the intersection between the flattened target and predicted batches.
        intersection = tf.reduce_sum(target_batch * predicted_batch)
        smooth = 1e-15
        return (2.0 * intersection + smooth) / (
            tf.reduce_sum(target_batch) + tf.reduce_sum(predicted_batch) + smooth
        )

    def compute_intersection_over_union(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes IoU for the current batch using actual & predicted values.

        Computes IoU for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the IoU computed on comparing target & predicted batch.
        """
        # Flattens the target and predicted batches.
        target_batch = tf.keras.layers.Flatten()(target_batch)
        predicted_batch = tf.keras.layers.Flatten()(predicted_batch)

        # Computes intersection & union for the target and predicted batch.
        intersection = tf.reduce_sum(target_batch * predicted_batch)
        union = (
            tf.reduce_sum(target_batch) + tf.reduce_sum(predicted_batch) - intersection
        )
        # Computes Intersection over Union metric.
        smooth = 1e-15
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains the model using input & target batches.

        Trains the model using input & target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes masked images for all input images in the batch, and computes batch loss.
        with tf.GradientTape() as tape:
            predicted_batch = self.model([input_batch], training=True, masks=None)[0]
            batch_loss = self.compute_loss(target_batch, predicted_batch)

        # Computes gradients using loss. Apply the computed gradients on model variables using optimizer.
        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes dice coefficient and iou score for the current batch.
        batch_dice = self.compute_dice_coefficient(target_batch, predicted_batch)
        batch_iou = self.compute_intersection_over_union(target_batch, predicted_batch)

        # Computes mean for loss, dice coefficient and iou score.
        self.train_loss(batch_loss)
        self.train_dice(batch_dice)
        self.train_iou(batch_iou)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates the model using input and target batches.

        Validates the model using input and target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes masked images for all input images in the batch.
        predicted_batch = self.model([input_batch], training=False, masks=None)[0]

        # Computes loss, dice coefficient & IoU for the target batch and predicted batch.
        batch_loss = self.compute_loss(target_batch, predicted_batch)
        batch_dice = self.compute_dice_coefficient(target_batch, predicted_batch)
        batch_iou = self.compute_intersection_over_union(target_batch, predicted_batch)

        # Computes mean for loss & accuracy.
        self.validation_loss(batch_loss)
        self.validation_dice(batch_dice)
        self.validation_iou(batch_iou)
