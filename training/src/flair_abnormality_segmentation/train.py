import os
import time

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
            "{}/configs/flair_abnormality_segmentation".format(self.home_directory_path)
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

        # Converts images & masks file paths into tensorflow dataset.
        self.dataset.shuffle_slice_datasets()

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
        if mode == "train":
            self.model = UNet(self.model_configuration)

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = (
            "{}/models/flair_abnormality_segmentation/v{}/checkpoints".format(
                self.home_directory_path, self.model_version
            )
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
            "flair_abnormality_segmentation/v{}/summary.txt".format(
                self.model_configuration["version"]
            ),
        )

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/flair_abnormality_segmentation/v{}/reports".format(
                self.model_version
            )
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
                "flair_abnormality_segmentation/v{}".format(
                    self.model_configuration["version"]
                ),
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

    def reset_metrics_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()
        self.train_dice.reset_state()
        self.validation_dice.reset_state()
        self.train_iou.reset_state()
        self.validation_iou.reset_state()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (image_file_paths, mask_file_paths) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads input & target batch images for file paths in current batch.
            input_batch, target_batch = self.dataset.load_input_target_images(
                list(image_file_paths.numpy()), list(mask_file_paths.numpy())
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()

            print(
                "Epoch={}, Batch={}, Train loss={}, Train dice coefficient={}, Train IoU={}, Time taken={} "
                "sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.train_dice.result().numpy(), 3)),
                    str(round(self.train_iou.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "train_loss": self.train_loss.result().numpy(),
                "train_dice_coefficient": self.train_dice.result().numpy(),
                "train_iou": self.train_iou.result().numpy(),
            },
            step=epoch,
        )
        print("")

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using the current validation dataset.

        Validates the model using the current validation dataset.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the validation dataset.
        for batch, (image_file_paths, mask_file_paths) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads input & target batch images for file paths in current batch.
            input_batch, target_batch = self.dataset.load_input_target_images(
                list(image_file_paths.numpy()), list(mask_file_paths.numpy())
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()

            print(
                "Epoch={}, Batch={}, Validation loss={}, Validation dice coefficient={}, Validation IoU={}, "
                "Time taken={} sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.validation_dice.result().numpy(), 3)),
                    str(round(self.validation_iou.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "validation_loss": self.train_loss.result().numpy(),
                "validation_dice_coefficient": self.train_dice.result().numpy(),
                "validation_iou": self.train_iou.result().numpy(),
            },
            step=epoch,
        )
        print("")

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current epoch.

        Saves the model after checking performance metrics in current epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print("Checkpoint saved at {}.".format(self.checkpoint_directory_path))

    def early_stopping(self) -> bool:
        """Stops the model from learning further if performance has not improved.

        Stops the model from learning further if the performance has not improved from previous epoch.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > str(
            round(self.validation_loss.result().numpy(), 3)
        ):
            self.patience_count = 0
            print(
                "Best validation loss changed from {} to {}".format(
                    str(self.best_validation_loss),
                    str(round(self.validation_loss.result().numpy(), 3)),
                )
            )
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < self.model_configuration["model"]["patience_count"]:
            self.patience_count += 1
            print("Best validation loss did not improve.")
            print("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for epoch in range(self.model_configuration["model"]["epochs"]):
            epoch_start_time = time.time()

            # Resets states for trackers before the start of each epoch.
            self.reset_metrics_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)

            epoch_end_time = time.time()
            print(
                "Epoch={}, Train loss={}, Validation loss={}, Train dice coefficient={}, "
                "Validation dice coefficient={}, Training IoU={}, Validation IoU={}, Time taken={} sec.".format(
                    epoch + 1,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.train_dice.result().numpy(), 3)),
                    str(round(self.validation_dice.result().numpy(), 3)),
                    str(round(self.train_iou.result().numpy(), 3)),
                    str(round(self.validation_iou.result().numpy(), 3)),
                    round(epoch_end_time - epoch_start_time, 3),
                )
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                print(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                print()
                break
            print()

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_metrics_trackers()

        # Iterates across batches in the validation dataset.
        for batch, (image_file_paths, mask_file_paths) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            # Loads input & target batch images for file paths in current batch.
            input_batch, target_batch = self.dataset.load_input_target_images(
                list(image_file_paths.numpy()), list(mask_file_paths.numpy())
            )

            # Tests the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)

        # Logs test metrics for current epoch.
        mlflow.log_metrics(
            {
                "test_loss": self.validation_loss.result().numpy(),
                "test_dice_coefficient": self.validation_dice.result().numpy(),
                "test_iou": self.validation_iou.result().numpy(),
            }
        )

        print(
            "Test loss={}.".format(str(round(self.validation_loss.result().numpy(), 3)))
        )
        print(
            "Test dice coefficient={}.".format(
                str(round(self.validation_dice.result().numpy(), 3))
            )
        )
        print(
            "Test IoU={}.".format(str(round(self.validation_iou.result().numpy(), 3)))
        )
        print("")

    def serialize_model(self) -> None:
        """Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Args:
            None.

        Returns:
            None.
        """
        # Defines input shape for exported model's input signature.
        input_shape = [
            None,
            self.model_configuration["model"]["final_image_height"],
            self.model_configuration["model"]["final_image_width"],
            self.model_configuration["model"]["n_channels"],
        ]

        class ExportModel(tf.Module):
            """Exports trained tensorflow model as tensorflow module for serving."""

            def __init__(self, model: tf.keras.Model) -> None:
                """Initializes the variables in the class.

                    Initializes the variables in the class.

                Args:
                    model: A tensorflow model for the model trained with latest checkpoints.

                Returns:
                    None.
                """
                # Asserts type of input arguments.
                assert isinstance(
                    model, tf.keras.Model
                ), "Variable model should be of type 'tensorflow.keras.Model'."

                # Initializes class variables.
                self.model = model

            @tf.function(
                input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
            )
            def predict(self, images: tf.Tensor) -> tf.Tensor:
                """Input image is passed through the model for prediction.

                Input image is passed through the model for prediction.

                Args:
                    images: A tensor for the processed image for which the model should predict the result.

                Return:
                    An integer for the number predicted by the model for the current image.
                """
                prediction = self.model([images], training=False, masks=None)
                return prediction

        # Exports trained tensorflow model as tensorflow module for serving.
        exported_model = ExportModel(self.model)

        # Predicts output for the sample input using the Exported model.
        output_0 = exported_model.predict(
            tf.ones(
                (
                    10,
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        )

        # Saves the tensorflow object created from the loaded model.
        home_directory_path = os.getcwd()
        tf.saved_model.save(
            exported_model,
            "{}/models/flair_abnormality_segmentation/v{}/serialized".format(
                home_directory_path, self.model_version
            ),
        )

        # Loads the serialized model to check if the loaded model is callable.
        exported_model = tf.saved_model.load(
            "{}/models/flair_abnormality_segmentation/v{}/serialized".format(
                home_directory_path, self.model_version
            ),
        )
        output_1 = exported_model.predict(
            tf.ones(
                (
                    10,
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        )

        # Checks if the shape between output from saved & loaded models matches.
        assert (
            output_0[0].shape == output_1[0].shape
        ), "Shape does not match between the output from saved & loaded models."
        print("Finished serializing model & configuration files.")
        print()

        # Logs serialized model as artifact.
        mlflow.log_artifacts(
            "{}/models/flair_abnormality_segmentation/v{}/serialized".format(
                home_directory_path, self.model_version
            ),
            "flair_abnormality_segmentation/v{}/model".format(
                self.model_configuration["version"]
            ),
        )

        # Logs updated model configuration as artifact.
        mlflow.log_dict(
            self.model_configuration,
            "flair_abnormality_segmentation/v{}/model_configuration.json".format(
                self.model_version
            ),
        )
