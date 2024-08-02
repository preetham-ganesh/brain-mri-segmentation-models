import tensorflow as tf
from typing import Dict, Any, List


class UNet(tf.keras.Model):
    """A tensorflow model which segments location of an FLAIR abnormality in a brain MRI image."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the UNet model, by adding various layers.

        Initializes the layers in the UNet model, by adding various layers.

        Args:
            model_configuration: A dictionary for the configuration of the model.

        Returns:
            None.
        """
        super(UNet, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers arrangement in model configuration to add layers to the model.
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            config = self.model_configuration["model"]["layers"]["configuration"][name]

            # If layer's name is like 'mobilenet', the MobileNet model is initialized based on layer configuration.
            if name.startswith("mobilenet"):
                mobilenet = tf.keras.applications.MobileNetV2(
                    include_top=config["include_top"],
                    weights=config["weights"],
                    input_shape=(
                        self.model_configuration["model"]["final_image_height"],
                        self.model_configuration["model"]["final_image_width"],
                        self.model_configuration["model"]["n_channels"],
                    ),
                    alpha=config["alpha"],
                )

                # Filters the model based on output & input layer. Sets the trainable flag based on model configuration.
                output_layer_names = config["output_layer"]
                mobilenet.layers[0]._name = "input_image"
                mobilenet_outputs = [
                    mobilenet.get_layer(name).output for name in output_layer_names
                ]
                self.model_layers[name] = tf.keras.Model(
                    mobilenet.input, mobilenet_outputs
                )
                del mobilenet
                del mobilenet_outputs
                self.model_layers[name].trainable = config["trainable"]

            # If layer's name is like 'conv2d_', a Conv2D layer is initialized based on layer configuration.
            elif name.startswith("conv2d"):
                self.model_layers[name] = tf.keras.layers.Conv2D(
                    filters=config["filters"],
                    kernel_size=config["kernel"],
                    padding=config["padding"],
                    strides=config["strides"],
                    activation=config["activation"],
                    kernel_initializer=config["kernel_initializer"],
                )

            # If layer's name is like 'maxpool2d_', a MaxPool2D layer is initialized based on layer configuration.
            elif name.startswith("maxpool2d"):
                self.model_layers[name] = tf.keras.layers.MaxPool2D(
                    pool_size=config["pool_size"]
                )

            # If layer's name is like 'upsample2d_', an UpSample2D layer is initialized based on layer configuration.
            elif name.startswith("upsample2d"):
                self.model_layers[name] = tf.keras.layers.UpSampling2D(
                    size=config["size"]
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.startswith("dropout"):
                self.model_layers[name] = tf.keras.layers.Dropout(rate=config["rate"])

            # If layer's name is like 'activation_', an Activation layer is initialized based on layer configuration.
            elif name.startswith("activation"):
                self.model_layers[name] = tf.keras.layers.Activation(
                    activation=config["activation"]
                )

            # If layer's name is like 'batchnorm_', a BatchNorm layer is initialized based on layer configuration.
            elif name.startswith("batchnorm"):
                self.model_layers[name] = tf.keras.layers.BatchNormalization()

            # If layer's name is like 'concatenate_', a Concatenate layer is initialized based on layer configuration.
            elif name.startswith("concat"):
                self.model_layers[name] = tf.keras.layers.Concatenate(
                    axis=config["axis"]
                )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        """Inputs & masks tensors are passed through the layers in the model.

        Inputs & masks tensors are passed through the layers in the model.

        Args:
            inputs: A list of tensors for inputs to the model.
            training: A boolean value for the flag whether model is in 'train' or 'predict' mode.
            masks: A list of tensors for masks to the model.

        Returns:
            A tensor for the output predicted by the model.
        """
        x = inputs[0]

        # If the first layer is mobilenet, then the features are extracted.
        if self.model_configuration["model"]["layers"]["arrangement"][0] == "mobilenet":
            mobilenet_output_names = self.model_configuration["model"]["layers"][
                "configuration"
            ]["mobilenet"]["output_layer"]
            mobilenet_outputs = self.model_layers["mobilenet"](x, training=training)
            x = mobilenet_outputs[0]
            mobilenet_outputs = {
                name: output
                for name, output in zip(
                    mobilenet_output_names[1:], mobilenet_outputs[1:]
                )
            }

        # Iterates across the layers arrangement, and predicts the output for each layer.
        for name in self.model_configuration["model"]["layers"]["arrangement"][1:]:
            # If layer's name is like 'dropout_' or 'batchnorm_' or 'mobilenet_', the following output is predicted.
            if name.startswith("dropout") or name.startswith("batchnorm"):
                x = self.model_layers[name](x, training=training)

            # If layer's name is like 'concat_', the following output is predicted.
            elif name.startswith("concat"):
                input_name = self.model_configuration["model"]["layers"][
                    "configuration"
                ][name]["input"][0]

                # If the pretrained flag is true, then the skipped connection layer is imported from pretrained model.
                if self.model_configuration["model"]["layers"]["configuration"][name][
                    "pretrained"
                ]:
                    x = self.model_layers[name]([x, mobilenet_outputs[input_name]])

                # Else, skipped connection layer from current model is imported.
                else:
                    x = self.model_layers[name]([x, self.model_layers[input_name]])

            # Else, the following output is predicted.
            else:
                x = self.model_layers[name](x)
        return [x]

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            None.

        Returns:
            A tensorflow model based on the model configuration.
        """
        # Creates the input layer using the model configuration.
        inputs = [
            tf.keras.layers.Input(
                shape=(
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        ]

        # Creates an object for the tensorflow model and returns it.
        return tf.keras.Model(
            inputs=inputs, outputs=self.call(inputs, training=False, masks=None)
        )
