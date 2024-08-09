# Use the official TensorFlow Serving base image
FROM tensorflow/serving

# Sets the working directory
WORKDIR /

# Copies serialized model.
COPY models/flair_abnormality_segmentation/v1.0.0 /models/flair_abnormality_segmentation_v1.0.0/1

# Copies configs/models.config to container.
COPY configs/tf_serving/models.config /models/models.config

# Expose the gRPC and REST API ports
EXPOSE 8500
EXPOSE 8501

# Starts TensorFlow Serving when the container runs.
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_config_file=/models/models.config"]