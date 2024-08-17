# Use the official TensorFlow Serving base image
FROM tensorflow/serving:2.8.0

# Sets the working directory
WORKDIR /

# Copies serialized model.
COPY models/flair_abnormality_segmentation/v1.0.0/model /models/flair_abnormality_segmentation_v1.0.0/1

# Copies configs/models.config to container.
COPY configs/tf_serving/models.config /models/models.config

# Copies entrypoint shell file to container.
COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

# Adds permissions to the entrypoint shell file.
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT []

# Starts TensorFlow Serving when the container runs.
CMD ["/usr/bin/tf_serving_entrypoint.sh"]