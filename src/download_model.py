import os
import argparse
import boto3
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)


from src.utils import check_directory_path_existence


def download_model_from_s3_bucket(
    model_name: str,
    model_version: str,
    s3_bucket_name: str,
    s3_artifact_directory_path: str,
) -> None:
    """Downloads files the serialized model from S3 bucket individually.

    Downloads files the serialized model from S3 bucket individually.

    Args:
        model_name: A string for name of the model in the S3 bucket.
        model_version: A string for the version of the model in the S3 bucket.
        s3_bucket_name: A string for the name of the bucket where the serialized model is stored.
        s3_artifact_directory_path: A string for the URI of the artifact directory path in S3 bucket.

    Returns:
        None.
    """
    # Creates the absolute path for the directory path given in argument if it does not already exist.
    serialized_model_directory_path = "models/{}/v{}/model".format(
        model_name, model_version
    )
    _ = check_directory_path_existence(serialized_model_directory_path)

    # Initializes S3 client
    s3 = boto3.client("s3")

    # Lists objects in the specified S3 directory.
    s3_objects = s3.list_objects_v2(
        Bucket=s3_bucket_name, Prefix=s3_artifact_directory_path
    )

    # Iterates across objects in the S3 directory.
    for obj in s3_objects.get("Contents", []):
        object_key = obj["Key"]

        # Skips if object is a directory.
        if object_key.endswith("/"):
            continue

        # Creates a relative file path for object.
        object_file_path = os.path.join(
            serialized_model_directory_path,
            os.path.relpath(object_key, s3_artifact_directory_path),
        )

        # Extracts directory name of object file path. Create the directory if it does not exist.
        object_sub_directory_path = os.path.dirname(object_file_path)
        _ = check_directory_path_existence(object_sub_directory_path)

        # Download the file
        s3.download_file(s3_bucket_name, object_key, object_file_path)
        print("Finished downloading {} to {}.".format(object_key, object_file_path))
    print()


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s3bn",
        "--s3_bucket_name",
        type=str,
        required=True,
        help="Name of the S3 bucket where the artifacts are located.",
    )
    parser.add_argument(
        "-s3adp",
        "--s3_artifact_directory_path",
        type=str,
        required=True,
        help="Location where the serialized model on S3 bucket.",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        required=True,
        help="Name of the model which will be downloaded from the S3 bucket.",
    )
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version of the model which will be downloaded from the S3 bucket.",
    )
    args = parser.parse_args()

    # Downloads files the serialized model from S3 bucket individually.
    download_model_from_s3_bucket(
        args.model_name,
        args.model_version,
        args.s3_bucket_name,
        args.s3_artifact_directory_path,
    )


if __name__ == "__main__":
    main()
