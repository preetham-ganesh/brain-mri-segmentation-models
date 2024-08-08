import argparse


def main():
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

    #
    download_from_s3_bucket(
        args.model_name,
        args.model_version,
        args.s3_bucket_name,
        args.s3_artifact_directory_path,
    )


if __name__ == "__main__":
    main()
