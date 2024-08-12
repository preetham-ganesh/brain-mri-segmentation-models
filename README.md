# Brain MRI Segmentation - Models

This repository includes the code for training and serving deep learning models used in the Brain MRI Segmentation project. The models are designed to detect FLAIR abnormalities in MRI images.

## Contents

- [Related Repositories](https://github.com/preetham-ganesh/brain-mri-segmentation-models#related-repositories)
- [Installation](https://github.com/preetham-ganesh/brain-mri-segmentation-models#installation)
- [Usage](https://github.com/preetham-ganesh/brain-mri-segmentation-models#usage)
- [Model Details](https://github.com/preetham-ganesh/brain-mri-segmentation-models#model-details)
- [Releases](https://github.com/preetham-ganesh/brain-mri-segmentation-models#releases)
- [Support](https://github.com/preetham-ganesh/brain-mri-segmentation-models#support)

## Related Repositories

- [Brain MRI Segmentation - App](https://github.com/preetham-ganesh/brain-mri-segmentation-app)

## Installation

### Download the repository

```bash
git clone https://github.com/preetham-ganesh/brain-mri-segmentation-models.git
cd brain-mri-segmentation-models
```

### Requirements Installation

Requires: [Pip](https://pypi.org/project/pip/)

```bash
pip install --no-cache-dir -r requirements.txt
```

## Usage

Use the following commands to run the code files in the repo:

Note: All code files should be executed in home directory.

### Dataset

- The data was downloaded from Kaggle - Brain MRI segmentation [[Link]](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
- After downloading the data, 'archive.zip' file should be saved in the following data directory path 'data/raw_data/'

### Model Training & Testing

```bash
python3 src/flair_mri_segmentation/run.py --model_version 1.0.0 --set_tracking_uri <URI> --experiment_name flair_abnormality_segmentation
```

or

```bash
python3 src/flair_mri_segmentation/train.py -mv 1.0.0 -stu <URI> -en flair_abnormality_segmentation
```

### Predict

```bash
python3 src/flair_mri_segmentation/predict.py --model_version 1.0.0 --image_file_path <file_path>
```

or

```bash
python3 src/flair_mri_segmentation/predict.py -mv 1.0.0 --ifp <file_path>
```

### Download Model

```bash
python3 src/download_model.py --model_name flair_abnormality_segmentation --model_version 1.0.0 --s3_bucket_name <name> --s3_artifact_directory_path <directory_path>
```

or

```bash
python3 src/download_model.py --mn flair_abnormality_segmentation --mv 1.0.0 --s3bn <name> --s3adp <directory_path>
```

## Model Details

Details of the model

| Model name                     | Model Version | Description                                                                                                                                                             |
| ------------------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flair Abnormality Segmentation | v1.0.0        | - A U-Net model with MobileNetV2 pretrained on ImageNet as Encoder, and custom layers as decoder. <br/> - Model was trained on Kaggle - Brain MRI segmentation dataset. |

## Releases

Details about the latest releases, including key features, bug fixes, and any other relevant information.

| Version | Release Date | Release Notes                                                                                            |
| ------- | ------------ | -------------------------------------------------------------------------------------------------------- |
| v1.0.0  | 08-12-2024   | Releases Training & Model Serving code (Sub-classing approach) for Flair Abnormality Segmentation model. |

## Support

For any queries regarding the repository please contact 'preetham.ganesh2021@gmail.com'.
