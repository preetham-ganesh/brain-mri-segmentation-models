# Brain MRI Segmentation - Models

This repository includes the code for training and serving deep learning models used in the Brain MRI Segmentation project. The models are designed to detect FLAIR abnormalities in MRI images.

## Contents

- [Installation](https://github.com/preetham-ganesh/brain-mri-segmentation-models#installation)
- [Usage](https://github.com/preetham-ganesh/brain-mri-segmentation-models#usage)
- [Model Versions](https://github.com/preetham-ganesh/brain-mri-segmentation-models#model-versions)
- [Releases](https://github.com/preetham-ganesh/brain-mri-segmentation-models#releases)

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
