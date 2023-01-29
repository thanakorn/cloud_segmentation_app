# Cloud segmentation pipeline

This project demonstrates how to build an ML application on GCP by building an ML pipeline that performs batch inference on images uploaded into a Google Cloud Storage bucket.

## Model
The model used in this project is UNet, a fully convolutional neural network designed for solving image segmentation tasks, which is trained to identify a pixel of cloud in an image. See [this repository](https://github.com/thanakorn/cloud_segmentation) for the model development part.

## Cloud Functions
The model is wrapped with CLoud Functions, a serverless compute service provided by Google Cloud Platform. When the function is triggered by the scheduler, it'll download the model from the model registry(managed by [DVC](https://dvc.org/)), download images from Google Cloud Storage, and run inference to generate the result. Finally, the output will be uploaded back to Google Cloud Storage bucket(The code for Cloud Functions is on `main.py`).
