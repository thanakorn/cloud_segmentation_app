from ast import mod
import dvc.api
import torch
import io
import yaml
import cv2 as cv
import numpy as np
import tempfile
from google.cloud import storage
from unet import UNet

input_bucket_name = 'cloud-segmentation-input'
output_bucket_name = 'cloud-segmentation-output'
model_path = 'model/model.pth'
model_config_path = 'params.yaml'
repo_url = 'https://github.com/thanakorn/cloud_segmentation'

storage_client = storage.Client()
input_bucket = storage_client.bucket(bucket_name=input_bucket_name)
output_bucket = storage_client.bucket(bucket_name=output_bucket_name)

def load_model():
    with dvc.api.open(path=model_config_path, repo=repo_url, mode='rb') as f:
        params = yaml.safe_load(io.BytesIO(f.read()))
        model_params = params['model']

    with dvc.api.open(path=model_path, repo=repo_url, mode='rb') as f:
        buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer)
    
    model = UNet(n_classes=model_params['n_classes'], in_channel=model_params['in_channels'])
    model.load_state_dict(state_dict)
    return model

def load_image(filename):
    blob = input_bucket.get_blob(filename)
    _, temp_local_filename = tempfile.mkstemp(suffix='.jpg')
    blob.download_to_filename(temp_local_filename)
    return cv.imread(temp_local_filename)

def save_image(img_data, filename):
    _, temp_local_filename = tempfile.mkstemp(suffix='.jpg')
    cv.imwrite(temp_local_filename, img_data)
    blob = output_bucket.blob(filename)
    blob.upload_from_filename(temp_local_filename)

if __name__=='__main__':
    for x in input_bucket.list_blobs():
        img = load_image(x.name)
        save_image(img, x.name)