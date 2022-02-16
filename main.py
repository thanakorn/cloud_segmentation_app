import os
import dvc.api
import torch
import io
import yaml
import cv2 as cv
import numpy as np
import tempfile
from dotenv import load_dotenv
from torch import sigmoid
from google.cloud import storage
from unet import UNet

load_dotenv()
input_bucket_name = os.getenv('INPUT_BUCKET_NAME')
output_bucket_name = os.getenv('OUTPUT_BUCKET_NAME')

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

    # with dvc.api.open(path=model_path, repo=repo_url, mode='rb') as f:
    #     buffer = io.BytesIO(f.read())
    #     state_dict = torch.load(buffer)
    state_dict = torch.load('/Users/thanakorn/git/cloud_segmentation/model/model.pth')
    
    model = UNet(n_classes=model_params['n_classes'], in_channel=model_params['in_channels'])
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image(filename):
    blob = input_bucket.get_blob(filename)
    _, temp_local_filename = tempfile.mkstemp(suffix='.jpg')
    blob.download_to_filename(temp_local_filename)
    img = cv.cvtColor(cv.imread(temp_local_filename), cv.COLOR_BGR2RGB)
    os.remove(temp_local_filename)
    return img

def save_image(img_data, filename):
    _, temp_local_filename = tempfile.mkstemp(suffix='.jpg')
    cv.imwrite(temp_local_filename, img_data)
    blob = output_bucket.blob(filename)
    blob.upload_from_filename(temp_local_filename)
    os.remove(temp_local_filename)

def inference(event, context):
    model = load_model()
    for obj in input_bucket.list_blobs():
        img = load_image(obj.name)
        x = torch.tensor(img.transpose(2,0,1)).unsqueeze(dim=0).float()
        out = sigmoid(model(x))
        out = (out.detach().numpy() > 0.5).astype(np.uint16) * 255
        save_image(out, obj.name)

if __name__=='__main__':
    inference(None, None)