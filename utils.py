import os
import sys
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from huggingface_hub import hf_hub_download
from transformers import AutoModel
import inspect

# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

            
# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

def pil_to_input(pil_image):
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(pil_image).unsqueeze(0)
    return input

def compute_score(input1, input2, aligner, fr_model, return_bbox=False):
    # input1 = F.interpolate(input1, size=(112,112), mode='bilinear')
    # input2 = F.interpolate(input2, size=(112,112), mode='bilinear')
    aligned_x1, orig_pred_ldmks1, aligned_ldmks1, score1, thetas1, normalized_bbox1 = aligner(input1)
    aligned_x2, orig_pred_ldmks2, aligned_ldmks2, score2, thetas2, normalized_bbox2 = aligner(input2)

    # recognize
    input_signature = inspect.signature(fr_model.model.net.forward)
    if input_signature.parameters.get('keypoints') is not None:
        feat1 = fr_model(aligned_x1, aligned_ldmks1)
        feat2 = fr_model(aligned_x2, aligned_ldmks2)
    else:
        feat1 = fr_model(aligned_x1)
        feat2 = fr_model(aligned_x2)

    # compute cosine similarity
    cossim = torch.nn.functional.cosine_similarity(feat1, feat2)
    if return_bbox:
        return cossim, normalized_bbox1, normalized_bbox2
    return cossim