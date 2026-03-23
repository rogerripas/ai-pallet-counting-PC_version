import argparse
from tqdm import tqdm
import os
import importlib
from pathlib import Path
import pickle
from PIL import Image, ImageDraw

import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as albu 
import torch




from skimage import  io
import glob
import matplotlib.pyplot as plt

devices = ["cpu" , "cuda"]
i_device = 1



def inference_image(model, images, device):
    images = images.to(device)
    predicted = model(images)
    masks = torch.sigmoid(predicted) 
    masks = masks.squeeze(1).cpu().detach().numpy()
    return masks


def inference_model(model, loader, device, use_flip):
    mask_dict = {}
    for image_ids, images in tqdm(loader):
        masks = inference_image(model, images, device)
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_masks = inference_image(model, flipped_imgs, device)
            flipped_masks = np.flip(flipped_masks, axis=2)
            masks = (masks + flipped_masks) / 2
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict
    
def inference_single_model(model, in_image):
    device = devices[i_device]
    image_ids, images = in_image        
    masks = inference_image(model, images, device)
     
    flipped_imgs = torch.flip(images, dims=(3,))
    flipped_masks = inference_image(model, flipped_imgs, device)
    flipped_masks = np.flip(flipped_masks, axis=2)
    
    masks = (masks + flipped_masks) / 2
       
    return masks[0].astype(np.float32)




def load_model_resnet(checkpoint_path):
    module = importlib.import_module('models.selim_zoo.unet')
    model_class = getattr(module, 'Resnet')
    model = model_class(**{'seg_classes': 1, 'backbone_arch': 'resnet18'}).to(devices[i_device])
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model
