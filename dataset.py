from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

import os
import torch
import json
import numpy as np

import utils

def build_dataset():
    pass

class DETRDataset(Dataset):
    def __init__(self, cfg,
                 img_dir,
                 json_dir,
                 processor):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.json_names = sorted(os.listdir(json_dir))
        
        # with open(self.json_dir, "r") as f:
        #     self.boxes = json.load(f)
        # self.boxes = {k.split("/")[-1]:v for k, v in self.boxes.items()}

        self.boxes = []
        self.labels = []

        for json_name in sorted(os.listdir(self.json_dir)):
            path = os.path.join(self.json_dir, json_name)

            with open(path, "r") as f:
                data = json.load(f)
                bbox = data["annotations"]["bbox"]

                weed_lbl = data["annoations"]["weeds_kind"]
                lbl = 0

                if (weed_lbl == 'N'):
                    label = 0
                elif (weed_lbl == 'E'):
                    label = 1
                elif (weed_lbl == 'A'):
                    label = 2
                elif (weed_lbl == 'V'):
                    label = 3
                elif (weed_lbl == 'I'):
                    label = 4
                elif (weed_lbl == 'B'): 
                    label = 5
                else: 
                    label = 6
            
            self.boxes.append(bbox)
            self.labels.append(label)

            
        self.img_files = [fn for fn in sorted(os.listdir(self.img_dir))]
        
        self.img_transform = utils.get_image_transform(cfg, processor)
        self.box_transform = utils.get_box_transform(cfg)
    
    def _load_image(self, image_dir):
        return Image.open(image_dir).convert("RGB")
    
    def __getitem__(self, idx):
        img = self._load_image(os.path.join(self.pos_img_dir, self.img_files[idx]))
        box = self.boxes[idx]
        img_processed = self.img_transform(img)
        label = self.labels[idx]
        
        return img_processed.pixel_values[0], img_processed.pixel_mask[0], \
            self.box_transform(box), label
    
    def __len__(self):
        return len(self.img_files)
    
