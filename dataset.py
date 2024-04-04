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

        self.img_files = [fn for fn in sorted(os.listdir(self.img_dir))]
        
        self.img_transform = utils.get_image_transform(cfg, processor)
        self.box_transform = utils.get_box_transform(cfg)
    
    def _load_image(self, image_dir):
        return Image.open(image_dir).convert("RGB")
    
    def __getitem__(self, idx):
        img = self._load_image(os.path.join(self.img_dir, self.img_files[idx]))

        json_pth = os.path.join(self.json_dir, self.img_files[idx].replace(".jpg", ".json"))
        with open(json_pth, "r") as f:
            data = json.load(f)
            bbox = data["annotations"]["bbox"]
            weed_lbl = data["annotations"]["weeds_kind"]
            lbl = 0

            if (weed_lbl == 'N'):
                lbl = 0
            elif (weed_lbl == 'E'):
                lbl = 1
            elif (weed_lbl == 'A'):
                lbl = 2
            elif (weed_lbl == 'V'):
                lbl = 3
            elif (weed_lbl == 'I'):
                lbl = 4
            elif (weed_lbl == 'B'): 
                lbl = 5
            else: 
                lbl = 6
        
        img_processed = self.img_transform(img)
        
        
        return img_processed.pixel_values[0], img_processed.pixel_mask[0], \
            self.box_transform(bbox), lbl
    
    def __len__(self):
        return len(self.img_files)
    

class FasterRCNNDataset(Dataset):
    def __init__(self, cfg,
                 img_dir,
                 json_dir):
        
        self.img_dir = img_dir
        self.json_dir = json_dir

        self.img_files = [fn for fn in sorted(os.listdir(self.img_dir))]

        self.img_transform = utils.get_image_transform(cfg)
        self.box_transform = utils.get_box_transform(cfg)

        self.img_size = cfg.img_size

    def _load_image(self, image_dir):
        return Image.open(image_dir).convert("RGB")

    def __getitem__(self, idx):
        img = self._load_image(os.path.join(self.img_dir, self.img_files[idx]))

        json_pth = os.path.join(self.json_dir, self.img_files[idx].replace(".jpg", ".json"))
        with open(json_pth, "r") as f:
            data = json.load(f)
            bbox = data["annotations"]["bbox"]
            weed_lbl = data["annotations"]["weeds_kind"]
            lbl = 0

            if (weed_lbl == 'N'):
                lbl = 0
            elif (weed_lbl == 'E'):
                lbl = 1
            elif (weed_lbl == 'A'):
                lbl = 2
            elif (weed_lbl == 'V'):
                lbl = 3
            elif (weed_lbl == 'I'):
                lbl = 4
            elif (weed_lbl == 'B'): 
                lbl = 5
            else: 
                lbl = 6
        
        img_processed = self.img_transform(img)
        # box_processed = self.box_transform(bbox)
        lbl = torch.tensor(lbl, dtype=torch.int64)

        bbox = [bbox[0]/ self.img_size, bbox[1]/ self.img_size, (bbox[0] + bbox[2])/ self.img_size, (bbox[1] + bbox[3])/ self.img_size]
        box = []
        box.append(bbox)
        label = []
        label.append(lbl)

        box = torch.tensor(box, dtype=torch.float32)        
        label = torch.tensor(label, dtype=torch.int64)

        target = {}
        

        target["boxes"] = box
        target["labels"] = label
        
        return img_processed, target
    
    def __len__(self):
        return len(self.img_files)