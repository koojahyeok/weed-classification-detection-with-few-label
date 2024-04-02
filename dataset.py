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
                 pos_img_dir,
                 neg_img_dir,
                 json_dir,
                 processor):
        self.pos_img_dir = pos_img_dir
        self.neg_img_dir = neg_img_dir 
        self.json_dir = json_dir
        
        with open(self.json_dir, "r") as f:
            self.boxes = json.load(f)
        self.boxes = {k.split("/")[-1]:v for k, v in self.boxes.items()}
            
        self.pos_img_files = [fn for fn in os.listdir(self.pos_img_dir) if ".json" not in fn]
        self.neg_img_files = [fn for fn in os.listdir(self.neg_img_dir) if ".json" not in fn]
        self.pos_img_files.sort()
        self.neg_img_files.sort()
        self.img_files = self.pos_img_files + self.neg_img_files
        self.labels = [0 for _ in range(len(self.pos_img_files))] + [1 for _ in range(len(self.neg_img_files))]
        
        self.img_transform = utils.get_image_transform(cfg, processor)
        self.box_transform = utils.get_box_transform(cfg)
    
    def _load_image(self, image_dir):
        try:
            return Image.open(image_dir).convert("RGB")
        except:
            return Image.open("/data/jaeyeong/bind_corp/table_ocr/table_detection/synthesized_train_images/00016_840_219.png").convert("RGB")
    
    def __getitem__(self, idx):
        if self.labels[idx] == 0:
            img = self._load_image(os.path.join(self.pos_img_dir, self.img_files[idx]))
            box = self.boxes[self.img_files[idx]]
        else:
            img = self._load_image(os.path.join(self.neg_img_dir, self.img_files[idx]))
            box = [0, 0, 0, 0]
        img_processed = self.img_transform(img)
        label = self.labels[idx]
        
        return img_processed.pixel_values[0], img_processed.pixel_mask[0], \
            self.box_transform(box), label
    
    def __len__(self):
        return len(self.img_files)
    
