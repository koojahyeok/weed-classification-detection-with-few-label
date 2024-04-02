from torch.utils.data import DataLoader

import os
import torch
import yaml
import wandb
import argparse

from models import table_transformer
import dataset
import utils
import train
import test

def main(cfg):
    model, processor = table_transformer.build_model()
    train_dset = dataset.DETRDataset(cfg, cfg.train_pos_img_dir,
                                     cfg.train_neg_img_dir,
                                     cfg.train_json_dir,
                                     processor)
    val_dset = dataset.DETRDataset(cfg, cfg.val_pos_img_dir,
                                     cfg.val_neg_img_dir,
                                     cfg.val_json_dir,
                                     processor)

    train_dl = DataLoader(train_dset,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=cfg.num_workers)
    val_dl = DataLoader(val_dset,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=cfg.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     cfg.milestones,
                                                     cfg.gamma,
                                                     last_epoch=cfg.last_epoch, 
                                                     verbose=cfg.scheduler_verbosity)
    
    model = torch.nn.DataParallel(model)
    
    if cfg.continual_training:
        ckpt = torch.load(cfg.continual_training)
        model.load_state_dict(ckpt['weight'])
        print("state dict loaded successfully.")
    
    train.train(cfg, model, processor, train_dl, val_dl, optimizer, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=str)
    args = parser.parse_args()
    
    with open(args.config_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    wandb.init(project="Table Detection DETR MS finetuning")
    
    wandb.run.name = "0116_mixed"
    wandb.run.save()
    wandb.config.update(cfg)    
    
    cfg = type('cfg', (), cfg)
    
    main(cfg)
    