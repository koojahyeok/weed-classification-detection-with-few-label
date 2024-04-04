from torch.utils.data import DataLoader

import os
import torch
import yaml
import wandb
import argparse

from models import table_transformer, fasterRCNN
import dataset
import utils
import train
import test

def main(cfg, device):

    if (cfg.model == "frcnn"):
        model = fasterRCNN.get_model_instance_segmentation(cfg)
        train_dset = dataset.FasterRCNNDataset(cfg, cfg.train_img_dir,
                                               cfg.train_json_dir)
        
        val_dset = dataset.FasterRCNNDataset(cfg, cfg.val_img_dir,
                                             cfg.val_json_dir)
        
        train_dl = DataLoader(train_dset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            collate_fn=utils.collate_fn,
                            num_workers=cfg.num_workers)
        
        val_dl = DataLoader(val_dset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            collate_fn=utils.collate_fn,
                            num_workers=cfg.num_workers)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        cfg.milestones,
                                                        cfg.gamma,
                                                        last_epoch=cfg.last_epoch, 
                                                        verbose=cfg.scheduler_verbosity)
        
        train.frcnn_train(cfg, model, train_dl, val_dl, scheduler, device)


    elif (cfg.model == "detr"):
        model, processor = table_transformer.build_model()
        train_dset = dataset.DETRDataset(cfg, cfg.train_img_dir,
                                        cfg.train_json_dir,
                                        processor)
        val_dset = dataset.DETRDataset(cfg, cfg.val_img_dir,
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
    
        # model = torch.nn.DataParallel(model)  
    
    
        train.detr_train(cfg, model, processor, train_dl, val_dl, optimizer, scheduler, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=str)
    args = parser.parse_args()

    GPU_NUM = 2
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    with open(args.config_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    wandb.init(project="DETR weeds finetuning")
    
    wandb.run.name = "padday_weeds_DETR_fintuning"
    wandb.run.save()
    wandb.config.update(cfg)    
    
    cfg = type('cfg', (), cfg)
    
    main(cfg, device)
    