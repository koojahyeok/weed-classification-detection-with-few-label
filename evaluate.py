from torch.utils.data import DataLoader

import os
import torch
import yaml
import wandb
import argparse
import traceback

from models import table_transformer
import dataset
import utils
import train
import test

def main(cfg):
    model, processor = table_transformer.build_model()

    val_dset = dataset.DETRDataset(cfg, cfg.val_pos_img_dir,
                                     cfg.val_neg_img_dir,
                                     cfg.val_json_dir,
                                     processor)

    val_dl = DataLoader(val_dset,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=cfg.num_workers)
    
    model = torch.nn.DataParallel(model)
    ckpt_files = sorted(os.listdir(cfg.ckpt_dir), reverse=False)
    
    for ckpt_file in ckpt_files:
        print(f"evaluate {ckpt_file}")
        model_ckpt = torch.load(os.path.join(cfg.ckpt_dir, ckpt_file))
        
        model.load_state_dict(model_ckpt['weight'])
        
        test.evaluate(cfg, model, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=str)

    args = parser.parse_args()
    
    with open(args.config_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    wandb.init(project="Table Detection DETR MS finetuning")
    wandb.run.name = "0114_evaulation"
    wandb.run.save()
    wandb.config.update(cfg)    
    
    cfg = type('cfg', (), cfg)
    
    try:
        main(cfg)
    except Exception as e:
        wandb.finish()
        print(traceback.format_exc())




