import os
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm

import test

def train(cfg, model, processor, train_dl, test_dl, optimizer, scheduler):
    device = cfg.device
    
    model = model.to(device)
    
    for epoch in range(cfg.n_epochs):
        loss_per_epoch = 0
        loss_per_epoch_dict = {}
        
        model.train()
        for batch_idx, (pixel_values, pixel_mask, bbox, label) in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"{epoch+1} training...", ncols=60):
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)
            
            out = model(pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        box=bbox,
                        label=label)
            
            # idx = torch.argmax(torch.nn.functional.softmax(out['logits'][0], dim=1)[:, 0])
            
            # print(out['loss_dict'])
            # print(out['loss'])
            
            loss = out['loss'].sum() / len(out['loss'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_per_epoch += loss
            
            for k, v in out['loss_dict'].items():
                if k not in loss_per_epoch_dict:
                    loss_per_epoch_dict[k] = 0
                loss_per_epoch_dict[k] += (v.sum())/len(v)
            
            if (batch_idx+1) % 100 == 0:
                print("train", (loss_per_epoch / (batch_idx+1)).item())
                wandb.log({'train total loss':(loss_per_epoch / (batch_idx+1)).item()})
                for k, v in loss_per_epoch_dict.items():
                    wandb.log({"train" + k:(loss_per_epoch_dict[k] / (batch_idx+1)).item()})
                    print("train" + k, (loss_per_epoch_dict[k] / (batch_idx+1)).item())
                    
            
        scheduler.step()
        
        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
            
        torch.save({'weight': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   os.path.join(cfg.ckpt_dir, f"{(epoch+1):0>5}_ckpt.pt"))
        
        test.evaluate(cfg, model, test_dl)
            
            
            

