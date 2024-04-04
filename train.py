import os
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm

import test

def detr_train(cfg, model, processor, train_dl, test_dl, optimizer, scheduler):
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
        
        test.detr_evaluate(cfg, model, test_dl)
            
            
            

def frcnn_train(cfg, model, train_dl, test_dl, device):
    device = cfg.device
    
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.milestones, cfg.gamma, last_epoch=cfg.last_epoch, verbose=cfg.scheduler_verbosity)
    
    for epoch in tqdm(range(cfg.n_epochs), total=cfg.n_epochs, desc="training...", ncols=60):
        model.train()

        i = 0
        epoch_loss = 0

        for batch_idx, (imgs, annotations) in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"{epoch+1} training...", ncols=60):
            i += 1
    
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        scheduler.step()

        print(f'epoch : {epoch+1}, Loss : {epoch_loss}')

        # test.frcnn_evaluate(cfg, model, test_dl)

        torch.save(model.state_dict(),f'model_{cfg.try_cnt}.pt')