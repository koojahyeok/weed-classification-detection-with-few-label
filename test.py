from tqdm import tqdm

import torch
import torch.nn as nn
import wandb

import utils

def detr_evaluate(cfg, model, test_dl):
    device = cfg.device
    
    model = model.to(device)
    
    loss_per_epoch = 0
    loss_per_epoch_dict = {}
    
    true_pos, false_pos, false_neg = 0, 0, 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (pixel_values, pixel_mask, bbox, label) in tqdm(enumerate(test_dl), total=len(test_dl), desc=f"evaluate...", ncols=60):
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)
            
            out = model(pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        box=bbox,
                        label=label)        
            
            loss = out['loss'].sum() / len(out['loss'])
            loss_per_epoch += loss
            
            for k, v in out['loss_dict'].items():
                if k not in loss_per_epoch_dict:
                    loss_per_epoch_dict[k] = 0
                loss_per_epoch_dict[k] += (v.sum())/len(v)
                
            box_indices = utils.select_box_indices_with_threshold(out['logits'], out['pred_boxes'], cfg.box_threshold)
        
            tp, fp, fn = utils.get_confusion_matrix(out['pred_boxes'], box_indices, bbox.to(box_indices.device), cfg.classification_threshold)
            true_pos += tp
            false_pos += fp
            false_neg += fn
            
    print(true_pos, false_pos, false_neg)
    print({'precision': true_pos/(true_pos+false_pos+1e-6)})
    print({'recall': true_pos/(true_pos+false_neg+1e-6)})
    print((loss_per_epoch / (batch_idx+1)).item())
    
    wandb.log({'val precision': true_pos/(true_pos+false_pos+1e-6)})
    wandb.log({'val recall': true_pos/(true_pos+false_neg+1e-6)})
    wandb.log({'val total loss':(loss_per_epoch / (batch_idx+1)).item()})
    
    for k, v in loss_per_epoch_dict.items():
        wandb.log({"val" + k:(loss_per_epoch_dict[k] / (batch_idx+1)).item()})
        print("val" + k, (loss_per_epoch_dict[k] / (batch_idx+1)).item())
                

            
            
# def frcnn_evaluate(cfg, model, test_dl):
#     device = cfg.device
    
#     model = model.to(device)
    
#     loss_per_epoch = 0
#     loss_per_epoch_dict = {}
    
#     true_pos, false_pos, false_neg = 0, 0, 0
    
#     model.eval()
    
#     with torch.no_grad():
#         for batch_idx, (pixel_values, pixel_mask, bbox, label) in tqdm(enumerate(test_dl), total=len(test_dl), desc=f"evaluate...", ncols=60):

            
#             out = model(pixel_values=pixel_values,
#                         pixel_mask=pixel_mask,
#                         box=bbox,
#                         label=label)        
            
#             loss = out['loss'].sum() / len(out['loss'])
#             loss_per_epoch += loss
            
#             for k, v in out['loss_dict'].items():
#                 if k not in loss_per_epoch_dict:
#                     loss_per_epoch_dict[k] = 0
#                 loss_per_epoch_dict[k] += (v.sum())/len(v)
                
    #         box_indices = utils.select_box_indices_with_threshold(out['logits'], out['pred_boxes'], cfg.box_threshold)
        
    #         tp, fp, fn = utils.get_confusion_matrix(out['pred_boxes'], box_indices, bbox.to(box_indices.device), cfg.classification_threshold)
    #         true_pos += tp
    #         false_pos += fp
    #         false_neg += fn
            
    # print(true_pos, false_pos, false_neg)
    # print({'precision': true_pos/(true_pos+false_pos+1e-6)})
    # print({'recall': true_pos/(true_pos+false_neg+1e-6)})
    # print((loss_per_epoch / (batch_idx+1)).item())
    
    # wandb.log({'val precision': true_pos/(true_pos+false_pos+1e-6)})
    # wandb.log({'val recall': true_pos/(true_pos+false_neg+1e-6)})
    # wandb.log({'val total loss':(loss_per_epoch / (batch_idx+1)).item()})
    
    # for k, v in loss_per_epoch_dict.items():
    #     wandb.log({"val" + k:(loss_per_epoch_dict[k] / (batch_idx+1)).item()})
    #     print("val" + k, (loss_per_epoch_dict[k] / (batch_idx+1)).item())