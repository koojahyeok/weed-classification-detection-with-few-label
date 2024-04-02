import torch
import torch.nn as nn

from transformers import AutoImageProcessor, TableTransformerForObjectDetection, TableTransformerModel

def build_model(model_name="microsoft/table-transformer-detection"):
    model = TableDETR(model_name="microsoft/table-transformer-detection")
    processor = AutoImageProcessor.from_pretrained(model_name)

    return model, processor

class TableDETR(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.label_wrapper = lambda x, y, device: {'boxes':x.to(device), 'class_labels': y.to(device)} 
        
    def forward(self, pixel_values, pixel_mask, box, label):
        labels = [self.label_wrapper(box[i].unsqueeze(0), label[i].unsqueeze(0), pixel_values.device) for i in range(len(box))]
        out = self.model(pixel_values=pixel_values,
                         pixel_mask=pixel_mask,
                         labels=labels)
        
        out = {'loss': out.loss,
                'loss_dict': out.loss_dict,
                'logits': out.logits,
                'pred_boxes': out.pred_boxes,
                'last_hidden_state': out.last_hidden_state}
        
        return out
    
    def forward_inference(self, pixel_values, pixel_mask):
        out = self.model(pixel_values=pixel_values,
                         pixel_mask=pixel_mask)
        
        out = {'logits': out.logits,
                'pred_boxes': out.pred_boxes,
                'last_hidden_state': out.last_hidden_state}
        
        return out