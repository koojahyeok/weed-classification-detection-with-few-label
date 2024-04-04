import os
import json
import torch
import torch.nn.functional as F

from torchvision.transforms import transforms

def get_image_transform(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.img_size, cfg.img_size)),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomVerticalFlip(0.5),
        # processor
    ])
    
    return transform

def get_box_transform(cfg):
    # return lambda x : torch.FloatTensor(x) / cfg.img_size
    return lambda x : torch.FloatTensor([x[1]+x[2]/2, x[0]+x[3]/2, x[2], x[3]]) / cfg.img_size
    # return lambda x: torch.FloatTensor(x)

## get target for faster rcnn
# def generate_target(pth):

#     boxes = []
#     labels = []

#     for json_name in os.listdir(pth):
#         json_pth = os.path.join(pth, json_name)

#         with open(json_pth, "r") as f:
#             data = json.load(f)
#             bbox = data["annotations"]["bbox"]
#             weed_lbl = data["annotations"]["weeds_kind"]
#             lbl = 0
            
#             if (weed_lbl == 'N'):
#                 lbl = 0
#             elif (weed_lbl == 'E'):
#                 lbl = 1
#             elif (weed_lbl == 'A'):
#                 lbl = 2
#             elif (weed_lbl == 'V'):
#                 lbl = 3
#             elif (weed_lbl == 'I'):
#                 lbl = 4
#             elif (weed_lbl == 'B'): 
#                 lbl = 5
#             else: 
#                 lbl = 6
    
#             boxes.append(bbox)
#             labels.append(lbl)
    
#     boxes = torch.as_tensor(boxes, dtype=torch.float32)
#     labels = torch.as_tensor(labels, dtype=torch.int64)
    
#     target = {}
#     target["boxes"] = boxes
#     target["labels"] = labels

#     return target
    
## faster rcnn function
def collate_fn(batch):
    return tuple(zip(*batch))

    
def calculate_iou(pred_boxes, gt_boxes):
    """
    Calculate the IoU (Intersection over Union) between predicted boxes and ground truth boxes.

    Parameters:
    pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4), each box is (cx, cy, w, h)
    gt_boxes (Tensor): Ground truth bounding boxes of shape (N, 4), each box is (cx, cy, w, h)

    Returns:
    Tensor: IoU for each pair of boxes
    """
    # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
    def convert_to_corners(boxes):
        x1 = boxes[:, 0] - 0.5 * boxes[:, 2]
        y1 = boxes[:, 1] - 0.5 * boxes[:, 3]
        x2 = boxes[:, 0] + 0.5 * boxes[:, 2]
        y2 = boxes[:, 1] + 0.5 * boxes[:, 3]
        return torch.stack((x1, y1, x2, y2), dim=1)

    pred_corners = convert_to_corners(pred_boxes)
    gt_corners = convert_to_corners(gt_boxes)

    # Intersection
    inter_x1 = torch.max(pred_corners[:, 0], gt_corners[:, 0])
    inter_y1 = torch.max(pred_corners[:, 1], gt_corners[:, 1])
    inter_x2 = torch.min(pred_corners[:, 2], gt_corners[:, 2])
    inter_y2 = torch.min(pred_corners[:, 3], gt_corners[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
    gt_area = (gt_corners[:, 2] - gt_corners[:, 0]) * (gt_corners[:, 3] - gt_corners[:, 1])

    union_area = pred_area + gt_area - inter_area

    # IoU
    iou = inter_area / union_area

    return iou

def calculate_giou(pred_boxes, gt_boxes):
    """
    Calculate the GIoU (Generalized Intersection over Union) between predicted boxes and ground truth boxes.

    Parameters:
    pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4), each box is (cx, cy, w, h)
    gt_boxes (Tensor): Ground truth bounding boxes of shape (N, 4), each box is (cx, cy, w, h)

    Returns:
    Tensor: GIoU for each pair of boxes
    """
    # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
    def convert_to_corners(boxes):
        x1 = boxes[:, 0] - 0.5 * boxes[:, 2]
        y1 = boxes[:, 1] - 0.5 * boxes[:, 3]
        x2 = boxes[:, 0] + 0.5 * boxes[:, 2]
        y2 = boxes[:, 1] + 0.5 * boxes[:, 3]
        return torch.stack((x1, y1, x2, y2), dim=1)

    pred_corners = convert_to_corners(pred_boxes)
    gt_corners = convert_to_corners(gt_boxes)

    # Intersection
    inter_x1 = torch.max(pred_corners[:, 0], gt_corners[:, 0])
    inter_y1 = torch.max(pred_corners[:, 1], gt_corners[:, 1])
    inter_x2 = torch.min(pred_corners[:, 2], gt_corners[:, 2])
    inter_y2 = torch.min(pred_corners[:, 3], gt_corners[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
    gt_area = (gt_corners[:, 2] - gt_corners[:, 0]) * (gt_corners[:, 3] - gt_corners[:, 1])

    union_area = pred_area + gt_area - inter_area

    # IoU
    iou = inter_area / union_area

    # Enclosing box for GIoU
    enc_x1 = torch.min(pred_corners[:, 0], gt_corners[:, 0])
    enc_y1 = torch.min(pred_corners[:, 1], gt_corners[:, 1])
    enc_x2 = torch.max(pred_corners[:, 2], gt_corners[:, 2])
    enc_y2 = torch.max(pred_corners[:, 3], gt_corners[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area

    return giou
    
def select_box_indices_with_threshold(logits, boxes, threshold):
    # boxes -> center coord + width, height
    
    table_probs = F.softmax(logits, dim=2)[:, :, 0]
    
    # n_boxes x 4
    # selected_boxes = boxes[table_probs > threshold]
    
    return table_probs > threshold
    
def get_IoU_matrix(pred_boxes, gt_boxes):
    iou_mat = torch.zeros((len(pred_boxes), len(gt_boxes))).to(pred_boxes.device)
    # print(iou_mat.shape)
    for i in range(len(pred_boxes)):
        iou_mat[i, :] += calculate_iou(pred_boxes[i:i+1], gt_boxes)
            
    return iou_mat
    
def get_confusion_matrix(pred_boxes, box_indices, gt_boxes, threshold):
    # pred boxes -> B x 15 x 4
    # box indices -> B x 15
    # gt boxes -> B x N x 4
    
    tp, fn, fp = 0, 0, 0
    
    for b in range(pred_boxes.shape[0]):
        # print(b)
        if box_indices[b].sum() == 0:
            # print("no box detected")
            # return 0, len(gt_boxes[b]), 0
            
            # assume one object per examples
            fn +=  1
            continue
        
        tp_tmp = 0
        # N_pred x N_gt
        iou_mat = get_IoU_matrix(pred_boxes[b][box_indices[b]], gt_boxes[b:b+1])
        
        # N_gt
        iou_argmax = torch.argmax(iou_mat, dim=0)
        
        for i in range(iou_mat.shape[1]):
            if iou_mat[iou_argmax[i]][i] > threshold:
                tp += 1
                tp_tmp += 1
            else:
                fn += 1
            
        fp += (box_indices[b].sum() - tp_tmp)
    
    return tp, fp, fn
    
def box_postprocessing(pred_boxes, img_size=800):
    box_lt_x_coords = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    box_lt_y_coords = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    
    return torch.concat([box_lt_x_coords, box_lt_y_coords, pred_boxes[:, 2:]], dim=1) * img_size
