import torch
import torch.nn as nn

def CIoU(pred, target):
    """
    Compute the Complete Intersection over Union (CIoU) loss.
    Args:
        pred (torch.Tensor): Predicted bounding boxes.
        target (torch.Tensor): Target bounding boxes.
    Returns:
        torch.Tensor: CIoU loss.
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target_x1, target_y1, target_x2, target_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # Calculate the center points of the predicted and target boxes
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    target_center_x = (target_x1 + target_x2) / 2
    target_center_y = (target_y1 + target_y2) / 2

    # Calculate the width and height of the predicted and target boxes
    pred_width = pred_x2 - pred_x1
    pred_height = pred_y2 - pred_y1
    target_width = target_x2 - target_x1
    target_height = target_y2 - target_y1

    # Calculate the area of the predicted and target boxes
    pred_area = pred_width * pred_height
    target_area = target_width * target_height

    # Calculate the intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height

    # Calculate the union area
    union_area = pred_area + target_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    # Calculate the center distance
    center_distance_squared = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2