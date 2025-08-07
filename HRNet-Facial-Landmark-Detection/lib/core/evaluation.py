# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds
import torch.nn.functional as F

def soft_argmax_points(heatmaps, temperature=0.1):
 
    N, K, H, W = heatmaps.shape

    heatmaps_softmax = F.softmax(heatmaps.view(N, K, -1) / temperature, dim=2)
    heatmaps_softmax = heatmaps_softmax.view(N, K, H, W)
    xs, ys = torch.meshgrid(torch.arange(W, device=heatmaps.device, dtype=torch.float32), 
                            torch.arange(H, device=heatmaps.device, dtype=torch.float32),
                            indexing='xy')
    xs = xs.view(1, 1, H, W)
    ys = ys.view(1, 1, H, W)

    pred_x = (heatmaps_softmax * xs).sum(dim=[2, 3])
    pred_y = (heatmaps_softmax * ys).sum(dim=[2, 3])
    
    return torch.stack([pred_x, pred_y], dim=2)
def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse

def decode_preds(output, center, scale, res):
    _, _, H, W = output.shape
    coords = soft_argmax_points(output) 
    coords_np = coords.detach().cpu().numpy()
    center_np = center.cpu().numpy()
    scale_np = scale.cpu().numpy()
    
    final_preds_np = np.zeros_like(coords_np)

    for i in range(coords_np.shape[0]):
        final_preds_np[i] = transform_preds(coords_np[i] + 1, center_np[i], scale_np[i], [W, H])

    final_preds = torch.from_numpy(final_preds_np)
    if final_preds.dim() < 3:
        final_preds = final_preds.view(1, final_preds.size())

    return final_preds