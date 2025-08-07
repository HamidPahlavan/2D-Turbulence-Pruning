import torch
import numpy as np
from analysis.io_utils import save_numpy_data

def n_step_rollout(model, ic, n=1, train_tendencies=False):
    """Produce an n-step forward roll-out
    prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
        n (int): number of steps to predict for.
    Returns:
        pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

    pred = []

    for i in range(n):
        pred_temp, ic = single_step_rollout(model, ic, train_tendencies)
        pred.append(pred_temp)

    pred = torch.cat(pred, dim=0)

    return pred

def single_step_rollout(model, ic, train_tendencies=False):
    """Produce an single-step forward roll-out
    prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
    Returns:
        pred: [B=1, C, T, X, Y] single-step model prediction (time along dim=0).
        ic: [B=1, C, T, X, Y] initial condition for prediction for next time step"""

    n=1
    with torch.no_grad():
        idx = torch.tensor([0], device=ic.device)
        if train_tendencies:
            # WARNING: if num_out_frames > 1, only top frame is kept in auto-regressive rollout
            if ic.shape[2] > 1:
                # Use index_select to prevent reducing along dim of size 1
                pred = torch.index_select(ic, 2, index=idx) + torch.index_select(model(ic), 2, index=idx)
                ic = torch.cat([pred, ic[:,:,:-1,:,:]], dim=2)
            else:
                pred = ic + torch.index_select(model(ic), 2, index=idx)
                ic = pred
        else:
            if ic.shape[2] > 1:
                pred = torch.index_select(model(ic), 2, index=idx)
                ic = torch.cat([pred, ic[:,:,:-1,:,:]], dim=2)
            else:
                pred = torch.index_select(model(ic), 2, index=idx)
                ic = pred

    return pred, ic

# def n_step_rollout(model, ic, n=1, train_tendencies=False):
#     """Produce an n-step forward roll-out
#     prediction.
#     Args:
#         model: trained pytorch model.
#         ic: [B=1, C, T, X, Y] initial condition for prediction.
#         n (int): number of steps to predict for.
#     Returns:
#         pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

#     pred = []
#     with torch.no_grad():
#         idx = torch.tensor([0])
#         if train_tendencies:
#             # WARNING: if num_out_frames > 1, only top frame is kept in auto-regressive rollout
#             if ic.shape[2] > 1:
#                 for i in range(n):
#                     # Use index_select to prevent reducing along dim of size 1
#                     pred_temp = torch.index_select(ic, 2, index=idx) + torch.index_select(model(ic), 2, index=idx)
#                     pred.append(pred_temp)
#                     ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
#             else:
#                 for i in range(n):
#                     pred_temp = ic + torch.index_select(model(ic), 2, index=idx)
#                     pred.append(pred_temp)
#                     ic = pred_temp
#         else:
#             if ic.shape[2] > 1:
#                 for i in range(n):
#                     pred_temp = torch.index_select(model(ic), 2, index=idx)
#                     pred.append(pred_temp)          
#                     ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
#             else:
#                 for i in range(n):
#                     pred_temp = torch.index_select(model(ic), 2, index=idx)
#                     pred.append(pred_temp)
#                     ic = pred_temp

#     pred = torch.cat(pred, dim=0)



#     return pred
