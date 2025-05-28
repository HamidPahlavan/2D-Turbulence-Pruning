import numpy as np
import torch
from analysis.metrics import get_rmse, get_acc, get_spectra
from analysis.rollout import n_step_rollout
from analysis.visualization import plot_analysis

def perform_short_analysis(model, dataloader, dataset, climo_u, climo_v, short_analysis_params, train_params, dataset_params, device):
    """
    Perform short-run analyses that do not require saving/loading of large data sets.
    Returns a dictionary of results.
    """
    results = {}
    rmse_flag = short_analysis_params["rmse"]
    acc_flag = short_analysis_params["acc"]
    spectra_flag = short_analysis_params["spectra"]

    if not (rmse_flag or acc_flag or spectra_flag):
        return results

    print('************ Short analysis ************')
    rmse_u, rmse_u_per, rmse_v, rmse_v_per = [], [], [], []
    acc_u, acc_u_per, acc_v, acc_v_per = [], [], [], []
    spectra_list, spectra_tar_list, wavenumbers_list = [], [], None

    for i, batch in enumerate(dataloader):
        if i > short_analysis_params["num_ensembles"]:
            break
        
        print(f'Prediction iteration: {i}')
        inputs, targets = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
        ic = inputs[0].unsqueeze(dim=0)
        n_steps = inputs.shape[0]

        pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=train_params["train_tendencies"])
        per_pred = inputs[0].repeat(n_steps, 1, 1, 1, 1)[:,:,0,:,:]

        # Move to CPU numpy
        pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
        per_pred = per_pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
        targets = targets.transpose(-1,-2).squeeze().detach().cpu().numpy()

        pred_u = pred[:,0]
        pred_v = pred[:,1]
        per_pred_u = per_pred[:,0]
        per_pred_v = per_pred[:,1]
        tar_u = targets[:,0]
        tar_v = targets[:,1]

        # Unnormalize
        pred_u = (pred_u * dataset.input_std[0] + dataset.input_mean[0])
        pred_v = (pred_v * dataset.input_std[1] + dataset.input_mean[1])
        tar_u = (tar_u * dataset.label_std[0] + dataset.label_mean[0])
        tar_v = (tar_v * dataset.label_std[1] + dataset.label_mean[1])
        
        if rmse_flag:
            rmse_u.append(get_rmse(tar_u, pred_u, climo=climo_u))
            rmse_u_per.append(get_rmse(tar_u, per_pred_u, climo=climo_u))
            rmse_v.append(get_rmse(tar_v, pred_v, climo=climo_v))
            rmse_v_per.append(get_rmse(tar_v, per_pred_v, climo=climo_v))

        if acc_flag:
            acc_u.append(get_acc(tar_u, pred_u, climo_u))
            acc_u_per.append(get_acc(tar_u, per_pred_u, climo_u))
            acc_v.append(get_acc(tar_v, pred_v, climo_v))
            acc_v_per.append(get_acc(tar_v, per_pred_v, climo_v))

        if spectra_flag:
            spectra_temp, wavenumbers = get_spectra(pred_u, pred_v)
            spectra_tar_temp, _ = get_spectra(tar_u, tar_v)
            spectra_list.append(spectra_temp)
            spectra_tar_list.append(spectra_tar_temp)
            if wavenumbers_list is None:
                wavenumbers_list = wavenumbers

    # Aggregate results
    if rmse_flag:

        results['rmse_u'] = np.stack(rmse_u, axis=0)
        results['rmse_v'] = np.stack(rmse_v, axis=0)
        results['rmse_u_median'] = np.quantile(np.stack(rmse_u, axis=0), 0.5, axis=0)
        results['rmse_u_uq'] = np.quantile(np.stack(rmse_u, axis=0), 0.75, axis=0)
        results['rmse_u_lq'] = np.quantile(np.stack(rmse_u, axis=0), 0.25, axis=0)
        results['rmse_u_per_median'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.5, axis=0)
        results['rmse_u_per_uq'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.75, axis=0)
        results['rmse_u_per_lq'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.25, axis=0)
        results['rmse_v_median'] = np.quantile(np.stack(rmse_v, axis=0), 0.5, axis=0)
        results['rmse_v_uq'] = np.quantile(np.stack(rmse_v, axis=0), 0.75, axis=0)
        results['rmse_v_lq'] = np.quantile(np.stack(rmse_v, axis=0), 0.25, axis=0)
        results['rmse_v_per_median'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.5, axis=0)
        results['rmse_v_per_uq'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.75, axis=0)
        results['rmse_v_per_lq'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.25, axis=0)

        results['rmse_u_mean'] = np.mean(rmse_u, axis=0)
        results['rmse_u_std'] = np.std(rmse_u, axis=0)
        results['rmse_u_per_mean'] = np.mean(rmse_u_per, axis=0)
        results['rmse_u_per_std'] = np.std(rmse_u_per, axis=0)
        results['rmse_v_mean'] = np.mean(rmse_v, axis=0)
        results['rmse_v_std'] = np.std(rmse_v, axis=0)
        results['rmse_v_per_mean'] = np.mean(rmse_v_per, axis=0)
        results['rmse_v_per_std'] = np.std(rmse_v_per, axis=0)

    if acc_flag:

        results['acc_u'] = np.stack(acc_u, axis=0)
        results['acc_v'] = np.stack(acc_v, axis=0)
        results['acc_u_median'] = np.quantile(np.stack(acc_u, axis=0), 0.5, axis=0)
        results['acc_u_uq'] = np.quantile(np.stack(acc_u, axis=0), 0.75, axis=0)
        results['acc_u_lq'] = np.quantile(np.stack(acc_u, axis=0), 0.25, axis=0)
        results['acc_u_per_median'] = np.quantile(np.stack(acc_u_per, axis=0), 0.5, axis=0)
        results['acc_u_per_uq'] = np.quantile(np.stack(acc_u_per, axis=0), 0.75, axis=0)
        results['acc_u_per_lq'] = np.quantile(np.stack(acc_u_per, axis=0), 0.25, axis=0)
        results['acc_v_median'] = np.quantile(np.stack(acc_v, axis=0), 0.5, axis=0)
        results['acc_v_uq'] = np.quantile(np.stack(acc_v, axis=0), 0.75, axis=0)
        results['acc_v_lq'] = np.quantile(np.stack(acc_v, axis=0), 0.25, axis=0)
        results['acc_v_per_median'] = np.quantile(np.stack(acc_v_per, axis=0), 0.5, axis=0)
        results['acc_v_per_uq'] = np.quantile(np.stack(acc_v_per, axis=0), 0.75, axis=0)
        results['acc_v_per_lq'] = np.quantile(np.stack(acc_v_per, axis=0), 0.25, axis=0)

        results['acc_u_mean'] = np.mean(acc_u, axis=0)
        results['acc_u_std'] = np.std(acc_u, axis=0)
        results['acc_u_per_mean'] = np.mean(acc_u_per, axis=0)
        results['acc_u_per_std'] = np.std(acc_u_per, axis=0)
        results['acc_v_mean'] = np.mean(acc_v, axis=0)
        results['acc_v_std'] = np.std(acc_v, axis=0)
        results['acc_v_per_mean'] = np.mean(acc_v_per, axis=0)
        results['acc_v_per_std'] = np.std(acc_v_per, axis=0)

    if spectra_flag:
        results['spectra'] = np.mean(spectra_list, axis=0)
        results['spectra_tar'] = np.mean(spectra_tar_list, axis=0)
        results['wavenumbers'] = wavenumbers_list

    if short_analysis_params["plot_analysis"]:
        plot_analysis(results, short_analysis_params, dataset_params, train_params)

    return results
