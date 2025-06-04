import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from analysis.plot_config import params
from analysis.metrics import return_period_empirical, ensemble_return_period_amplitude

def plot_analysis(results, analysis_dict, dataset_params, train_params):

    plot_dir = os.path.join(dataset_params["root_dir"], dataset_params["run_num"], 'plots')

    font = {'size': 14}
    mpl.rc('font', **font)

    if analysis_dict['rmse']:
        # U
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.rcParams.update(params)


        x = np.arange(1, 1+len(results['rmse_u_median'])) 
        ax.plot(x, results['rmse_u_median'], '-k', label='Emulator')
        upper = results['rmse_u_uq'] # + results['rmse_u_std']
        lower = results['rmse_u_lq'] # - results['rmse_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_u_per_median'], '--b', label='Persistence')
        upper = results['rmse_u_per_uq'] # + results['rmse_u_per_std']
        lower = results['rmse_u_per_lq'] # - results['rmse_u_per_std']
        ax.fill_between(x, lower, upper, color='b', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($dt$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, len(results['rmse_u_median'])])

        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(plot_dir + '/RMSE_U_' + '.svg')
        # V
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.rcParams.update(params)

        ax.plot(x, results['rmse_v_median'], '-k', label='Emulator')
        upper = results['rmse_v_uq'] # + results['rmse_v_std']
        lower = results['rmse_v_lq'] # - results['rmse_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_v_per_median'], '--b', label='Persistence')
        upper = results['rmse_v_per_uq'] # + results['rmse_v_per_std']
        lower = results['rmse_v_per_lq'] # - results['rmse_v_per_std']
        ax.fill_between(x, lower, upper, color='b', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($dt$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, len(results['rmse_v_median'])])

        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(plot_dir + '/RMSE_V_' + '.svg')

    if analysis_dict['acc']:
        # U
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.rcParams.update(params)

        x = np.arange(1, 1+len(results['acc_u_median'])) 
        ax.plot(x, results['acc_u_median'], '-k', label='Emulator')
        upper = results['acc_u_uq'] # + results['acc_u_std']
        lower = results['acc_u_lq'] # - results['acc_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_u_per_median'], '--b', label='Persistence')
        upper = results['acc_u_per_uq'] # + results['acc_u_per_std']
        lower = results['acc_u_per_lq'] # - results['acc_u_per_std']
        ax.fill_between(x, lower, upper, color='b', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($dt$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(results['acc_u_median'])])

        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(plot_dir + '/ACC_U_' + '.svg')
        # V
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.rcParams.update(params)

        ax.plot(x, results['acc_v_median'], '-k', label='Emulator')
        upper = results['acc_v_uq'] # + results['acc_v_std']
        lower = results['acc_v_lq'] # - results['acc_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_v_per_median'], '--b', label='Persistence')
        upper = results['acc_v_per_uq'] # + results['acc_v_per_std']
        lower = results['acc_v_per_lq'] # - results['acc_v_per_std']
        ax.fill_between(x, lower, upper, color='b', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($dt$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(results['acc_v_median'])])

        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(plot_dir + '/ACC_V_' + '.svg')

    if analysis_dict['spectra']:
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.rcParams.update(params)
        x = results['wavenumbers']
        ax.plot(x, results['spectra_tar'][0], '-k', label='Truth')
        for lead in analysis_dict['spectra_leadtimes']:
            spec = results['spectra'][lead]

            label = rf'{lead+1} $dt$' 

            ax.plot(x, spec, label=label)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('$k$')
            ax.set_ylabel('Power')
            ax.set_xlim([0.8, (train_params['img_size']+10)//2])
            ax.set_ylim([10**(-9), 10])
            ax.legend(frameon=False)
            plt.tight_layout()
            fig.savefig(plot_dir + '/Power_Spectra_' + '.svg')

    # if analysis_dict['zonal_pca']:
    #     # Plot EOFs
    #     pred_u_pcs = results['pred_u_pc']
    #     pred_u_eofs = results['pred_u_eof']
    #     tar_u_pcs = results['tar_u_pc']
    #     tar_u_eofs = results['tar_u_eof']
    #     eofs = [pred_u_eofs, tar_u_eofs]
    #     colors = ['k', 'r', 'b', 'g']
    #     x = np.linspace(0, 2*np.pi, pred_u_eofs.shape[1])
    #     for i in range(pred_u_eofs.shape[0]):
    #         fig, ax = plt.subplots()
    #         ax.plot(pred_u_eofs[i, :], x, f'--{colors[i]}', label=f'ML EOF{i+1}')
    #         ax.plot(tar_u_eofs[i, :], x, f'-{colors[i]}', label=f'Truth EOF{i+1}')
    #         ax.set_xlim([-0.25, 0.25])
    #         ax.set_ylabel('x')
    #         ax.set_title(f'EOF{i+1} of zonally-averaged U')
    #         ax.legend()
    #         plt.tight_layout()
    #         fig.savefig(plot_dir + f'/EOF{i+1}_' + '.svg')

    #     for i in range(pred_u_pcs.shape[1]):
    #         fig, ax = plt.subplots()
    #         x = np.arange(1, 1+pred_u_pcs.shape[0])
    #         ax.plot(x, pred_u_pcs[:,i], '-k', label='ML')
    #         ax.plot(x, tar_u_pcs[:,i], '--k', label='Truth')
    #         ax.set_xlabel('ML timestep')
    #         ax.set_ylabel('PC')
    #         ax.legend()
    #         plt.tight_layout()
    #         fig.savefig(plot_dir + '/PC{i+1}_' + '.svg')

    # if analysis_dict['div']:
    #     fig, ax = plt.subplots()
    #     x = np.arange(1, 1+results['pred_div'].shape[0])
    #     ax.semilogy(x, results['pred_div'], '--k', label='ML')
    #     ax.semilogy(x, results['tar_div'], '-k', label='Truth')
    #     ax.set_xlabel('ML timestep')
    #     #ax.set_ylim([-1, 1])
    #     ax.legend()
    #     plt.tight_layout()
    #     fig.savefig(plot_dir + '/Div_' + '.svg')

def make_video_old(pred, tar):
    """
    Args:
        pred, tar: [B=n_steps, C, X, Y]
    """

    frames = []
    for t in range(pred.shape[0]):
        if t%10 == 0:
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            axs = axs.flatten()
            data = [pred[t, 0, :, :], pred[t, 1, :, :], tar[t, 0, :, :], tar[t, 1, :, :]]
            titles = ['ML: U', 'ML: V', 'Truth: U', 'Truth: V']
            for i, ax in enumerate(axs):

                data_i = data[i] #.transpose((-1,-2))
                im = ax.imshow(data_i, cmap='bwr', vmin=-5, vmax=5, aspect='equal')
                xlen = data_i.shape[-1]
                ax.set_title(titles[i])
                ax.set_xticks([0, xlen/2, xlen], [0, r'$\pi$', r'$2\pi$']) 
                ax.set_yticks([0, xlen/2, xlen], [0, r'$\pi$', r'$2\pi$'])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            fig.suptitle(rf'{t+1}$\Delta t$')

            fig.savefig('temp_frame.png', bbox_inches='tight')
            plt.close()

            frames.append(imageio.imread('temp_frame.png'))


    imageio.mimsave(f'Video_' + run_num + '.gif', frames, fps=1)

### Plotting Code for Ensembles
def align_eofs(eofs, reference_idx=0):
    """
    Aligns the sign of each EOF (1D vector) to the 'reference' EOF.
    
    Parameters
    ----------
    eofs : ndarray, shape (n_ensembles, n_points)
        The raw EOFs you computed for each ensemble. Each row is one EOF (1D array).
    reference_idx : int, default=0
        Index of the ensemble to use as the reference. By default, we align
        everything to eofs[0].
    
    Returns
    -------
    aligned : ndarray, shape (n_ensembles, n_points)
        A copy of `eofs` where each row has been multiplied by +1 or –1 so that
        its dot‐product with the reference EOF is ≥ 0.
    """
    # Make a copy so we don’t modify the original array in place
    aligned = eofs.copy()
    
    # Choose the reference vector
    ref = aligned[reference_idx]
    
    # Loop over each EOF (each row)
    for i in range(aligned.shape[0]):
        dot = np.dot(aligned[i], ref)
        # If the dot‐product is negative, flip the sign
        if dot < 0:
            aligned[i] = -aligned[i]
    
    return aligned

def align_all_eofs(data_arr):
    """
    Aligns all EOFs in a 3D array along the ensemble axis for each EOF index.

    Parameters
    ----------
    data_arr : ndarray, shape (n_ensembles, n_points, n_eofs)
        Array containing EOFs for each ensemble and each EOF index.

    Returns
    -------
    aligned_eofs : ndarray, shape (n_ensembles, n_points, n_eofs)
        Array with all EOFs aligned in sign for each EOF index.
    """
    aligned_eofs = []
    # Loop over each EOF index (last axis)
    for eof_idx in range(data_arr.shape[2]):
        # Extract all ensembles for this EOF index
        eofs = data_arr[:, :, eof_idx]
        # Align the signs of the EOFs
        aligned = align_eofs(eofs)
        aligned_eofs.append(aligned)
    # Stack along the last axis to restore original shape
    aligned_eofs = np.stack(aligned_eofs, axis=2)
    return aligned_eofs



def ensembles_analysis(data_analysis_dir, data_run_names, var_name, data_type='truth', central_tendency='mean', error_bands='std'):

    # var_name = 'spectra_U_zonal_avg'
    # data_analysis_dir = truth_analysis_dir
    # data_run_names = truth_run_names
    # # Load parameters

    # central_tendency = 'mean'  # 'mean' or 'median'
    # error_bands = 'std'  # 'std' or '95ci' or 50ci'

    data_arr = []
    # data_type = 'truth' # emulate train
    for run_name in data_run_names:

        analysis_dir = os.path.join(data_analysis_dir, run_name, 'analysis', data_type)

        if var_name in ['spectra_U_zonal_avg', 'spectra_V_zonal_avg', 'spectra_Omega_zonal_avg']:
            filename = 'spectra.npz'

        elif var_name in ['U_zonal_mean', 'V_zonal_mean', 'Omega_zonal_mean']:
            filename = 'zonal_mean.npz'

        elif var_name in ['U_eofs', 'Omega_eofs', 'U_expvar', 'Omega_expvar', 'U_pc_acf', 'Omega_pc_acf']:
            filename = 'zonal_eof_pc.npz'

        elif var_name in ['U_sample_mean', 'V_sample_mean', 'Omega_sample_mean']:
            filename = 'temporal_mean.npz'

        elif var_name in ['div']:
            filename = 'div.npy'

        data = np.load(analysis_dir + f"/{filename}", allow_pickle=True)

        if var_name == 'div':
            # div is saved as an array rather than dict
            data_arr.append(data)
        elif var_name in ["U_pc_acf", "Omega_pc_acf"]:
            acf = data['U_pc_acf']
            # Stack the 'acf' arrays from each element in 'a' into a 2D numpy array: shape (len(a), acf_length)
            acf_array = np.stack([item['acf'] for item in acf], axis=1)
            data_arr.append(acf_array)
        else:
            data_arr.append(data[var_name])

    try:
        data_arr = np.array(data_arr)
    except ValueError as e:
        print(f"Error converting data to numpy array: {e}")
        print("Ensebmles arrays have different shapes or types. Resizing the ensembles to the smallest ensemble memeber.")
        # Ensuring all ensembles are of same size: Rextra time steps if needed
        min_size = min(arr.shape[0] for arr in data_arr)
        # Truncate all arrays to the smallest size along axis 0
        data_arr = np.array([arr[:min_size] for arr in data_arr])

    # print(data_arr.shape)
    
    if var_name in [ 'U_expvar', 'Omega_expvar']:
        data_arr = np.round(100* data_arr,1) # Convert to percentage

    if data_type in ['train']:
        if var_name in ['U_eofs', 'Omega_eofs', 'U_sample_mean', 'V_sample_mean', 'Omega_sample_mean']:
            return data_arr.reshape(data[var_name].shape[0], data[var_name].shape[1]), None, None
        elif var_name in ['U_pc_acf', 'Omega_pc_acf']:
            return data_arr.reshape(acf_array.shape[0], acf_array.shape[1]), None, None
        else:
            return data_arr.flatten(), None, None

    # Align EOFs to smimilar signs
    if var_name in ['U_eofs', 'Omega_eofs']:
        data_arr = align_all_eofs(data_arr)

    # Compute central tendency
    if central_tendency == 'mean':
        data_central = np.mean(data_arr, axis=0)
    elif central_tendency == 'median':
        data_central = np.median(data_arr, axis=0)

    # Error bars
    if error_bands == 'std':
        data_std = np.std(data_arr, axis=0)
        data_lower = data_central - data_std
        data_upper = data_central + data_std
    elif error_bands == '95ci':  # use percentiles for 95% confidence interval
        data_lower = np.percentile(data_arr, 2.5, axis=0)
        data_upper = np.percentile(data_arr, 97.5, axis=0)
    elif error_bands == '50ci':
        data_lower = np.percentile(data_arr, 25, axis=0)
        data_upper = np.percentile(data_arr, 75, axis=0)

    if var_name in ['U_eofs', 'Omega_eofs', 'U_pc_acf', 'Omega_pc_acf', 'U_sample_mean', 'V_sample_mean', 'Omega_sample_mean']:
        return data_central, data_lower, data_upper
    else:
        return data_central.flatten(), data_lower.flatten(), data_upper.flatten()


def ensembles_analysis_return_period(data_analysis_dir, data_run_names, var_name, anom=False, std=1, dt=1, bins_num=50, data_type='truth', central_tendency='mean', error_bands='std'):

    # var_name = 'spectra_U_zonal_avg'
    # data_analysis_dir = truth_analysis_dir
    # data_run_names = truth_run_names
    # # Load parameters

    # central_tendency = 'mean'  # 'mean' or 'median'
    # error_bands = 'std'  # 'std' or '95ci' or 50ci'

    data_arr = []
    # data_type = 'truth' # emulate train
    for run_name in data_run_names:

        analysis_dir = os.path.join(data_analysis_dir, run_name, 'analysis', data_type)
        
        if var_name in ['U_max_arr', 'V_max_arr', 'Omega_max_arr', 'U_min_arr', 'V_min_arr', 'Omega_min_arr']:
            if anom:
                filename = 'extremes_anom.npz'
            else:
                filename = 'extremes.npz'

        data = np.load(analysis_dir + f"/{filename}", allow_pickle=True)
        data_arr.append(data[var_name])

    try:
        data_arr = np.array(data_arr)
    except ValueError as e:
        print(f"Error converting data to numpy array: {e}")
        print("Ensebmles arrays have different shapes or types. Resizing the ensembles to the smallest ensemble memeber.")
        # Ensuring all ensembles are of same size: Rextra time steps if needed
        min_size = min(arr.shape[0] for arr in data_arr)
        # Truncate all arrays to the smallest size along axis 0
        data_arr = np.array([arr[:min_size] for arr in data_arr])

    data_arr = data_arr/std

    # Training data and emulation data are saved at different time steps
    if data_type in ['train']:
        data_arr = data_arr[::3]
        return_periods, data_amplitude, _, _  = ensemble_return_period_amplitude(
            np.asarray(np.abs(data_arr)), dt=dt, bins_num=bins_num, central_tendency=central_tendency, error_bands=None)

        if var_name in ['U_min_arr', 'V_min_arr', 'Omega_min_arr']:
            data_amplitude = -1 * data_amplitude
        return return_periods, data_amplitude, None, None

    return_periods, data_amplitude, data_amplitude_min, data_amplitude_max = ensemble_return_period_amplitude(
        np.asarray(np.abs(data_arr)), dt=dt, bins_num=bins_num, central_tendency=central_tendency, error_bands=error_bands)

    if var_name in ['U_min_arr', 'V_min_arr', 'Omega_min_arr']:
        data_amplitude = -1 * data_amplitude
        data_amplitude_min = -1 * data_amplitude_min
        data_amplitude_max = -1 * data_amplitude_max
    return return_periods, data_amplitude, data_amplitude_min, data_amplitude_max
