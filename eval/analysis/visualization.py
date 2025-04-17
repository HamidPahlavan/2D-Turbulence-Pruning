import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from analysis.plot_config import params

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
