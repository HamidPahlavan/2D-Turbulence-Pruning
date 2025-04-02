import os
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import imageio

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import UV2Omega, Omega2UV

from analysis.metrics import manual_eof, manual_svd_eof, divergence, PDF_compute
from analysis.rollout import n_step_rollout
from analysis.io_utils import load_numpy_data, get_npy_files, get_mat_files_in_range, run_notebook_as_script

def perform_long_analysis(save_dir, analysis_dir, dataset_params, long_analysis_params, train_params):

    """
    Perform long-run analysis. Checks if saved data exists.
    If save_data is True and data is already saved, it uses that.
    Else, it generates predictions and optionally saves them.
    """
    print('************ Long analysis ************')

    Lx, Ly = 2*np.pi, 2*np.pi
    Nx = train_params["img_size"]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Nx, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Nx, Lx, Ly, INDEXING='ij')

    if long_analysis_params["temporal_mean"] or long_analysis_params["zonal_mean"] or long_analysis_params["zonal_eof"] or long_analysis_params["div"] or long_analysis_params["return_period"]:
        # Load data
        perform_analysis = True
    else:
        perform_analysis = False

    for dataset in ['train', 'truth', 'emulate']:

        if not perform_analysis:
            break

        print('-------------- Calculating for dataset: ', dataset)

        if dataset == 'emulate':

            if long_analysis_params["long_analysis_emulator"]:
                # Data predicted by the emualtor
                files = get_npy_files(save_dir)
                print(f"Number of saved predicted .npy files: {len(files)}")
                analysis_dir_save = os.path.join(analysis_dir, 'emulate')
            else:
                continue

        elif dataset == 'train':
            if long_analysis_params["long_analysis_train"]:
                # Load training data
                files = get_mat_files_in_range(os.path.join(train_params["data_dir"],'data'), train_params["train_file_range"])
                print(f"Number of training .mat files: {len(files)}")
                analysis_dir_save = os.path.join(analysis_dir, 'train')
            else:
                continue

        elif dataset == 'truth':
            if long_analysis_params["long_analysis_truth"]:
                # Load training data
                # length of analysis for truth is same as emulator data
                files = get_mat_files_in_range(os.path.join(train_params["data_dir"],'data'), long_analysis_params["truth_file_range"])
                print(f"Number of truth .mat files: {len(files)}")
                analysis_dir_save = os.path.join(analysis_dir, 'truth')
            else:
                continue

        os.makedirs(analysis_dir_save, exist_ok=True)

        U_mean_temp = np.zeros((Nx, Nx))
        V_mean_temp = np.zeros((Nx, Nx))
        Omega_mean_temp = np.zeros((Nx, Nx))

        U_zonal = []
        Omega_zonal = []
        div = []
        U_max = []
        U_min = []
        V_max = []
        V_min = []
        Omega_max = []
        Omega_min = []

        Omega_arr = []
        U_arr = []

        for i, file in enumerate(files):
            # if dataset == 'emulate' and i > long_analysis_params["analysis_length"]:
            #     break
            if dataset == 'emulate' and i > long_analysis_params["analysis_length"]:
                total_files_analyzed = i # i starts from 0 total_files_analyzed = (i+1)-1
                print('break after analyzing # files ', total_files_analyzed)
                break
            else:
                total_files_analyzed = i+1

            if i%100 == 0:
                if dataset == 'emulate':
                    print(f'File {i}/{ long_analysis_params["analysis_length"]}')
                else:
                    print(f'File {i}/{len(files)}')

            if dataset == 'emulate':
                data  = np.load(os.path.join(save_dir, file))
                U = data[0,:]
                V = data[1,:]
                Omega_transpose = UV2Omega(U.T, V.T, Kx, Ky, spectral = False)
                Omega = Omega_transpose.T

            elif dataset == 'train' or dataset == 'truth':

                data = loadmat(os.path.join(train_params["data_dir"], 'data', file))
                Omega = data['Omega'].T
                U_transpose, V_transpose = Omega2UV(Omega.T, Kx, Ky, invKsq, spectral = False)
                U, V = U_transpose.T, V_transpose.T

            ## dataset should be float 32 as geenrated by the emulator
            U = U.astype(np.float32)
            V = V.astype(np.float32)
            Omega = Omega.astype(np.float32)

            if long_analysis_params["temporal_mean"]:
                U_mean_temp += U
                V_mean_temp += V
                Omega_mean_temp += Omega

            if long_analysis_params["zonal_mean"] or long_analysis_params["zonal_eof"]:
                U_zonal_temp = np.mean(U, axis=1)
                Omega_zonal_temp = np.mean(Omega, axis=1)        
                U_zonal.append(U_zonal_temp)
                Omega_zonal.append(Omega_zonal_temp)

            if long_analysis_params["div"]:
                div_temp = divergence(U, V)
                div.append(np.mean(np.abs(div_temp)))

            if long_analysis_params["return_period"]:

                U_max.append(np.max(U))
                U_min.append(np.min(U))
                V_max.append(np.max(V))
                V_min.append(np.min(V))
                Omega_max.append(np.max(Omega))
                Omega_min.append(np.min(Omega))

            if long_analysis_params['PDF']:
                # Calculating PDF will may need large memory
                Omega_arr.append(Omega)
                U_arr.append(U)

        if long_analysis_params["temporal_mean"]:

            U_mean = U_mean_temp/total_files_analyzed
            V_mean = V_mean_temp/total_files_analyzed
            Omega_mean = Omega_mean_temp/total_files_analyzed

            print('mean', dataset, total_files_analyzed)


            np.savez(os.path.join(analysis_dir_save, 'temporal_mean.npz'), U_mean=U_mean, V_mean=V_mean, Omega_mean=Omega_mean)

        if long_analysis_params["zonal_eof"] or long_analysis_params["zonal_mean"]:

            U_zonal_mean = np.mean(U_zonal, axis=0)
            U_zonal_anom = np.array(U_zonal) - U_zonal_mean
            EOF_U, PC_U, exp_var_U = manual_eof(U_zonal_anom, long_analysis_params["eof_ncomp"])

            Omega_zonal_mean = np.mean(Omega_zonal, axis=0)
            Omega_zonal_anom = np.array(Omega_zonal) - Omega_zonal_mean
            EOF_Omega, PC_Omega, exp_var_Omega = manual_eof(Omega_zonal_anom, long_analysis_params["eof_ncomp"])

            # ## Scikit-learn

            # pca = PCA(n_components=long_analysis_params["eof_ncomp"])
            # PC_U_sklearn = pca.fit_transform(U_zonal_anom)
            # EOF_U_sklearn = pca.components_.T
            # expvar_U_sklearn = pca.explained_variance_ratio_

            # pca = PCA(n_components=long_analysis_params["eof_ncomp"])
            # PC_Omega_sklearn = pca.fit_transform(Omega_zonal_anom)
            # EOF_Omega_sklearn = pca.components_.T
            # expvar_Omega_sklearn = pca.explained_variance_ratio_

            # ## SVD
            # EOF_U_svd, PC_U_svd, expvar_U_svd = manual_svd_eof(U_zonal_anom)
            # EOF_Omega_svd, PC_Omega_svd, expvar_Omega_svd = manual_svd_eof(Omega_zonal_anom)

            # np.savez(os.path.join(analysis_dir_save, 'zonal_eof.npz'), EOF_U=EOF_U, PC_U=PC_U, exp_var_U=exp_var_U, EOF_Omega=EOF_Omega, PC_Omega=PC_Omega, exp_var_Omega=exp_var_Omega, EOF_U_sklearn=EOF_U_sklearn, PC_U_sklearn=PC_U_sklearn, expvar_U_sklearn=expvar_U_sklearn, EOF_Omega_sklearn=EOF_Omega_sklearn, PC_Omega_sklearn=PC_Omega_sklearn, expvar_Omega_sklearn=expvar_Omega_sklearn, EOF_U_svd=EOF_U_svd, PC_U_svd=PC_U_svd, expvar_U_svd=expvar_U_svd, EOF_Omega_svd=EOF_Omega_svd, PC_Omega_svd=PC_Omega_svd, expvar_Omega_svd=expvar_Omega_svd)
            np.savez(os.path.join(analysis_dir_save, 'zonal_eof.npz'), EOF_U=EOF_U, PC_U=PC_U, exp_var_U=exp_var_U, EOF_Omega=EOF_Omega, PC_Omega=PC_Omega, exp_var_Omega=exp_var_Omega, long_analysis_params=long_analysis_params, dataset_params=dataset_params)
            np.savez(os.path.join(analysis_dir_save, 'zonal_mean.npz'), U_zonal_mean=U_zonal_mean, Omega_zonal_mean=Omega_zonal_mean, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["div"]:
            div = np.array(div, dtype=np.float32) # Torch Emulator data is float32
            np.save(os.path.join(analysis_dir_save, 'div'), div)

        if long_analysis_params["return_period"]:
            np.savez(os.path.join(analysis_dir_save, 'extremes.npz'), U_max=np.asarray(U_max), U_min=np.asarray(U_min), V_max=np.asarray(V_max), V_min=np.asarray(V_min), Omega_max=np.asarray(Omega_max), Omega_min=np.asarray(Omega_min), long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["PDF"]:
            Omega_arr = np.array(Omega_arr)
            Omega_mean, Omega_std, Omega_pdf, Omega_bins, bw_scott = PDF_compute(Omega_arr)

            U_arr = np.array(U_arr)
            U_mean, U_std, U_pdf, U_bins, bw_scott = PDF_compute(U_arr)

            np.savez(os.path.join(analysis_dir_save, 'pdf.npz'), Omega_mean=Omega_mean, Omega_std=Omega_std, Omega_pdf=Omega_pdf, Omega_bins=Omega_bins, bw_scott=bw_scott, U_mean=U_mean, U_std=U_std, U_pdf=U_pdf, U_bins=U_bins, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

    # Plotting and saving figures for long analysis
    notebook_file = "plot_long_analysis_single.ipynb"

    # Ensure the notebook file exists
    try: 
        if os.path.exists(notebook_file):
            run_notebook_as_script(notebook_file)
        else:
            print(f"Notebook file {notebook_file} not found!")

    except Exception as e:
        print(f"Error running notebook")
        print(e)

    if long_analysis_params["video"]:

        print("---------------------- Making Video")

        # Data predicted by the emualtor
        files_emulate = get_npy_files(save_dir)
        files_train = get_mat_files_in_range(os.path.join(train_params["data_dir"],'data'), train_params["train_file_range"])

        plt_save_dir = os.path.join(dataset_params["root_dir"], dataset_params["run_num"], "plots")
        os.makedirs(plt_save_dir, exist_ok=True)
        
        frames = []
        for t in range(long_analysis_params["video_length"]):
            if t%1 == 0:
                fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
                axs = axs.flatten()

                data_emulate = np.load(os.path.join(save_dir, files_emulate[t]))
                U_emulate = data_emulate[0,:]
                V_emulate = data_emulate[1,:]

                data_train = loadmat(os.path.join(train_params["data_dir"], 'data', files_train[3*t]))
                Omega_train = data_train['Omega'].T
                U_transpose, V_transpose = Omega2UV(Omega_train.T, Kx, Ky, invKsq, spectral = False)
                U_train, V_train = U_transpose.T, V_transpose.T

                data = [U_emulate, V_emulate, U_train, V_train]
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
                fig.suptitle(f'{t+1}$\Delta t$')
                fig.savefig('temp_frame.png', bbox_inches='tight')
                plt.close()

                frames.append(imageio.imread('temp_frame.png'))

                if t%1 == 0:
                    print(f'Frame {t}/{long_analysis_params["video_length"]}')

        imageio.mimsave(plt_save_dir + '/Video.gif', frames, fps=15)
