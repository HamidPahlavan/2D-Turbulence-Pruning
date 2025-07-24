import os
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import imageio
from statsmodels.tsa.stattools import acf
from multiprocessing import Pool, cpu_count
from collections import defaultdict

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import UV2Omega, Omega2UV
from py2d.spectra import spectrum_angled_average, spectrum_zonal_average

from analysis.metrics import manual_eof, manual_svd_eof, divergence, PDF_compute
from analysis.rollout import n_step_rollout
from analysis.io_utils import load_numpy_data, get_npy_files, get_mat_files_in_range, run_notebook_as_script, frame_generator
from analysis.plot_config import params


def generate_args(files, dataset, save_dir, data_dir, Kx, Ky, invKsq, long_analysis_params, climatology):
    for fname in files:
        yield (dataset, fname, save_dir, data_dir, Kx, Ky, invKsq, long_analysis_params, climatology)


def process_file(args):
    dataset, fname, save_dir, data_dir, Kx, Ky, invKsq, long_analysis_params, climatology = args
    results = {
        "U_sum": 0.0,
        "V_sum": 0.0,
        "Omega_sum": 0.0,
        "count": 0,
        "U_zonal": [],
        "V_zonal": [],
        "Omega_zonal": [],
        "div": [],
        "U_max": [],
        "U_min": [],
        "V_max": [],
        "V_min": [],
        "Omega_max": [],
        "Omega_min": [],
        "U_max_anom": [],
        "U_min_anom": [],
        "V_max_anom": [],
        "V_min_anom": [],
        "Omega_max_anom": [],
        "Omega_min_anom": [],
        "spectra_U_angular": [],
        "spectra_V_angular": [],
        "spectra_Omega_angular": [],
        "spectra_U_zonal": [],
        "spectra_V_zonal": [],
        "spectra_Omega_zonal": [],
        "U_arr": [],
        "Omega_arr": []
    }

    if dataset == "emulate":
        chunk = np.load(os.path.join(save_dir, fname))
        frames = chunk
    else:
        mat = loadmat(os.path.join(data_dir, "data", fname))
        Omega = mat["Omega"].T.astype(np.float32)
        U_t, V_t = Omega2UV(Omega.T, Kx, Ky, invKsq, spectral=False)
        U, V = U_t.T.astype(np.float32), V_t.T.astype(np.float32)
        frames = [(U, V, Omega)]

    for frame in frames:
        if dataset == "emulate":
            U, V = frame[0], frame[1]
            Omega = UV2Omega(U.T, V.T, Kx, Ky, spectral=False).T
        else:
            U, V, Omega = frame

        results["count"] += 1
        if long_analysis_params["temporal_mean"]:
            results["U_sum"] += U
            results["V_sum"] += V
            results["Omega_sum"] += Omega

        if long_analysis_params["zonal_mean"] or long_analysis_params["zonal_eof_pc"] or \
            long_analysis_params["zonal_U"] or long_analysis_params["zonal_V"] or long_analysis_params["zonal_Omega"]:
            results["U_zonal"].append(np.mean(U, axis=1))
            results["V_zonal"].append(np.mean(V, axis=1))
            results["Omega_zonal"].append(np.mean(Omega, axis=1))

        if long_analysis_params["div"]:
            results["div"].append(np.mean(np.abs(divergence(U, V))))

        if long_analysis_params["return_period"]:
            results["U_max"].append(np.max(U))
            results["U_min"].append(np.min(U))
            results["V_max"].append(np.max(V))
            results["V_min"].append(np.min(V))
            results["Omega_max"].append(np.max(Omega))
            results["Omega_min"].append(np.min(Omega))

        if long_analysis_params["return_period_anomaly"] and climatology is not None:
            U_anom = U - climatology["U"]
            V_anom = V - climatology["V"]
            Omega_anom = Omega - climatology["Omega"]

            results["U_max_anom"].append(np.max(U_anom))
            results["U_min_anom"].append(np.min(U_anom))
            results["V_max_anom"].append(np.max(V_anom))
            results["V_min_anom"].append(np.min(V_anom))
            results["Omega_max_anom"].append(np.max(Omega_anom))
            results["Omega_min_anom"].append(np.min(Omega_anom))

        if long_analysis_params["spectra"]:
            U_abs_hat = np.sqrt(np.fft.fft2(U)*np.conj(np.fft.fft2(U)))
            V_abs_hat = np.sqrt(np.fft.fft2(V)*np.conj(np.fft.fft2(V)))
            Omega_abs_hat = np.sqrt(np.fft.fft2(Omega)*np.conj(np.fft.fft2(Omega)))

            spU, wa = spectrum_angled_average(U_abs_hat, spectral=True)
            spV, _ = spectrum_angled_average(V_abs_hat, spectral=True)
            spO, _ = spectrum_angled_average(Omega_abs_hat, spectral=True)
            results["spectra_U_angular"].append(spU)
            results["spectra_V_angular"].append(spV)
            results["spectra_Omega_angular"].append(spO)
            results["wavenumber_angular"] = wa

            szU, wz = spectrum_zonal_average(U.T)
            szV, _ = spectrum_zonal_average(V.T)
            szO, _ = spectrum_zonal_average(Omega.T)
            results["spectra_U_zonal"].append(szU)
            results["spectra_V_zonal"].append(szV)
            results["spectra_Omega_zonal"].append(szO)
            results["wavenumber_zonal"] = wz

        if long_analysis_params["PDF_U"]:
            results["U_arr"].append(U)

        if long_analysis_params["PDF_Omega"]:
            results["Omega_arr"].append(Omega)

    return results


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

    if long_analysis_params["temporal_mean"] or long_analysis_params["zonal_mean"] or long_analysis_params["zonal_eof_pc"] or long_analysis_params["div"] or \
        long_analysis_params["return_period"] or long_analysis_params["return_period_anomaly"] or long_analysis_params["PDF_U"] or long_analysis_params["PDF_Omega"]:
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

        # Loading climatology for reurn period anomaly calculation
        if long_analysis_params["return_period_anomaly"]:

            try:
                data = np.load(os.path.join(analysis_dir_save, 'temporal_mean.npz'))

                U_sample_mean_climatology = data['U_sample_mean']
                V_sample_mean_climatology = data['V_sample_mean']
                Omega_sample_mean_climatology = data['Omega_sample_mean']

                climatology = {
                    "U": U_sample_mean_climatology,
                    "V": V_sample_mean_climatology,
                    "Omega": Omega_sample_mean_climatology
                }

            except FileNotFoundError:
                print(f"File not found: {os.path.join(analysis_dir_save, 'temporal_mean.npz')}")
                print("Skipping return period anomaly calculation.")
                long_analysis_params["return_period_anomaly"] = False

                climatology = None
        else:
                climatology = None

        os.makedirs(analysis_dir_save, exist_ok=True)


        # initialize all your accumulators here…
        total_files_analyzed = 0

        args_gen = generate_args(files, dataset, save_dir, train_params["data_dir"], Kx, Ky, invKsq, long_analysis_params, climatology)
        
        agg_results = defaultdict(list)

        print(f'CPU COUNT: {min(12, cpu_count())}')
        with Pool(processes=min(12, cpu_count())) as pool:
            for partial in pool.imap(process_file, args_gen, chunksize=1):

                count = partial.pop("count", 0)
                total_files_analyzed += count  

                if dataset == 'emulate' and total_files_analyzed >= long_analysis_params["analysis_length"]:
                    print('break after analyzing # files ', total_files_analyzed)
                    break

                if total_files_analyzed%100 == 0:
                    if dataset == 'emulate':
                        print(f'File {total_files_analyzed}/{long_analysis_params["analysis_length"]}')
                    else:
                        print(f'File {total_files_analyzed}/{len(files)}')

                for key, value in partial.items():

                    if isinstance(value, list):
                        agg_results[key].extend(value)
                    else:
                        agg_results[key].append(value)

        # Aggregate and save results
        print(f'Completed aggregation. Saving Results for {dataset}')
        if long_analysis_params["temporal_mean"]:

            U_mean = np.sum(agg_results["U_sum"], axis=0) / total_files_analyzed
            V_mean = np.sum(agg_results["V_sum"], axis=0) / total_files_analyzed
            Omega_mean = np.sum(agg_results["Omega_sum"], axis=0) / total_files_analyzed

            np.savez(os.path.join(analysis_dir_save, 'temporal_mean.npz'),
                     U_sample_mean=U_mean, V_sample_mean=V_mean, Omega_sample_mean=Omega_mean,
                     long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["spectra"]:

            spectra_U_angular_avg = np.mean(agg_results["spectra_U_angular"], axis=0)
            spectra_V_angular_avg = np.mean(agg_results["spectra_V_angular"], axis=0)
            spectra_Omega_angular_avg = np.mean(agg_results["spectra_Omega_angular"], axis=0)
            wavenumber_angular_avg = agg_results["wavenumber_angular"][0]

            spectra_U_zonal_avg = np.mean(agg_results["spectra_U_zonal"], axis=0)
            spectra_V_zonal_avg = np.mean(agg_results["spectra_V_zonal"], axis=0)
            spectra_Omega_zonal_avg = np.mean(agg_results["spectra_Omega_zonal"], axis=0)
            wavenumber_zonal_avg = agg_results["wavenumber_zonal"][0]

            np.savez(os.path.join(analysis_dir_save, 'spectra.npz'), 
                spectra_U_angular_avg=spectra_U_angular_avg, spectra_V_angular_avg=spectra_V_angular_avg, spectra_Omega_angular_avg=spectra_Omega_angular_avg, wavenumber_angular_avg=wavenumber_angular_avg, 
                spectra_U_zonal_avg=spectra_U_zonal_avg, spectra_V_zonal_avg=spectra_V_zonal_avg, spectra_Omega_zonal_avg=spectra_Omega_zonal_avg, wavenumber_zonal_avg=wavenumber_zonal_avg,
                long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["zonal_eof_pc"] or long_analysis_params["zonal_mean"] or \
            long_analysis_params["zonal_U"] or long_analysis_params["zonal_V"] or long_analysis_params["zonal_Omega"]:

            U_zonal_mean = np.mean(agg_results["U_zonal"], axis=0)
            Omega_zonal_mean = np.mean(agg_results["Omega_zonal"], axis=0)
            V_zonal_mean = np.mean(agg_results["V_zonal"], axis=0)

            np.savez(os.path.join(analysis_dir_save, 'zonal_mean.npz'), U_zonal_mean=U_zonal_mean, Omega_zonal_mean=Omega_zonal_mean, V_zonal_mean=V_zonal_mean, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

            np.savez(os.path.join(analysis_dir_save, 'zonal_U.npz'), U_zonal=np.asarray(agg_results["U_zonal"]), long_analysis_params=long_analysis_params, dataset_params=dataset_params)
            np.savez(os.path.join(analysis_dir_save, 'zonal_V.npz'), V_zonal=np.asarray(agg_results["V_zonal"]), long_analysis_params=long_analysis_params, dataset_params=dataset_params)
            np.savez(os.path.join(analysis_dir_save, 'zonal_Omega.npz'), Omega_zonal=np.asarray(agg_results["Omega_zonal"]), long_analysis_params=long_analysis_params, dataset_params=dataset_params)

            if long_analysis_params["zonal_eof_pc"]:
                U_zonal_stack = np.stack(agg_results["U_zonal"], axis=0)        # shape (T, Nx)
                Omega_zonal_stack = np.stack(agg_results["Omega_zonal"], axis=0)  # shape (T, Nx)

                U_zonal_mean = np.mean(U_zonal_stack, axis=0)
                Omega_zonal_mean = np.mean(Omega_zonal_stack, axis=0)

                U_zonal_anom = U_zonal_stack - U_zonal_mean  # shape (T, Nx)
                Omega_zonal_anom = Omega_zonal_stack - Omega_zonal_mean

                EOF_U, PC_U, exp_var_U = manual_eof(U_zonal_anom, long_analysis_params["eof_ncomp"])
                EOF_Omega, PC_Omega, exp_var_Omega = manual_eof(Omega_zonal_anom, long_analysis_params["eof_ncomp"])

                PC_acf_U = []
                PC_acf_Omega = []

                if dataset in ['train', 'truth']:
                    n_lags = train_params["target_step"] * long_analysis_params["PC_autocorr_nlags"]
                else:
                    n_lags = long_analysis_params["PC_autocorr_nlags"]

                for i in range(long_analysis_params["eof_ncomp"]):
                    acf_i, confint_i = acf(PC_U[:, i], nlags=n_lags, alpha=0.5)
                    PC_acf_U.append({"acf": acf_i, "confint": confint_i})

                    acf_i, confint_i = acf(PC_Omega[:, i], nlags=n_lags, alpha=0.5)
                    PC_acf_Omega.append({"acf": acf_i, "confint": confint_i})

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

                # np.savez(os.path.join(analysis_dir_save, 'zonal_eof_pc.npz'), EOF_U=EOF_U, PC_U=PC_U, exp_var_U=exp_var_U, EOF_Omega=EOF_Omega, PC_Omega=PC_Omega, exp_var_Omega=exp_var_Omega, EOF_U_sklearn=EOF_U_sklearn, PC_U_sklearn=PC_U_sklearn, expvar_U_sklearn=expvar_U_sklearn, EOF_Omega_sklearn=EOF_Omega_sklearn, PC_Omega_sklearn=PC_Omega_sklearn, expvar_Omega_sklearn=expvar_Omega_sklearn, EOF_U_svd=EOF_U_svd, PC_U_svd=PC_U_svd, expvar_U_svd=expvar_U_svd, EOF_Omega_svd=EOF_Omega_svd, PC_Omega_svd=PC_Omega_svd, expvar_Omega_svd=expvar_Omega_svd)
                np.savez(os.path.join(analysis_dir_save, 'zonal_eof_pc.npz'), U_eofs=EOF_U, U_pc=PC_U, U_expvar=exp_var_U, U_pc_acf=PC_acf_U, Omega_eofs=EOF_Omega, Omega_PC=PC_Omega, Omega_expvar=exp_var_Omega, Omega_pc_acf=PC_acf_Omega, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["div"]:
            div = np.array(agg_results["div"], dtype=np.float32) # Torch Emulator data is float32
            np.save(os.path.join(analysis_dir_save, 'div'), div)

        if long_analysis_params["return_period"]:
            np.savez(os.path.join(analysis_dir_save, 'extremes.npz'), U_max_arr=np.asarray(agg_results["U_max"]), U_min_arr=np.asarray(agg_results["U_min"]), V_max_arr=np.asarray(agg_results["V_max"]), V_min_arr=np.asarray(agg_results["V_min"]), Omega_max_arr=np.asarray(agg_results["Omega_max"]), Omega_min_arr=np.asarray(agg_results["Omega_min"]), long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["return_period_anomaly"]:
            np.savez(os.path.join(analysis_dir_save, 'extremes_anom.npz'), U_max_arr=np.asarray(agg_results["U_max_anom"]), U_min_arr=np.asarray(agg_results["U_min_anom"]), V_max_arr=np.asarray(agg_results["V_max_anom"]), V_min_arr=np.asarray(agg_results["V_min_anom"]), Omega_max_arr=np.asarray(agg_results["Omega_max_anom"]), Omega_min_arr=np.asarray(agg_results["Omega_min_anom"]), long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["PDF_U"]:
            U_arr = np.array(agg_results["U_arr"])
            U_mean, U_std, U_pdf, U_bins, bw_scott = PDF_compute(U_arr)
            np.savez(os.path.join(analysis_dir_save, 'PDF_U.npz'), bw_scott=bw_scott, U_mean=U_mean, U_std=U_std, U_pdf=U_pdf, U_bins=U_bins, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

        if long_analysis_params["PDF_Omega"]:
            Omega_arr = np.array(agg_results["Omega_arr"])
            Omega_mean, Omega_std, Omega_pdf, Omega_bins, bw_scott = PDF_compute(Omega_arr)
            np.savez(os.path.join(analysis_dir_save, 'PDF_Omega.npz'), Omega_mean=Omega_mean, Omega_std=Omega_std, Omega_pdf=Omega_pdf, Omega_bins=Omega_bins, bw_scott=bw_scott, U_mean=U_mean, long_analysis_params=long_analysis_params, dataset_params=dataset_params)

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

        files_emulate = get_npy_files(save_dir)
        files_train   = get_mat_files_in_range(
                            os.path.join(train_params["data_dir"], 'data'),
                            train_params["train_file_range"]
                        )

        # build our two frame‐by‐frame generators
        frame_gen_emulate = frame_generator(
            "emulate", files_emulate, save_dir,
            train_params["data_dir"], Kx, Ky, invKsq
        )
        frame_gen_train = frame_generator(
            "train", files_train, save_dir,
            train_params["data_dir"], Kx, Ky, invKsq
        )

        plt_save_dir = os.path.join(
            dataset_params["root_dir"],
            dataset_params["run_num"],
            "plots"
        )
        os.makedirs(plt_save_dir, exist_ok=True)

        frames = []
        for frame_idx in range(long_analysis_params["video_length"]):
            try:
                U_emulate, V_emulate, _ = next(frame_gen_emulate)
            except StopIteration:
                print("Emulator generator exhausted at frame", frame_idx)
                break
            
            for i in range(train_params["target_step"]):
                try:
                    U_train, V_train, _ = next(frame_gen_train)
                except StopIteration:
                    print("Train generator exhausted at frame", frame_idx)
                    break

            # now make your 2×2 plot
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            plt.rcParams.update(params)
            axs = axs.flatten()

            data_list = [U_emulate, V_emulate, U_train, V_train]
            titles    = [r'$u$ Emulator', r'$v$ Emulator',
                        r'$u$ Truth',    r'$v$ Truth']

            for ax, d, title in zip(axs, data_list, titles):
                im = ax.imshow(d, cmap='bwr', vmin=-5, vmax=5, aspect='equal')
                L = d.shape[-1]
                ax.set_xticks([0, L/2, L], [0, r'$\pi$', r'$2\pi$'])
                ax.set_yticks([0, L/2, L], [0, r'$\pi$', r'$2\pi$'])
                ax.set_title(title)

            fig.subplots_adjust(right=0.85)
            cax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cax)
            fig.suptitle(f"{frame_idx+1} $\Delta t$")

            tmpfile = os.path.join(plt_save_dir, f"temp.png")
            fig.savefig(tmpfile, dpi=100)
            plt.close(fig)

            frames.append(imageio.imread(tmpfile))
            print(f"Rendered frame {frame_idx+1}/{long_analysis_params['video_length']}")

        # write the gif
        gif_path = os.path.join(plt_save_dir, "Video.gif")
        imageio.mimsave(gif_path, frames, fps=15)
        print("Saved video to", gif_path)
