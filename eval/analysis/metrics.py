import numpy as np
from scipy.stats import gaussian_kde, pearsonr

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import Omega2Psi, Psi2UV

def get_rmse(y, y_hat, climo=None):
    
    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    y_anom = y - climo
    y_hat_anom = y_hat - climo
    err = (y_anom - y_hat_anom) ** 2
    err = np.mean(err, axis=(-1, -2))
    rmse = np.sqrt(err)
    
    return rmse

def get_acc(y, y_hat, climo=None):
    """
    Args:
        y, y_hat: [B=n_steps, X, Y]
    """

    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    corr = []
    for i in range(y.shape[0]):
        y_i = y[i] - climo
        y_hat_i = y_hat[i] - climo
        #acc = (
        #        np.sum(y_i * y_hat_i) /
        #        np.sqrt(
        #            np.sum(y_i ** 2) * np.sum(y_hat_i ** 2)
        #            )
        #        )
        #corr.append(acc)
        corr.append(np.corrcoef(y_i.flatten(), y_hat_i.flatten())[1, 0])

    return np.array(corr)

def spectrum_zonal_average_2D_FHIT(U,V):
  """
  Zonal averaged spectrum for 2D flow variables

  Args:
    U: 2D square matrix, velocity
    V: 2D square matrix, velocity

  Returns:
    E_hat: 1D array
    wavenumber: 1D array
  """

  # Check input shape
  if U.ndim != 2 and V.ndim != 2:
    raise ValueError("Input flow variable is not 2D. Please input 2D matrix.")
  if U.shape[0] != U.shape[1] and V.shape[0] != V.shape[1]:
    raise ValueError("Dimension mismatch for flow variable. Flow variable should be a square matrix.")

  N_LES = U.shape[0]

  # fft of velocities along the first dimension
  U_hat = np.fft.rfft(U, axis=1)/ N_LES  #axis=1
  V_hat = np.fft.rfft(V, axis=1)/ N_LES  #axis=1

  U_hat[1:] = 2*U_hat[1:] # Multiply by 2 to account for the negative wavenumbers
  V_hat[1:] = 2*V_hat[1:] 

  # Energy
  #E_hat = 0.5 * U_hat * np.conj(U_hat) + 0.5 * V_hat * np.conj(V_hat)
  E_hat = U_hat

  # Average over the second dimension
  # Multiplying by 2 to account for the negative wavenumbers
  E_hat = np.mean(np.abs(E_hat), axis=0) #axis=0
  wavenumbers = np.linspace(0, N_LES//2, N_LES//2+1)

  return E_hat, wavenumbers

def get_spectra(U, V):
    """`
    Args:
        U, V: [B=n_steps, X, Y]
    Returns:
        spectra: [B=n_steps, k]
    """
    
    spectra = []
    for i in range(U.shape[0]):
        E_hat, wavenumbers = spectrum_zonal_average_2D_FHIT(U[i], V[i])
        spectra.append(E_hat)

    spectra = np.stack(spectra, axis=0)

    return spectra, wavenumbers


# def get_zonal_PCA(zdata, n_comp=1):
#     """
#     Compute PCA of zonally-averaged fields.
#     Args:
#         data: [B=n_steps, X, Y] np.array of data
#     Returns:
#         pcs: [B, n_comp]
#         eofs: [n_comp, X]
#     """

#     # Zonally average data
#     print(f'zdata.shape: {zdata.shape}')

#     # initiate PCA
#     pca = PCA(n_components=n_comp)

#     pcs = pca.fit_transform(zdata)      # [B, n_comp]
#     eofs = pca.components_              # [n_comp, X]
#     print(f'pcs.shape: {pcs.shape}')
#     print(f'eofs.shape: {eofs.shape}')

#     return pcs, eofs

def manual_eof(X_demeaned, n_comp=1):
    # Step 1: Demean the data
    # X_demeaned = X - np.mean(X, axis=0)
    # Step 2: Covariance matrix
    C = np.cov(X_demeaned, rowvar=False)
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Step 4: Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # First three EOFs and PCs
    EOFs_manual = eigenvectors[:, :n_comp]
    PCs_manual = X_demeaned @ EOFs_manual

    # Calculate the explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    return EOFs_manual, PCs_manual, np.flip(explained_variance)[:3]

# Method 2: Manual calculation of EOFs and PCs using SVD
def manual_svd_eof(X_demeaned):

    # Step 2: Apply SVD on the demeaned data
    # U: left singular vectors (PCs)
    # s: singular values
    # Vt: right singular vectors (EOFs, transposed)
    U, s, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    
    # Step 3: Extract the first three EOFs and PCs
    EOFs_svd = Vt.T    # EOFs and transpose to shape (N, 3)
    PCs_svd = U * s # Scale U by the singular values to get PCs
    
    # Step 4: Calculate explained variance
    total_variance = np.sum(s ** 2)
    explained_variance = (s ** 2) / total_variance
    
    return EOFs_svd, PCs_svd, explained_variance

def get_div(U, V):
    """
    Args:
        U: [B=n_steps, X, Y] 
        V: [B=n_steps, X, Y]
    Returns:
        div: [B,] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[1], U.shape[2]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    div = []
    for i in range(U.shape[0]):
        Dx = derivative(U[i,:,:], [0,1], Kx, Ky, spectral=False) #[1,0]
        Dy = derivative(V[i,:,:], [1,0], Kx, Ky, spectral=False) #[0,1]
        div.append(np.mean(np.abs(Dx + Dy)))

    return np.array(div)

def divergence(U, V):
    """
    Args:
        U: [X, Y] 
        V: [X, Y]
    Returns:
        div: [X,Y] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[0], U.shape[1]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    Ux = derivative(U.T, [1,0], Kx, Ky, spectral=False) #[1,0]
    Vy = derivative(V.T, [0,1], Kx, Ky, spectral=False) #[0,1]
    div = Ux + Vy

    return div

def PDF_compute(data, bw_factor=1):
    data_arr = np.array(data).flatten()
    del data

    # Calculate mean and standard deviation
    data_mean, data_std = np.mean(data_arr), np.std(data_arr)

    # Define bins within 10 standard deviations from the mean, but also limit them within the range of the data
    bin_max = np.min(np.abs([np.min(data_arr), np.max(data_arr)]))
    bin_min = -bin_max
    bins = np.linspace(bin_min, bin_max, 100)

    print('PDF Clculation')
    print('bin min', bin_min)
    print('bin max', bin_max)
    print('data Shape', data_arr.shape)
    print('data mean', data_mean)
    print('data_std', data_std)
    print('Total nans', np.sum(np.isnan(data_arr)))

    # Compute PDF using Scipy
    bw1 = bw_factor*(data_arr.shape[0])**(-1/5) # custom bw method scott method n**(-1/5)
    kde = gaussian_kde(data_arr, bw_method=bw1)

    # # Define a range over which to evaluate the density
    data_bins = bins
    bw_scott = kde.factor
    # # Evaluate the density over the range
    data_pdf = kde.evaluate(data_bins)

    return data_mean, data_std, data_pdf, data_bins, bw_scott

def empirical_return_period(X, dt=1):
    """Get empirical return period. Returns 'return_period' and 'data_amplitude';
    both are used to plot empirical return period."""

    # Empirical return period
    data_amplitude = np.sort(X)
    n = len(X)
    m = np.arange(1, n + 1)
    cdf_empirical = m / (n + 1)
    return_period = 1 / (1 - cdf_empirical)

    return return_period*dt, data_amplitude


def ensemble_return_period_amplitude(data, dt=1, bins_num=50, confidence_level_type='percentile', confidence_level=25):
    '''
    Calculate return period and error band using ensemble of data. The error bands are calculated for data amplitude
    data: 2D array of data [ensemble, time]
    dt: time step
    bins_num: number of bins for binning the data
    confidence_level_type: 'percentile' / 'std'
        Determines the method for calculating the confidence interval
    confidence_level: level (For confidence_level_type='percentile', % e.g.:25, 50, 75;'std', number of standard deviations e.g.: 1, 2, 3)
    '''
    
    return_period_arr = []
    data_amplitude_arr = []

    number_ensemble = data.shape[0]
    total_data_points = data.shape[1]

    for i in range(number_ensemble):
        data_ensemble = data[i, :]
        return_period, data_amplitude = empirical_return_period(data_ensemble, dt=dt)
        # print(i, data_amplitude.shape, return_period.shape)
        return_period_arr.append(return_period)
        data_amplitude_arr.append(data_amplitude)

    # Error band
    
    bin_min = np.min(return_period_arr)
    bin_max = np.max(return_period_arr)
    bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=bins_num)

    print('Number of ensembles:', number_ensemble)
    print('bins:', bin_min, bin_max)

    data_amplitude_interp_arr = []


    for i in range(number_ensemble):
        data_amplitude_interp_arr.append(np.interp(bins, return_period_arr[i], data_amplitude_arr[i]))
        # print(i)

    if confidence_level_type == 'percentile':
        mean_data_amplitude_interp, lb_data_amplitude_interp, ub_data_amplitude_interp = percentile_data(
            np.asarray(data_amplitude_interp_arr), percentile=confidence_level)
    elif confidence_level_type == 'std':
        mean_data_amplitude_interp, lb_data_amplitude_interp, ub_data_amplitude_interp = std_dev_data(
            np.asarray(data_amplitude_interp_arr), std_dev=confidence_level)

    return mean_data_amplitude_interp, lb_data_amplitude_interp, ub_data_amplitude_interp, bins

def percentile_data(data, percentile):
    """
    Calculate error bands and
    return the lower/upper bounds in percentile

    Parameters:
    -----------
    data : np.ndarray
        2D NumPy array of shape (N, samples)
    percentile : float
        A number between 0 and 100 (typically <= 50 for symmetrical bounds)
    
    Returns:
    --------
    means : np.ndarray
        Array of sample means (shape: N)
    lower_bounds: np.ndarray
        Array of lower bounds in percentage relative to the mean (shape: N)
    upper_bounds: np.ndarray
        Array of upper bounds in percentage relative to the mean (shape: N)
    """
    # Mean of each row
    means = np.mean(data, axis=0)
    
    # Lower (p-th) and upper ((100-p)-th) percentiles of each row
    lower_vals = np.percentile(data, percentile, axis=0)
    upper_vals = np.percentile(data, 100 - percentile, axis=0)

    print(data.shape)
    # print(lower_vals.shape, upper_vals.shape)
    # print(lower_vals, upper_vals)
    
    # Calculate percentage difference relative to the mean
    # (m - L)/m * 100 for lower, (U - m)/m * 100 for upper
    # lower_bounds = 100.0 * (means - lower_vals) / means
    # upper_bounds = 100.0 * (upper_vals - means) / means
    
    return means, lower_vals, upper_vals

def std_dev_data(data, std_dev=1):
    """
    Calculate error bands and
    return the lower/upper bounds in percentile

    Parameters:
    -----------
    data : np.ndarray
        2D NumPy array of shape (N, samples)
    std_dev : float
        A number between 0 and 100 (typically <= 50 for symmetrical bounds)
    
    Returns:
    --------
    means : np.ndarray
        Array of sample means (shape: N)
    lower_bounds: np.ndarray
        Array of lower bounds in percentage relative to the mean (shape: N)
    upper_bounds: np.ndarray
        Array of upper bounds in percentage relative to the mean (shape: N)
    """
    # Mean of each row
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Lower (p-th) and upper ((100-p)-th) percentiles of each row
    lower_vals = means - stds*std_dev
    upper_vals = means + stds*std_dev
    
    return means, lower_vals, upper_vals

def corr_truth_train_model(truth, train, model):
    # Correlation between truth, train, model fields
    corr_truth_train, _ = pearsonr(truth.flatten(), train.flatten())
    corr_truth_model, _ = pearsonr(truth.flatten(), model.flatten())
    corr_train_model, _ = pearsonr(train.flatten(), model.flatten())
    return np.round(corr_truth_train,2), np.round(corr_truth_model,2), np.round(corr_train_model,2)