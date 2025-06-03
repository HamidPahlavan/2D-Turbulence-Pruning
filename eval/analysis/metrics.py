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
  E_hat = np.mean(np.abs(E_hat) ** 2, axis=0) #axis=0
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

def return_period_empirical(X, dt=1):
    """
    Calculate the empirical return period for a dataset.

    Args:
        X: 1D array-like, input data (e.g., time series of amplitudes)
        dt: float, time step (default=1)

    Returns:
        return_period: 1D array, empirical return period for each data point (same length as X)
        data_amplitude: 1D array, sorted data amplitudes (ascending order)
    """

    # Sort the data in ascending order
    data_amplitude = np.sort(X)
    n = len(X)
    # Rank order (1-based)
    m = np.arange(1, n + 1)
    # Empirical cumulative distribution function (CDF)
    cdf_empirical = m / (n + 1)
    # Empirical return period formula: T = 1 / (1 - F)
    return_period = 1 / (1 - cdf_empirical)

    # Scale return period by time step
    return return_period * dt, data_amplitude

def return_period_bins(data, dt=1, bins_num=100):
    """
    Calculate return period of data, bin it and interpolate the amplitude values to the bins.

    Args:
        data: 1D array of data [time]
        dt: time step
        bins_num: number of bins for binning the data

    Returns:
        bins: array of bin edges (return periods)
        interp_data_amplitude: interpolated amplitude values at each bin
    """

    # Compute empirical return period and sorted data amplitude
    return_period, data_amplitude = return_period_empirical(data, dt=dt)

    # Set bin range 
    bin_min = np.min(return_period)
    bin_max = np.max(return_period)
    # Create logarithmically spaced bins for return period
    bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=bins_num)

    # Interpolate the amplitude values to the bins
    interp_data_amplitude = np.interp(bins, return_period, data_amplitude)

    return bins, interp_data_amplitude


def ensemble_return_period_amplitude(data, dt=1, bins_num=100, central_tendency='mean', error_bands='std'):
    '''
    Calculate return period and error band using ensemble of data. The error bands are calculated for data amplitude.

    Args:
        data: 2D array of data [ensemble, time]
        dt: time step
        bins_num: number of bins for binning the data
        central_tendency: 'mean' or 'median'
            Determines the central tendency for the amplitude (default: 'mean')
        error_bands: 'std', '50ci', '95ci', or None
            Determines the method for calculating the error/confidence interval

    Returns:
        bins: array of bin edges (return periods)
        central_data_amplitude_interp: central tendency (mean/median) of amplitude at each bin
        lb_data_amplitude_interp: lower bound of error band at each bin
        ub_data_amplitude_interp: upper bound of error band at each bin
    '''

    # Arrays to store return periods and amplitudes for each ensemble member
    return_period_arr = []
    data_amplitude_arr = []

    number_ensemble = data.shape[0]
    total_data_points = data.shape[1]

    # Compute empirical return period and amplitude for each ensemble member
    for i in range(number_ensemble):
        data_ensemble = data[i, :]
        return_period, data_amplitude = return_period_empirical(data_ensemble, dt=dt)
        return_period_arr.append(return_period)
        data_amplitude_arr.append(data_amplitude)

    # Define bins for return period (logarithmic spacing)
    bin_min = np.min(return_period_arr)
    bin_max = np.max(return_period_arr)
    bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=bins_num)

    # Interpolate amplitude values to the common bins for each ensemble member
    data_amplitude_interp_arr = []
    for i in range(number_ensemble):
        data_amplitude_interp_arr.append(np.interp(bins, return_period_arr[i], data_amplitude_arr[i]))

    # Compute central tendency (mean or median) across the ensemble
    if central_tendency == 'mean':
        central_data_amplitude_interp = np.mean(data_amplitude_interp_arr, axis=0)
    elif central_tendency == 'median':
        central_data_amplitude_interp = np.median(data_amplitude_interp_arr, axis=0)

    # Compute error bands (confidence intervals or standard deviation)
    if error_bands in ['50ci', '95ci']:
        if error_bands == '50ci':
            confidence_level = 25
        elif error_bands == '95ci':
            confidence_level = 2.5

        # Use percentiles for error bands
        _, lb_data_amplitude_interp, ub_data_amplitude_interp = percentile_data(
            np.asarray(data_amplitude_interp_arr), percentile=confidence_level)

    elif error_bands == 'std':
        # Use standard deviation for error bands
        _, lb_data_amplitude_interp, ub_data_amplitude_interp = std_dev_data(
            np.asarray(data_amplitude_interp_arr), std_dev=1)

    elif error_bands is None:
        # No error bands requested
        return bins, central_data_amplitude_interp, None, None

    return bins, central_data_amplitude_interp, lb_data_amplitude_interp, ub_data_amplitude_interp

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