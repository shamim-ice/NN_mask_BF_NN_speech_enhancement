import numpy as np


def apply_beamformer(beamformer, complex_spectrum):
    number_of_channels, number_of_bins = np.shape(complex_spectrum)
    complex_spectrum = np.expand_dims(complex_spectrum, 1)
    enhanced_spectrum = np.zeros((1, number_of_bins), dtype=np.float)
    # print('bin', self.number_of_bins)
    # x = np.conjugate(beamformer[:, 0]).T
    # print(x.shape, x)
    # y = complex_spectrum[:, :, 0]
    # print(y.shape, y)
    for f in range(0, number_of_bins):
        enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
    return enhanced_spectrum

def mvdr_beamforming_with_mask(signals, mask):
    """
    Perform MVDR beamforming on a set of signals using a mask.
    signals: array of shape (num_sensors, num_samples)
    mask: binary mask indicating speech presence
    """
    num_sensors, num_samples = signals.shape

    # Calculate the sample covariance matrix
    sample_cov = np.cov(signals)
    #print('cov',sample_cov.shape,sample_cov)

    # Calculate the noise subspace
    eig_vals, eig_vecs = np.linalg.eig(sample_cov)
    #print('eig vec',eig_vecs.shape,eig_vecs,eig_vals)
    noise_subspace = eig_vecs[:, eig_vals < 1e-3]

    #print('n subspace',noise_subspace.shape,noise_subspace)

    # Calculate the spatial filter
    #x = np.dot(noise_subspace, noise_subspace.conj().T)
    #print('x',x.shape,x)
    #print('mask bf',mask.shape)
    spatial_filter = np.dot(np.dot(noise_subspace, noise_subspace.conj().T), mask)
    #print('noise subspace',noise_subspace.conj().T,noise_subspace.T)
    #print('spatial filter',spatial_filter.shape,spatial_filter)
    spatial_filter /= np.dot(np.dot(noise_subspace, noise_subspace.T), mask)
    #print('spatial filter', spatial_filter.shape, spatial_filter)

    # Apply the spatial filter to the signals
    output = apply_beamformer(spatial_filter,signals) #np.dot(spatial_filter[0,:], signals)

    return output
