
import numpy as np
import copy
import matplotlib.pyplot as pl
from . import bimutil

class complexGMM_mvdr:
    
    def __init__(self,
                 sampling_frequency,
                 fft_length,
                 fft_shift,
                 #number_of_EM_iterate,
                 #min_segment_dur,
                 condition_number_inv_threshold=10**(-6),
                 scm_inv_threshold=10**(-10),
                 beamformer_inv_threshold=10**(-4)):
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        #self.number_of_EM_iterate=number_of_EM_iterate
        #self.min_segment_dur=min_segment_dur
        self.condition_number_inv_threshold=condition_number_inv_threshold
        self.scm_inv_threshold=scm_inv_threshold
        self.beamformer_inv_threshold=beamformer_inv_threshold

    

    def get_spatial_correlation_matrix_from_mask_for_LSTM(self, speech_data, speech_mask, noise_mask, less_frame):
        """
        if noise_mask.any() == None:
            print('make_noise_mask')
            noise_mask = (1 - speech_mask)+0.01
        else:
            noise_mask = noise_mask.T
        """
        #print(np.shape(speech_mask), np.shape(noise_mask))
        #print('data',speech_data.shape,speech_data)
        complex_spectrum, _ = bimutil.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        #print(complex_spectrum.shape)
        tmp_complex_spectrum  = copy.deepcopy(complex_spectrum)
        #return tmp_complex_spectrum
        # safe guard for difference size between speakerbeam's mask and complex spectrum
        
        # ad-hock selection 5/14
        complex_spectrum = complex_spectrum[:, less_frame:-(less_frame + 1), :]
        speech_mask = speech_mask[:, less_frame:-(less_frame + 1)]
        noise_mask = noise_mask[:, less_frame:-(less_frame + 1)]
        #print(np.shape(speech_mask), np.shape(noise_mask),np.shape(complex_spectrum))
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        #print('bin frames', number_of_bins, number_of_frames)
        _, number_of_frames_on_speakerbeam_mask = np.shape(noise_mask)
        
        if number_of_frames != number_of_frames_on_speakerbeam_mask:
            maximum_number_of_frames = np.min([number_of_frames, number_of_frames_on_speakerbeam_mask])
            complex_spectrum = complex_spectrum[:, 0:maximum_number_of_frames, :]
            speech_mask = speech_mask[:, 0:maximum_number_of_frames]
            noise_mask = noise_mask[:, 0:maximum_number_of_frames]
            number_of_frames = maximum_number_of_frames       
        noise_mask = np.fliplr(noise_mask.T)
        speech_mask = np.fliplr(speech_mask.T)
        """
        pl.figure()
        
        pl.imshow(noise_mask, aspect='auto')
        pl.title('n_mask_median')
        pl.figure()
        pl.imshow(speech_mask, aspect='auto')
        pl.title('s_mask_median')
        pl.show()
        """
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        R_xn = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)                
        # init R_noisy and R_noise
        '''for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                print('noise sum', np.sum(noise_mask[:, f], dtype=np.complex64))
                print('speech sum', np.sum(speech_mask[:, f], dtype=np.complex64))'''
        #print('mask',speech_mask.shape,speech_mask[250,:10],noise_mask[250,:10])
        #print('bin frames',number_of_bins,number_of_frames)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                #print('h',h)
                yyh[:, :, t, f] = h
                R_xn[:, :, f] = R_xn[:, :, f] + h
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R_x = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        #print('h',yyh.shape,yyh)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + noise_mask[t, f] * yyh[:, :, t, f]
                R_x[:, :, f] = R_x[:, :, f] + speech_mask[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(noise_mask[:, f], dtype=np.complex64)
            R_x[:, :, f] = R_x[:, :, f] / np.sum(speech_mask[:, f], dtype=np.complex64)
        #R_x = R_xn - R_n

        return (tmp_complex_spectrum, R_x, R_n, noise_mask, speech_mask)

    

    def get_mvdr_beamformer(self, R_x, R_n):
        number_of_channels, _, number_of_bins = np.shape(R_x)
        beamformer = np.ones((number_of_channels, number_of_bins), dtype=np.complex64)        
        for f in range(0, number_of_bins):
            _, eigen_vector = np.linalg.eig(R_x[:, :, f])
            steering_vector = eigen_vector[:, 0]
            Rn_inv = np.linalg.pinv(R_n[:, :, f], rcond=self.beamformer_inv_threshold)
            w1 = np.matmul(Rn_inv, steering_vector)
            w2 = np.matmul(np.conjugate(steering_vector).T, Rn_inv)
            w2 = np.matmul(w2, steering_vector)
            w2 = np.reshape(w2, [1, 1])
            w = w1 / w2
            w = np.reshape(w, number_of_channels)
            beamformer[:, f] = w
        return (beamformer, steering_vector)
    
    def get_mvdr_beamformer_by_maxsnr(self, R_x, R_n):
        '''
        Improved MVDR beamforming using single-channel mask
                prediction networks [Erdogan, 2016]
        '''
        
        number_of_channels, _, number_of_bins = np.shape(R_x)

        # beamformer >> (selectablebeam, number_of_channels, number_of_bins)
        beamformer = np.ones((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)  
        # all channles beamformer
        selected_SNR = np.zeros(number_of_channels, dtype=np.float32)
        for c in range(0, number_of_channels):
            r = np.zeros(number_of_channels, dtype=np.complex64)
            r[c] = 1
            #print('channel', c+1)
            #print(r)
            for f in range(0, number_of_bins):
                Rn_inv = np.linalg.pinv(R_n[:, :, f], rcond=self.beamformer_inv_threshold)
                w1_1 = np.matmul(Rn_inv, R_x[:, :, f])
                w1 = np.matmul(w1_1, r)
                #print('w1_1',w1_1)
                #print('w1',w1)
                # normalize factor
                w2 = np.trace(w1_1)

                w2 = np.reshape(w2, [1, 1])
                w = w1 / w2
                w = np.reshape(w, number_of_channels)
                beamformer[c, :, f] = w
            w1_sum = 0
            w2_sum = 0
            for f2 in range(0, number_of_bins):
                snr_post_w1 = np.matmul(np.conjugate(beamformer[c, :, f2]).T, R_x[:, :, f2])
                snr_post_w1 = np.matmul(snr_post_w1, beamformer[c, :, f2])
                snr_post_w2 = np.matmul(np.conjugate(beamformer[c, :, f2]).T, R_n[:, :, f2])
                snr_post_w2 = np.matmul(snr_post_w2, beamformer[c, :, f2])
                w1_sum = w1_sum + snr_post_w1
                w2_sum = w2_sum + snr_post_w2
            selected_SNR[c] = np.float32(w1_sum) / np.float32(w2_sum)
        #print('snr', selected_SNR)
        max_index = np.argmax(selected_SNR)
        #print('snr indx',max_index)
        return beamformer[max_index, :, :]
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        #print('spec',enhanced_spectrum.shape)
        return bimutil.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)
