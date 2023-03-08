import matplotlib.pyplot as pl
from beamformer import time_domain_CGMM_mvdr_snr_selective as cgmm_snr
from maskestimator import model as mdl, shaper, feature

import soundfile as sf
import numpy as np
import time
import tensorflow as tf
from tqdm import tqdm
from scipy.signal import stft, istft
import run_pesq_test as performance
from beamformer import util
import warnings
from scipy.signal import hilbert
warnings.filterwarnings("ignore")

##########################

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128
fs=16000

# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]



# ==========================================
# ANALYSIS PARAMETERS
# ==========================================
CHANNEL_INDEX = [1,2,3]
FFTL = 1024
SHIFT = 256
MAX_SEQUENCE = 5000

# ==========================================
# ESURAL MASL ESTIMATOR PARAMETERS
# ==========================================
LEFT_CONTEXT = 0
RIGHT_CONTEXT = 0
NUMBER_OF_SKIP_FRAME = 0

# ==========================================
# ESURAL MASL ESTIMATOR TRAINNING PARAMERTERS
# ==========================================
TRUNCATE_GRAD = 1
IS_DEBUG_SHOW_PREDICT_MASK = False
WEIGHT_PATH = r'model\model'

NUMBER_OF_STACK = LEFT_CONTEXT + RIGHT_CONTEXT + 1

OPERATION = 'median'
RECURRENT_CELL_INIT = 0.00001  # 0.04

# ==========================================
# get model
# ==========================================
mask_estimator_generator = mdl.NeuralMaskEstimation(TRUNCATE_GRAD,
                                                    NUMBER_OF_STACK,
                                                    0.1,
                                                    FFTL // 2 + 1,
                                                    recurrent_init=RECURRENT_CELL_INIT)

mask_estimator = mask_estimator_generator.get_model(is_stateful=True, is_show_detail=True, is_adapt=False)

mask_estimator = mask_estimator_generator.load_weight_param(mask_estimator, WEIGHT_PATH)
# ==========================================
# predicting data shaper
# ==========================================
data_shaper = shaper.Shape_data(LEFT_CONTEXT,
                                RIGHT_CONTEXT,
                                TRUNCATE_GRAD,
                                NUMBER_OF_SKIP_FRAME)

# ==========================================
# get features
# ==========================================
feature_extractor = feature.Feature(SAMPLING_FREQUENCY, FFTL, SHIFT)


#=====================
# Load data
sig1 = sf.read('Input_metro_noise/noisy_mic_1.wav',dtype='float32')[0]
sig2 = sf.read('Input_metro_noise/noisy_mic_2.wav',dtype='float32')[0]
sig3 = sf.read('Input_metro_noise/noisy_mic_3.wav',dtype='float32')[0]
ll = min(len(sig1),len(sig2),len(sig3))

print('Total Signal duration',ll//fs ,'s' )
print('How many seconds would you like to process in each iteration?')
ss = int(input())
sft = ss*(ll//(ll//fs))
ll = sft*(ll//sft)
sig1=sig1[0:ll]
sig2=sig2[0:ll]
sig3=sig3[0:ll]
audio = sig1 + sig2 + sig3




# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio


# create buffer
in_buffer = np.zeros((block_len)).astype('float32')+1e-9
out_buffer = np.zeros((block_len)).astype('float32')+1e-9

in_buffer1 = np.zeros((block_len)).astype('float32')+1e-9
out_buffer1 = np.zeros((block_len)).astype('float32')+1e-9

in_buffer2 = np.zeros((block_len)).astype('float32')+1e-9
out_buffer2 = np.zeros((block_len)).astype('float32')+1e-9

in_buffer3 = np.zeros((block_len)).astype('float32')+1e-9
out_buffer3 = np.zeros((block_len)).astype('float32')+1e-9







#print('sft',sft)
out_file = np.zeros(sft).astype('float32')
out_file1 = np.zeros(sft).astype('float32')
out_file2 = np.zeros(sft).astype('float32')
out_file3 = np.zeros(sft).astype('float32')
output = np.zeros(ll).astype('float32')
# print('len',ll)
time_array = []
for itr in tqdm(range(0, ll, sft)):
    start_time = time.time()
    noisy = audio[itr:itr+sft]
    speech1 = sig1[itr:itr+sft]
    speech2 = sig2[itr:itr+sft]
    speech3 = sig3[itr:itr + sft]
    #print('sft',sft)
    num_blocks = sft // block_shift
    # print(sft,num_blocks)

    for ii in range(0, len(CHANNEL_INDEX)):
        if ii == 0:
            speech = speech1
        elif ii == 1:
            speech = speech2
        else:
            speech = speech3
        # print('speech',speech.shape)

        noisy_spectrogram = feature_extractor.get_feature(speech)
        # print('spec',noisy_spectrogram.shape)
        noisy_spectrogram = (np.flipud(noisy_spectrogram))
        noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)
        # print('spec',noisy_spectrogram.shape)

        features = data_shaper.convert_for_predict(noisy_spectrogram)
        features = np.array(features)
        # print('features', features.shape)

        mask_estimator.reset_states()

        padding_feature, original_batch_size = data_shaper.get_padding_features(features)
        # print('padding features',padding_feature.shape)

        sp_mask, n_mask = mask_estimator.predict(padding_feature, batch_size=MAX_SEQUENCE)
        sp_mask = sp_mask[:original_batch_size, :]
        n_mask = n_mask[:original_batch_size, :]

        if IS_DEBUG_SHOW_PREDICT_MASK == True:
            pl.subplot(len(CHANNEL_INDEX), 2, ((ii + 1) * 2) - 1)
            pl.imshow(((n_mask).T), aspect='auto')
            pl.subplot(len(CHANNEL_INDEX), 2, ((ii + 1) * 2))
            pl.imshow(((sp_mask).T), aspect='auto')

        if ii == 0:
            aa, bb = np.shape(n_mask)
            n_median = np.zeros((aa, bb, len(CHANNEL_INDEX)))
            sp_median = np.zeros((aa, bb, len(CHANNEL_INDEX)))

            n_median[:, :, ii] = n_mask
            sp_median[:, :, ii] = sp_mask
            dump_speech = np.zeros((len(speech), len(CHANNEL_INDEX)))
            dump_speech[:, ii] = speech
        else:
            n_median[:, :, ii] = n_mask
            sp_median[:, :, ii] = sp_mask
            dump_speech[:, ii] = speech

    if OPERATION == 'median':
        n_median_s = np.median(n_median, axis=2)
        sp_median_s = np.median(sp_median, axis=2)
    else:
        n_median_s = np.mean(n_median, axis=2)
        sp_median_s = np.mean(sp_median, axis=2)

    if IS_DEBUG_SHOW_PREDICT_MASK == True:
        pl.figure()
        pl.subplot(3, 1, 1)
        pl.imshow((np.log10(noisy_spectrogram[:, TRUNCATE_GRAD // 2:- TRUNCATE_GRAD // 2] ** 2) * 10), aspect='auto')
        pl.subplot(3, 1, 2)
        pl.imshow(((n_median_s.T)), aspect="auto")
        pl.title('noise mask')
        pl.subplot(3, 1, 3)
        pl.imshow(((sp_median_s.T)), aspect="auto")
        pl.title('speech mask')
        pl.show()

    dump_speech = np.zeros((len(speech1), len(CHANNEL_INDEX)))
    dump_speech[:, 0] = speech1
    dump_speech[:, 1] = speech2
    dump_speech[:,2] = speech3

    #print('mask',sp_median_s.shape, sp_median_s.max(), sp_median_s.min())
    sp_median_s = tf.signal.overlap_and_add(sp_median_s,SHIFT)
    n_median_s = tf.signal.overlap_and_add(n_median_s,SHIFT)
    #print('mask', sp_median_s.shape)
    sp_median_s = tf.signal.frame(sp_median_s, FFTL, SHIFT)
    n_median_s = tf.signal.frame(n_median_s, FFTL, SHIFT)
    #print('mask', sp_median_s.shape)

    cgmm_bf_snr = cgmm_snr.complexGMM_mvdr(SAMPLING_FREQUENCY, FFTL, SHIFT)
    tmp_complex_spectrum, R_x, R_n, tt, nn = cgmm_bf_snr.get_spatial_correlation_matrix_from_mask_for_LSTM(
        dump_speech.T,
        speech_mask= np.array( sp_median_s).T,
        noise_mask=np.array(n_median_s).T,
        less_frame=0)

    #print('complex', tmp_complex_spectrum.shape)
    #print('R_x', R_x.shape, R_x)
    selected_beamformer = cgmm_bf_snr.get_mvdr_beamformer_by_maxsnr(R_x, R_n)
    #print('beamformer', selected_beamformer.shape, selected_beamformer)
    enhan_speech2 = cgmm_bf_snr.apply_beamformer(selected_beamformer, tmp_complex_spectrum)
    enhan_speech2 = enhan_speech2 / np.max(np.abs(enhan_speech2)) * 0.65
    #print('enhan',enhan_speech2.shape,enhan_speech2)


    for idx in range(num_blocks-3):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = enhan_speech2[idx * block_shift:(idx * block_shift) + block_shift]
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        # print('in block',in_block.shape)
        out_block = infer(tf.constant(in_block))['conv1d_1']
        out_block = np.array(out_block)
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block/4.0)
        # write block to output file
        out_file[(idx) * block_shift:((idx) * block_shift) + block_shift] = out_buffer[:block_shift]

    output[itr:itr+sft] = out_file
    time_array.append(time.time() - start_time)




#prefix = 'output/TD_frame_by_frame_{}sec.wav'
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

prefix = 'Output_metro_LSTM_DTLN/LSTM_mask_TD_BF+DTLN_{}sec.wav'
d = { "{}": str(ss)}

prefix = replace_all(prefix, d)
#print(prefix)

sf.write(prefix, output, fs)


print('\n\n\t\t\t\t\t\t\t Performance')
print('-' * 100)
clean = sf.read('Output_metro_LSTM_DTLN/clean.wav', dtype='float32')[0]
noisy = sf.read('Output_metro_LSTM_DTLN/noisy.wav',dtype='float32')[0]

title = 'Noisy'
enhanced = noisy
performance.score(enhanced,clean,fs,(title.replace('{}', str(ss))))

title = 'LSTM_mask_TD_BF+DTLN_{}sec'
enhanced = output
performance.score(enhanced,clean,fs,(title.replace('{}', str(ss))))

print('\n\n\t\t\t\t\t\t\tProcessing Time [ms]:')
print('-' * 100)
print(np.mean(np.stack(time_array)) * 1000)
print('Processing finished.')
