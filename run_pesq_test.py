from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio

import os



def plot_spectrogram(sig, title="Spectrogram", xlim=None,i=0):
    sig=torch.from_numpy(sig)
    sig = sig.to(torch.double)
    N_FFT = 512
    N_HOP = 128
    stft = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=N_HOP,
        power=None,
    )
    stft = stft(sig)
    magnitude = stft.abs()
    spectrogram = 20*torch.log10(magnitude + 1e-8).numpy()
    return spectrogram


def score(deg,ref,rate,title):
    #rate, deg = wavfile.read(enhanced)
    #rate, ref = wavfile.read(clean)
    #print('fs',rate)

    #print(ref.shape,deg.shape)
    ll = min(ref.shape[0],deg.shape[0])
    #print(ll)
    ref=ref[0:ll]
    deg=deg[0:ll]
    wb = pesq(rate, ref, deg, 'wb')
    nb = pesq(rate, ref, deg, 'nb')
    stoic = stoi(ref, deg, rate, extended=False)
    print(f'{title} = wide-band PESQ: {wb:.2f} | narrow-band PESQ: {nb:.2f} | STOI: {stoic:.2f}')
    print('-' * 100)
    ref=np.reshape(ref,(-1,1)).T
    deg=np.reshape(deg,(-1,1)).T

    figure, axis = plt.subplots(2, 1)
    spectrogram =plot_spectrogram(ref)
    img = axis[0].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[0].set_title('clean signal')

    spectrogram = plot_spectrogram(deg)
    img = axis[1].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[1].set_title(title)

    plt.colorbar(img, ax=axis)
    plt.show()

def score_2(enhanced,clean,title):
    rate, deg = wavfile.read(enhanced)
    rate, ref = wavfile.read(clean)
    ll = min(ref.shape[0], deg.shape[0])
    # print(ll)
    ref = ref[0:ll]
    deg = deg[0:ll]
    wb = pesq(rate, ref, deg, 'wb')
    nb = pesq(rate, ref, deg, 'nb')
    stoic = stoi(ref, deg, rate, extended=False)
    print(f'wide-band PESQ: {wb:.2f} | narrow-band PESQ: {nb:.2f} | STOI: {stoic:.2f}')
    print('-' * 60)

    '''ref = np.reshape(ref, (-1, 1)).T
    deg = np.reshape(deg, (-1, 1)).T

    figure, axis = plt.subplots(2, 1)
    spectrogram = plot_spectrogram(ref)
    img = axis[0].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[0].set_title('clean signal')

    spectrogram = plot_spectrogram(deg)
    img = axis[1].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[1].set_title(title)

    plt.colorbar(img, ax=axis)
    plt.show()'''

if __name__=="__main__":
    '''print('opt 1 for Individually enhancing both signals in DTLN to get beamforer weight for the first frame')
    print('opt 2 for Adding both signals and then enhancing in DTLN to get beamforer weight for the first frame')
    print('opt 3 for  Only 1 signal enhancing in DTLN to get beamforer weight for the first frame')'''

    folder = r"Output_metro_LSTM_DTLN/"
    clean = r'Output_metro_LSTM_DTLN/clean.wav'
    # score(enhanced,clean)
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            print(name[:-4])
            score_2(os.path.join(root, name),clean,name[:-4])




