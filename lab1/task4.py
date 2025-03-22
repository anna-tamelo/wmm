import numpy as np
import matplotlib.pyplot as plt

a1, f1 = 0.2, 2000
a2, f2 = 0.5, 6000
a3, f3 = 0.6, 10000
fs = 48000

N_list = [2048, int(3/2 * 2048)]

for N in N_list:
    t = np.arange(N) / fs
    s = a1 * np.sin(2*np.pi*f1*t) + a2 * np.sin(2*np.pi*f2*t) + a3 * np.sin(2*np.pi*f3*t)

    X = np.fft.fft(s)
    PSD = (np.abs(X)**2) / N
    freq = np.fft.fftfreq(N, d=1/fs)

    freq_shift = np.fft.fftshift(freq)
    PSD_shift = np.fft.fftshift(PSD)

    plt.figure()
    plt.plot(freq_shift, PSD_shift)
    plt.title(f'Power Spectral Density (N = {N})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.xlim(0, fs/2)  # show positive frequencies only
    plt.show()

    print(f"Completed PSD plot for N = {N}")
