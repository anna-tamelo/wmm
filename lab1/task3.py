import numpy as np
import matplotlib.pyplot as plt

A = 4
N = 12
def s1(n):
    return A*((n % N) / N)
N0 = [0, N, 4*N, 9*N]

for n_val in N0:
    signal = [s1(n) for n in range(N)]
    M = len(signal) + n_val
    dft = np.fft.fft(signal, n=M)

    sig_amp = dft.copy()
    sig_amp[np.abs(dft) < 1e-16] = 0

    sig_faz = dft.copy()
    sig_faz[np.abs(dft) < 1e-8] = 0

    freq = np.fft.fftshift(np.fft.fftfreq(M, d=1))
    sig_amp = np.fft.fftshift(sig_amp)
    sig_faz = np.fft.fftshift(sig_faz)

    plt.figure()
    plt.stem(freq, np.abs(sig_amp))
    plt.title(f'Amplitude Spectrum, zero-pad = {n_val}')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('Magnitude')
    plt.show()

    plt.figure()
    plt.stem(freq, np.angle(sig_faz))
    plt.title(f'Phase Spectrum, zero-pad = {n_val}')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('Phase (radians)')
    plt.show()