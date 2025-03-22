import numpy as np
import matplotlib.pyplot as plt

A = 2
N = 88
def s(n):
    return A * np.cos((2 * np.pi * n) / N)
n0_values = [0, N//4, N//2, 3*N//4]

for n0 in n0_values:
    signal = [s(n - int(n0)) for n in range(N)]
    X = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, d=1)

    X[np.abs(X) < 1e-6] = 0

    plt.figure()
    plt.stem(freq, np.abs(X))
    plt.title(f'Amplitude Spectrum, n0 = {n0}')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('Magnitude')
    plt.show()

    plt.figure()
    plt.stem(freq, np.angle(X).round(2))
    plt.title(f'Phase Spectrum, n0 = {n0}')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('Phase (radians)')
    plt.show()

    print(f"Completed FFT for shift n0 = {n0}")
