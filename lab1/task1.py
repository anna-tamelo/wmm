import numpy as np
import matplotlib.pyplot as plt
import time

N = 8
T = 0.5
dt = T / N
t = np.arange(N) * dt
s = np.sin(4 * np.pi * t)

plt.figure()
plt.stem(t, s)
plt.title('Sampled Signal s[n]')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

X = np.fft.fft(s)
freq = np.fft.fftfreq(N, dt)

plt.figure()
plt.stem(freq, np.abs(X))
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()

plt.figure()
plt.stem(freq, np.angle(X))
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.show()

energy_time = np.sum(np.abs(s)**2)
energy_freq = (1/N) * np.sum(np.abs(X)**2)
print(f"Energy in time domain: {energy_time:.6f}")
print(f"Energy in frequency domain: {energy_freq:.6f}")

Ns = [2**l for l in range(1, 13)]
times = []
for N in Ns:
    x = np.random.rand(N)
    start = time.perf_counter()
    np.fft.fft(x)
    end = time.perf_counter()
    times.append(end - start)

plt.figure()
plt.plot(Ns, times)
plt.title('FFT Computation Time vs Number of Samples')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Time (seconds)')
plt.show()

for N, t_elapsed in zip(Ns, times):
    print(f"N={N:5d}, FFT time={t_elapsed:.6e} s")
