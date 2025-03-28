import cv2
import numpy as np

def calcPSNR(img1, img2):
    imax = 255.**2
    mse = ((img1.astype(np.float64) - img2.astype(np.float64))**2).sum() / img1.size
    if mse == 0:
        return 100
    return 10.0 * np.log10(imax / mse)

path = "./"
original = cv2.imread(path + "susie_col.png", cv2.IMREAD_COLOR)
img_gauss_noise = cv2.imread(path + "susie_col_noise.png", cv2.IMREAD_COLOR)
img_impulse_noise = cv2.imread(path + "susie_col_inoise1.png", cv2.IMREAD_COLOR)

if original is None or img_gauss_noise is None or img_impulse_noise is None:
    exit()

kernel_sizes = [3, 5, 7]
results = {
    'gauss_noise': {
        'gaussian_filter': {},
        'median_filter': {}
    },
    'impulse_noise': {
        'gaussian_filter': {},
        'median_filter': {}
    }
}

for k in kernel_sizes:
    g_g = cv2.GaussianBlur(img_gauss_noise, (k, k), 0)
    g_psnr = calcPSNR(original, g_g)
    results['gauss_noise']['gaussian_filter'][k] = g_psnr
    cv2.imshow(f"GaussNoise_Gaussian_{k}", g_g)
    cv2.imwrite(f"GaussNoise_Gaussian_{k}.png", g_g)

    g_m = cv2.medianBlur(img_gauss_noise, k)
    m_psnr = calcPSNR(original, g_m)
    results['gauss_noise']['median_filter'][k] = m_psnr
    cv2.imshow(f"GaussNoise_Median_{k}", g_m)
    cv2.imwrite(f"GaussNoise_Median_{k}.png", g_m)

for k in kernel_sizes:
    i_g = cv2.GaussianBlur(img_impulse_noise, (k, k), 0)
    ig_psnr = calcPSNR(original, i_g)
    results['impulse_noise']['gaussian_filter'][k] = ig_psnr
    cv2.imshow(f"ImpulseNoise_Gaussian_{k}", i_g)
    cv2.imwrite(f"ImpulseNoise_Gaussian_{k}.png", i_g)

    i_m = cv2.medianBlur(img_impulse_noise, k)
    im_psnr = calcPSNR(original, i_m)
    results['impulse_noise']['median_filter'][k] = im_psnr
    cv2.imshow(f"ImpulseNoise_Median_{k}", i_m)
    cv2.imwrite(f"ImpulseNoise_Median_{k}.png", i_m)

cv2.waitKey(0)
cv2.destroyAllWindows()

for noise_type in results:
    print(noise_type)
    for filt_type in results[noise_type]:
        for k in kernel_sizes:
            print(f"{filt_type}, {k}x{k}, PSNR = {results[noise_type][filt_type][k]:.2f}")
