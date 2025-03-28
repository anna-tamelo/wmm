import cv2
import numpy as np

img_in = cv2.imread("images/susie_col.png", cv2.IMREAD_GRAYSCALE)
img_blur = cv2.GaussianBlur(img_in, (3, 3), 0)
lap = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)

weights = [-0.2, -0.4, -0.6, -0.8, -1.0]

for w in weights:
    img_in_f = img_in.astype(np.float32)
    lap_f = lap.astype(np.float32)

    sharp_float = cv2.addWeighted(img_in_f, 1.0, lap_f, w, 0.0)

    sharp_8u = cv2.convertScaleAbs(sharp_float)

    cv2.imshow(f"Sharp (w={w})", sharp_8u)
    cv2.imwrite(f"susie_sharp_w{w}.png", sharp_8u)

cv2.waitKey(0)
cv2.destroyAllWindows()
