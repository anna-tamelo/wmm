import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/susie_col.png")
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
image_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
cv2.imwrite("images/susie_col_eq.png", image_eq)

orig_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
eq_ycrcb = cv2.cvtColor(image_eq, cv2.COLOR_BGR2YCrCb)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Y before")
plt.hist(orig_ycrcb[:, :, 0].ravel(), bins=256, range=(0, 256))
plt.subplot(1, 2, 2)
plt.title("Y after")
plt.hist(eq_ycrcb[:, :, 0].ravel(), bins=256, range=(0, 256))
plt.tight_layout()
plt.show()
