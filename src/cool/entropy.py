import cv2
import numpy as np
import skimage.filters
import skimage.morphology
import os

image = cv2.imread("data/original/Solarsystemscope_texture_8k_earth_nightmap.jpg")

image_gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0)

image_entropy = np.float32(skimage.filters.rank.entropy(image_gray, skimage.morphology.disk(7)))
image_entropy -= np.min(image_entropy)
image_entropy /= np.max(image_entropy)

cv2.imwrite("data/super/image_entropy.png", cv2.cvtColor(np.uint8(np.round(image_entropy * 255)), cv2.COLOR_GRAY2BGR))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", image_entropy)
cv2.waitKey(0)