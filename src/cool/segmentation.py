import cv2
import numpy as np
import skimage.filters
import skimage.morphology
import os

image = cv2.imread("data/super/image_entropy.png")

image_gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0)
image_mask = image_gray != 0

"""
# Открытие (отмыкание, размыкание, раскрытие): (𝐴 ⊖ 𝐵) ⊕ 𝐵
selem = skimage.morphology.disk(7)
image_mask = skimage.morphology.erosion(image_mask, selem)
image_mask = skimage.morphology.dilation(image_mask, selem)
"""
# Закрытие (замыкание): (𝐴 ⊕ 𝐵) ⊖ 𝐵
selem = skimage.morphology.disk(21)
image_mask = skimage.morphology.dilation(image_mask, selem)
image_mask = skimage.morphology.erosion(image_mask, selem)


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", np.uint8(image_mask * 255))
cv2.waitKey(0)