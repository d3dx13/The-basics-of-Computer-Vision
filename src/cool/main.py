import cv2
import os

images = {}
for image_path in os.listdir("data/original"):
    images[image_path] = cv2.imread(f"data/original/{image_path}")

for name, image in images.items():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.medianBlur(image, 5)
    channels = cv2.split(image)
    for i in range(len(channels)):
        channels[i] = clahe.apply(channels[i])
    image_CLAHE = cv2.merge(channels)
    cv2.imwrite(f"data/out/{name}", image_CLAHE)
