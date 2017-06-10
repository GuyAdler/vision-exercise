import numpy as np
import scipy.signal as scisig
import scipy.ndimage.filters as scifilt
import matplotlib.pyplot as plt
from skimage import feature, color, data
from skimage.util import random_noise


OriginalImage = np.array([[4, 1, 6, 1, 3],
                          [3, 2, 7, 7, 2],
                          [2, 5, 7, 3, 7],
                          [1, 4, 7, 1, 3],
                          [0, 1, 6, 4, 4]]).astype(dtype='Float32')

mean_filter = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9

print("Original matrix:")
print(OriginalImage)

OriginalWithMeanFilter = scisig.convolve2d(OriginalImage, mean_filter)
OriginalWithMeanFilter = OriginalWithMeanFilter[1:6, 1:6]

print("\nApplying mean filter:")
print(OriginalWithMeanFilter)

# default size for this function is 3X3
OriginalWithMedianFilter = scisig.medfilt2d(OriginalImage)

print("\nApplying median filter:")
print(OriginalWithMedianFilter)

# Sigma chosen randomly.
# mode='constant' is padding with zeros (the default constant value is 0)
OriginalWithGaussianFilter = scifilt.gaussian_filter(OriginalImage, sigma=0.5, mode='constant')

print("\nApplying gaussian filter:")
print(OriginalWithGaussianFilter)

OriginalWithSobel = scifilt.sobel(OriginalImage)

print("\nApplying Sobel operator:")
print(OriginalWithSobel)

# This part with real photos
EdgesThreshold = 0.3
CameraMan = color.rgb2gray(data.camera())
Coffee = color.rgb2gray(data.coffee())
Astronaut = color.rgb2gray(data.astronaut())

fig1, axes1 = plt.subplots(3, 3)

axes1[0, 0].imshow(CameraMan, cmap='gray')
axes1[0, 0].axis('off')
axes1[0, 0].set_title("Original camera man")
axes1[0, 1].imshow(Coffee, cmap='gray')
axes1[0, 1].axis('off')
axes1[0, 1].set_title("Original coffee")
axes1[0, 2].imshow(Astronaut, cmap='gray')
axes1[0, 2].axis('off')
axes1[0, 2].set_title("Original Astronaut")

CameraManEdges = feature.canny(CameraMan, sigma=0, high_threshold=EdgesThreshold)
CameraManGaussianEdges = feature.canny(CameraMan, sigma=2, high_threshold=EdgesThreshold)
CoffeeEdges = feature.canny(Coffee, sigma=0, high_threshold=EdgesThreshold)
CoffeeGaussianEdges = feature.canny(Coffee, sigma=2, high_threshold=EdgesThreshold)
AstronautEdges = feature.canny(Astronaut, sigma=0, high_threshold=EdgesThreshold)
AstronautGaussianEdges = feature.canny(Astronaut, sigma=2, high_threshold=EdgesThreshold)

axes1[1, 0].imshow(CameraManEdges, cmap='gray')
axes1[1, 0].axis('off')
axes1[1, 0].set_title("Camera man edges")
axes1[1, 1].imshow(CoffeeEdges, cmap='gray')
axes1[1, 1].axis('off')
axes1[1, 1].set_title("Coffee edges")
axes1[1, 2].imshow(AstronautEdges, cmap='gray')
axes1[1, 2].axis('off')
axes1[1, 2].set_title("Astronaut edges")

axes1[2, 0].imshow(CameraManGaussianEdges, cmap='gray')
axes1[2, 0].axis('off')
axes1[2, 0].set_title("Camera man edges after filter")
axes1[2, 1].imshow(CoffeeGaussianEdges, cmap='gray')
axes1[2, 1].axis('off')
axes1[2, 1].set_title("Coffee edges after filter")
axes1[2, 2].imshow(AstronautGaussianEdges, cmap='gray')
axes1[2, 2].axis('off')
axes1[2, 2].set_title("Astronaut edges after filter")

# Adding noise and doing the same thing
CameraManNoise = random_noise(CameraMan, mode='gaussian')
CoffeeNoise = random_noise(Coffee, mode='gaussian')
AstronautNoise = random_noise(Astronaut, mode='gaussian')

fig2, axes2 = plt.subplots(3, 3)
axes2[0, 0].imshow(CameraManNoise, cmap='gray')
axes2[0, 0].axis('off')
axes2[0, 0].set_title("Camera man with noise")
axes2[0, 1].imshow(CoffeeNoise, cmap='gray')
axes2[0, 1].axis('off')
axes2[0, 1].set_title("Coffee with noise")
axes2[0, 2].imshow(AstronautNoise, cmap='gray')
axes2[0, 2].axis('off')
axes2[0, 2].set_title("Astronaut with noise")

CameraManEdges = feature.canny(CameraManNoise, sigma=0, high_threshold=EdgesThreshold)
CameraManGaussianEdges = feature.canny(CameraManNoise, sigma=2, high_threshold=EdgesThreshold)
CoffeeEdges = feature.canny(CoffeeNoise, sigma=0, high_threshold=EdgesThreshold)
CoffeeGaussianEdges = feature.canny(CoffeeNoise, sigma=2, high_threshold=EdgesThreshold)
AstronautEdges = feature.canny(AstronautNoise, sigma=0, high_threshold=EdgesThreshold)
AstronautGaussianEdges = feature.canny(AstronautNoise, sigma=2, high_threshold=EdgesThreshold)

axes2[1, 0].imshow(CameraManEdges, cmap='gray')
axes2[1, 0].axis('off')
axes2[1, 0].set_title("Camera man edges")
axes2[1, 1].imshow(CoffeeEdges, cmap='gray')
axes2[1, 1].axis('off')
axes2[1, 1].set_title("Coffee edges")
axes2[1, 2].imshow(AstronautEdges, cmap='gray')
axes2[1, 2].axis('off')
axes2[1, 2].set_title("Astronaut edges")

axes2[2, 0].imshow(CameraManGaussianEdges, cmap='gray')
axes2[2, 0].axis('off')
axes2[2, 0].set_title("Camera man edges after filter")
axes2[2, 1].imshow(CoffeeGaussianEdges, cmap='gray')
axes2[2, 1].axis('off')
axes2[2, 1].set_title("Coffee edges after filter")
axes2[2, 2].imshow(AstronautGaussianEdges, cmap='gray')
axes2[2, 2].axis('off')
axes2[2, 2].set_title("Astronaut edges after filter")

fig3, axes3 = plt.subplots(2, 3)

CameraManFiltered075 = scifilt.gaussian_filter(CameraMan, sigma=0.75)
CameraManFiltered250 = scifilt.gaussian_filter(CameraMan, sigma=2.5)
CoffeeFiltered075 = scifilt.gaussian_filter(Coffee, sigma=0.75)
CoffeeFiltered250 = scifilt.gaussian_filter(Coffee, sigma=2.5)
AstronautFiltered075 = scifilt.gaussian_filter(Astronaut, sigma=0.75)
AstronautFiltered250 = scifilt.gaussian_filter(Astronaut, sigma=2.5)

axes3[0, 0].imshow(2*CameraMan - CameraManFiltered075, cmap='gray')
axes3[0, 0].axis('off')
axes3[0, 0].set_title("Camera man, unsharped masking - sigma = 0.75")
axes3[1, 0].imshow(2*CameraMan - CameraManFiltered250, cmap='gray')
axes3[1, 0].axis('off')
axes3[1, 0].set_title("Camera man, unsharped masking - sigma = 2.5")
axes3[0, 1].imshow(2 * Coffee - CoffeeFiltered075, cmap='gray')
axes3[0, 1].axis('off')
axes3[0, 1].set_title("Coffee, unsharped masking - sigma = 0.75")
axes3[1, 1].imshow(2 * Coffee - CoffeeFiltered250, cmap='gray')
axes3[1, 1].axis('off')
axes3[1, 1].set_title("Coffee, unsharped masking - sigma = 2.5")
axes3[0, 2].imshow(2 * Astronaut - AstronautFiltered075, cmap='gray')
axes3[0, 2].axis('off')
axes3[0, 2].set_title("Astronaut, unsharped masking - sigma = 0.75")
axes3[1, 2].imshow(2 * Astronaut - AstronautFiltered250, cmap='gray')
axes3[1, 2].axis('off')
axes3[1, 2].set_title("Astronaut, unsharped masking - sigma = 2.5")


plt.show()
