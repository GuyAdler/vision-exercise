import numpy as np
import scipy.signal as scisig
import scipy.ndimage.filters as scifilt
import matplotlib.pyplot as plt
from skimage import feature, color
from skimage.util import random_noise
from scipy import ndimage

OriginalImage = np.array([[4,1,6,1,3],
                          [3,2,7,7,2],
                          [2,5,7,3,7],
                          [1,4,7,1,3],
                          [0,1,6,4,4]]).astype(dtype='Float32')

mean_filter = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]]) / 9

print("Original matrix:")
print(OriginalImage)

OriginalWithMeanFilter = scisig.convolve2d(OriginalImage,mean_filter)

print("\nApplying mean filter:")
print(OriginalWithMeanFilter)

# default size for this function is 3X3
OriginalWithMedianFilter = scisig.medfilt2d(OriginalImage)

print("\nApplying median filter:")
print(OriginalWithMedianFilter)

# Sigma chosen randomly.
OriginalWithGaussianFilter = scifilt.gaussian_filter(OriginalImage,sigma=0.5)

print("\nApplying gaussian filter:")
print(OriginalWithGaussianFilter)

OriginalWithSobel = scifilt.sobel(OriginalImage)

print("\nApplying Sobel operator:")
print(OriginalWithSobel)

# This part with real photos
EdgesThreshold = 0.3
CameraMan = color.rgb2gray(ndimage.imread("HW1-Images\Q1\cameraman.jpg"))
House = color.rgb2gray(ndimage.imread("HW1-Images\Q1\house.jpg"))
Lena = color.rgb2gray(ndimage.imread("HW1-Images\Q1\lena.jpg"))

fig1, axes1 = plt.subplots(3, 3)

axes1[0, 0].imshow(CameraMan, cmap='gray')
axes1[0, 0].axis('off')
axes1[0, 0].set_title("Original camera man")
axes1[0, 1].imshow(House, cmap='gray')
axes1[0, 1].axis('off')
axes1[0, 1].set_title("Original house")
axes1[0, 2].imshow(Lena, cmap='gray')
axes1[0, 2].axis('off')
axes1[0, 2].set_title("Original Lena")

CameraManEdges = feature.canny(CameraMan,sigma=0,high_threshold=EdgesThreshold)
CameraManGaussianEdges = feature.canny(CameraMan,sigma=2,high_threshold=EdgesThreshold)
HouseEdges = feature.canny(House,sigma=0,high_threshold=EdgesThreshold)
HouseGaussianEdges = feature.canny(House,sigma=2,high_threshold=EdgesThreshold)
LenaEdges = feature.canny(Lena,sigma=0,high_threshold=EdgesThreshold)
LenaGaussianEdges = feature.canny(Lena,sigma=2,high_threshold=EdgesThreshold)

axes1[1, 0].imshow(CameraManEdges, cmap='gray')
axes1[1, 0].axis('off')
axes1[1, 0].set_title("Camera man edges")
axes1[1, 1].imshow(HouseEdges, cmap='gray')
axes1[1, 1].axis('off')
axes1[1, 1].set_title("House edges")
axes1[1, 2].imshow(LenaEdges, cmap='gray')
axes1[1, 2].axis('off')
axes1[1, 2].set_title("Lena edges")

axes1[2, 0].imshow(CameraManGaussianEdges, cmap='gray')
axes1[2, 0].axis('off')
axes1[2, 0].set_title("Camera man edges after filter")
axes1[2, 1].imshow(HouseGaussianEdges, cmap='gray')
axes1[2, 1].axis('off')
axes1[2, 1].set_title("House edges after filter")
axes1[2, 2].imshow(LenaGaussianEdges, cmap='gray')
axes1[2, 2].axis('off')
axes1[2, 2].set_title("Lena edges after filter")

# Adding noise and doing the same thing
CameraManNoise = random_noise(CameraMan, mode='gaussian')
HouseNoise = random_noise(House, mode='gaussian')
LenaNoise = random_noise(Lena, mode='gaussian')

fig2, axes2 = plt.subplots(3,3)
axes2[0, 0].imshow(CameraManNoise, cmap='gray')
axes2[0, 0].axis('off')
axes2[0, 0].set_title("Camera man with noise")
axes2[0, 1].imshow(HouseNoise, cmap='gray')
axes2[0, 1].axis('off')
axes2[0, 1].set_title("House with noise")
axes2[0, 2].imshow(LenaNoise, cmap='gray')
axes2[0, 2].axis('off')
axes2[0, 2].set_title("Lena with noise")

CameraManEdges = feature.canny(CameraManNoise,sigma=0,high_threshold=EdgesThreshold)
CameraManGaussianEdges = feature.canny(CameraManNoise,sigma=2,high_threshold=EdgesThreshold)
HouseEdges = feature.canny(HouseNoise,sigma=0,high_threshold=EdgesThreshold)
HouseGaussianEdges = feature.canny(HouseNoise,sigma=2,high_threshold=EdgesThreshold)
LenaEdges = feature.canny(LenaNoise,sigma=0,high_threshold=EdgesThreshold)
LenaGaussianEdges = feature.canny(LenaNoise,sigma=2,high_threshold=EdgesThreshold)

axes2[1, 0].imshow(CameraManEdges, cmap='gray')
axes2[1, 0].axis('off')
axes2[1, 0].set_title("Camera man edges")
axes2[1, 1].imshow(HouseEdges, cmap='gray')
axes2[1, 1].axis('off')
axes2[1, 1].set_title("House edges")
axes2[1, 2].imshow(LenaEdges, cmap='gray')
axes2[1, 2].axis('off')
axes2[1, 2].set_title("Lena edges")

axes2[2, 0].imshow(CameraManGaussianEdges, cmap='gray')
axes2[2, 0].axis('off')
axes2[2, 0].set_title("Camera man edges after filter")
axes2[2, 1].imshow(HouseGaussianEdges, cmap='gray')
axes2[2, 1].axis('off')
axes2[2, 1].set_title("House edges after filter")
axes2[2, 2].imshow(LenaGaussianEdges, cmap='gray')
axes2[2, 2].axis('off')
axes2[2, 2].set_title("Lena edges after filter")

fig3, axes3 = plt.subplots(2,3)

CameraManFiltered075 = scifilt.gaussian_filter(CameraMan, sigma=0.75)
CameraManFiltered250 = scifilt.gaussian_filter(CameraMan, sigma=2.5)
HouseFiltered075 = scifilt.gaussian_filter(House, sigma=0.75)
HouseFiltered250 = scifilt.gaussian_filter(House, sigma=2.5)
LenaFiltered075 = scifilt.gaussian_filter(Lena, sigma=0.75)
LenaFiltered250 = scifilt.gaussian_filter(Lena, sigma=2.5)

axes3[0, 0].imshow(2*CameraMan - CameraManFiltered075, cmap='gray')
axes3[0, 0].axis('off')
axes3[0, 0].set_title("Camera man, unsharped masking - sigma = 0.75")
axes3[1, 0].imshow(2*CameraMan - CameraManFiltered250, cmap='gray')
axes3[1, 0].axis('off')
axes3[1, 0].set_title("Camera man, unsharped masking - sigma = 2.5")
axes3[0, 1].imshow(2*House - HouseFiltered075, cmap='gray')
axes3[0, 1].axis('off')
axes3[0, 1].set_title("House, unsharped masking - sigma = 0.75")
axes3[1, 1].imshow(2*House - HouseFiltered250, cmap='gray')
axes3[1, 1].axis('off')
axes3[1, 1].set_title("House, unsharped masking - sigma = 2.5")
axes3[0, 2].imshow(2*Lena - LenaFiltered075, cmap='gray')
axes3[0, 2].axis('off')
axes3[0, 2].set_title("Lena, unsharped masking - sigma = 0.75")
axes3[1, 2].imshow(2*Lena - LenaFiltered250, cmap='gray')
axes3[1, 2].axis('off')
axes3[1, 2].set_title("Lena, unsharped masking - sigma = 2.5")


plt.show()
