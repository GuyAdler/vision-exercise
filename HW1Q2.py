from skimage import feature, color, filters
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.ndimage.filters as scifilt


def DispGrayImage(axis, pic, title):
    axis.imshow(pic, cmap='gray')
    axis.axis('off')
    axis.set_title(title)


def DispColorImage(axis, pic, title):
    axis.imshow(pic)
    axis.axis('off')
    axis.set_title(title)

# Paragraph A
Pic1 = ndimage.imread("HW1-Images\Q2\colorful1.jpg")
Pic2 = ndimage.imread("HW1-Images\Q2\colorful2.jpg")
Pic3 = ndimage.imread("HW1-Images\Q2\colorful3.jpg")

fig1, axes1 = plt.subplots(4, 3)

DispColorImage(axes1[0, 0], Pic1, "Original 1")
DispColorImage(axes1[0, 1], Pic2, "Original 2")
DispColorImage(axes1[0, 2], Pic3, "Original 3")

Pic1Gray = color.rgb2gray(Pic1)
Pic2Gray = color.rgb2gray(Pic2)
Pic3Gray = color.rgb2gray(Pic3)

sigma = 2
Threshold = 0.2

Pic1Canny = feature.canny(Pic1Gray, sigma=sigma, high_threshold=Threshold)
Pic2Canny = feature.canny(Pic2Gray, sigma=sigma, high_threshold=Threshold)
Pic3Canny = feature.canny(Pic3Gray, sigma=sigma, high_threshold=Threshold)

Pic1Sobel = filters.sobel(Pic1Gray)
Pic2Sobel = filters.sobel(Pic2Gray)
Pic3Sobel = filters.sobel(Pic3Gray)

Pic1GaussLaplace = scifilt.gaussian_laplace(Pic1Gray, sigma=sigma)
Pic2GaussLaplace = scifilt.gaussian_laplace(Pic2Gray, sigma=sigma)
Pic3GaussLaplace = scifilt.gaussian_laplace(Pic3Gray, sigma=sigma)

DispGrayImage(axes1[1, 0], Pic1Canny, "Canny 1 " + r"$\sigma = $" + str(sigma))
DispGrayImage(axes1[1, 1], Pic2Canny, "Canny 2 " + r"$\sigma = $" + str(sigma))
DispGrayImage(axes1[1, 2], Pic3Canny, "Canny 3 " + r"$\sigma = $" + str(sigma))

DispGrayImage(axes1[2, 0], Pic1Sobel > 0.07, "Sobel 1")
DispGrayImage(axes1[2, 1], Pic2Sobel > 0.07, "Sobel 2")
DispGrayImage(axes1[2, 2], Pic3Sobel > 0.07, "Sobel 3")

DispGrayImage(axes1[3, 0], Pic1GaussLaplace, "LoG 1")
DispGrayImage(axes1[3, 1], Pic2GaussLaplace, "LoG 2")
DispGrayImage(axes1[3, 2], Pic3GaussLaplace, "LoG 3")

plt.show()

# Paragraph B
