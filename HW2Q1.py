import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize
import scipy.ndimage.filters as scifilt
from skimage import color


def DispGrayImage(axis, pic, title):
    axis.imshow(pic, cmap='gray', shape=pic.shape)
    axis.axis('off')
    axis.set_title(title)


def CreateGaussianAndLaplacianPyramid(pic, level):
    FilteredPic = scifilt.gaussian_filter(pic, sigma=1, mode='constant')
    DownsampledPic = imresize(FilteredPic, 0.5)

    Gaussian = [pic]
    Laplacian = [pic - FilteredPic]

    if level > 1:
        TempGauss, TempLap = CreateGaussianAndLaplacianPyramid(DownsampledPic, level - 1)
        GaussianPyramid = Gaussian + TempGauss
        LaplacianPyramid = Laplacian + TempLap
    else:
        GaussianPyramid = Gaussian
        LaplacianPyramid = Laplacian

    return GaussianPyramid, LaplacianPyramid

MandrillOriginal = color.rgb2gray(ndimage.imread('HW2-Images\\Q1\\mandril.tif'))
ToucanOriginal = color.rgb2gray(ndimage.imread('HW2-Images\\Q1\\toucan.tif'))

fig1, axes1 = plt.subplots(5, 2)

GaussianPyramid, LaplacianPyramid = CreateGaussianAndLaplacianPyramid(MandrillOriginal, 5)
for k in range(4):
    DispGrayImage(axes1[k, 0], GaussianPyramid[k], "Mandrill gaussian " + str(k))
    DispGrayImage(axes1[k, 1], LaplacianPyramid[k], "Mandrill laplacian " + str(k))

plt.show()