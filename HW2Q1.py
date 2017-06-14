# Based on the tutorial for using Gaussian pyramids from scikit-image.
# Link: http://scikit-image.org/docs/dev/auto_examples/transform/plot_pyramid.html
# I wrote my own function for computing the pyramids and I display a comparison of the two.

import matplotlib.pyplot as plt
import skimage.filters as scifilt
import numpy as np
from scipy.misc import imresize
from skimage import data, color, img_as_float
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from scipy import ndimage


def display_image(axis, pic, title):
    axis.imshow(pic)
    axis.axis('off')
    axis.set_title(title)


def display_image_gray(axis, pic, title):
    axis.imshow(pic, cmap='gray')
    axis.axis('off')
    axis.set_title(title)


def CreateGaussianAndLaplacianPyramid(pic, level):
    FilteredPic = scifilt.gaussian(pic, sigma=0.5, mode='nearest', multichannel=True)
    # mode='nearest' is the mode chosen by scikit-image pyramids filters.
    DownsampledPic = img_as_float(imresize(FilteredPic, 0.5))

    if 3 == pic.shape.__len__():
        pic_mod = color.rgb2gray(pic)
        filtered_mod = color.rgb2gray(FilteredPic)
    else:
        pic_mod = pic
        filtered_mod = FilteredPic

    if pic_mod.dtype != np.float64:
        filtered_mod = img_as_float(filtered_mod)

    Gaussian = [pic]
    Laplacian = [pic_mod - filtered_mod]

    if level > 1:
        TempGauss, TempLap = CreateGaussianAndLaplacianPyramid(DownsampledPic, level - 1)
        GaussianPyramid = Gaussian + TempGauss
        LaplacianPyramid = Laplacian + TempLap
    else:
        GaussianPyramid = Gaussian
        LaplacianPyramid = Laplacian

    return GaussianPyramid, LaplacianPyramid


image = data.astronaut()
rows, cols, dim = image.shape

# original code from example
GaussianPyramid = tuple(pyramid_gaussian(image, downscale=2))
LaplacianPyramid = tuple(pyramid_laplacian(image, downscale=2))

composite_image_gauss = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image_lapla = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image_gauss[:rows, :cols, :] = GaussianPyramid[0]
composite_image_lapla[:rows, :cols, :] = LaplacianPyramid[0]

i_row = 0
for p in GaussianPyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image_gauss[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

i_row = 0
for p in LaplacianPyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image_lapla[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig1, ax1 = plt.subplots(1, 2)
display_image(ax1[0], composite_image_gauss, 'scikit-image\'s pyramid_gaussian')
display_image_gray(ax1[1], color.rgb2gray(composite_image_lapla), 'scikit-image\'s pyramid_laplacian')

# My code
Gaussian, Laplacian = CreateGaussianAndLaplacianPyramid(image, 8)

composite_image2_gauss = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image2_lapla = np.zeros((rows, cols + cols // 2), dtype=np.double)
composite_image2_gauss[:rows, :cols, :] = Gaussian[0]
composite_image2_lapla[:rows, :cols] = Laplacian[0]

i_row = 0
for p in Gaussian[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image2_gauss[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

i_row = 0

for p in Laplacian[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image2_lapla[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig2, ax2 = plt.subplots(1, 2)
display_image(ax2[0], composite_image2_gauss.astype(np.float64) / 255, 'My CreateGaussianAndLaplacianPyramid: Gauss')
display_image_gray(ax2[1], composite_image2_lapla.astype(np.float64) / 255,
                   'My CreateGaussianAndLaplacianPyramid: Laplace')

# Expanding on the example: combining two pictures with different focus.
Focus1 = img_as_float(color.rgb2gray(ndimage.imread('HW2-Images\\Q1\\focus1.tif')))
Focus2 = img_as_float(color.rgb2gray(ndimage.imread('HW2-Images\\Q1\\focus2.tif')))

Focus1_padded = np.pad(Focus1, ((18,18), (155, 155)), 'edge')
Focus2_padded = np.pad(Focus2, ((18,18), (155, 155)), 'edge')

Focus1Gaussian, Focus1Laplacian = CreateGaussianAndLaplacianPyramid(Focus1_padded, 3)
Focus2Gaussian, Focus2Laplacian = CreateGaussianAndLaplacianPyramid(Focus2_padded, 3)

# Reconstructed = (Focus1Gaussian[-1] + Focus2Gaussian[-1]) / (2 * 255)  # Averaging from both pyramids.
Reconstructed = Focus1Gaussian[2]
for p in reversed(range(Focus1Laplacian.__len__()-1)):
    Reconstructed = img_as_float(imresize(Reconstructed, 2.0))
    mask = (Focus1Laplacian[p] > Focus2Laplacian[p]).astype(np.uint8)
    Reconstructed += mask*Focus1Laplacian[p] + (np.ones(mask.shape, dtype=np.uint8) - mask)*Focus2Laplacian[p]  # blended pyramid


fig3, ax3 = plt.subplots(1, 3)
display_image_gray(ax3[0], Focus1, 'Focus 1')
display_image_gray(ax3[1], Focus2, 'Focus 2')
display_image_gray(ax3[2], Reconstructed[18:(18+Focus1.shape[0]), 155:(155+Focus1.shape[1])], 'Combined focuses')
# The result is actually not so good. I used CreateGaussianAndLaplacianPyramid to reconstruct other bw images and it
# worked well, but here the reconstruction fails. I guess there are too many high frequencies, so the gaussian doesn't
# clean them so well and they are creating aliasing.
# Since I cannot reconstruct the separate photos, it is no wonder that I cannot combine the different focuses.

plt.show()
